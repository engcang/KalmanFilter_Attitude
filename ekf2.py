#!/usr/bin/env python

#author: mason

import rospy
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage
from mav_msgs.msg import RateThrust

from math import atan2,pow,sqrt,sin,cos,tan
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix, quaternion_multiply

import time
import numpy as np
from numpy.linalg import inv

import matplotlib.pyplot as plt
import matplotlib.animation as ani
from collections import deque

import sys
import signal
def signal_handler(signal, frame): # ctrl + c -> exit program
        print('You pressed Ctrl+C!')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

global G
G=9.81007
global d2r
d2r=np.pi/180
global r2d
r2d=180/np.pi
global tg
global prev_tg
tg=0
prev_tg=0
global first
first = 0 

class ekf():
    def __init__(self):
        rospy.init_node('estimator', anonymous=True)
        self.imu_sub = rospy.Subscriber('/uav/sensors/imu', Imu, self.imu_callback)
        self.tf_sub = rospy.Subscriber('/tf', TFMessage, self.tf_callback)
        self.input_sub = rospy.Subscriber('/uav/input/rateThrust', RateThrust, self.input_callback)

        self.p=self.q=self.r=0
        self.prev_ax=self.prev_ay=self.prev_az=0
        self.imu_check=0

        self.euler_integrated_yaw=0
        self.prev_t=self.curr_t=self.prev_ti=self.curr_ti=0
        self.dt=self.dti=self.dtg=0
        self.input_check=0

        ''' For KF '''

        self.P=np.array([[100,0,0],[0,100,0],[0,0,100]])
        self.x=np.array([0,0,0]).reshape((3,1)) #EKF
        self.iP = 0.001 * np.array([[1,0,0,-1,0,0,0,0,0],
                                   [0,1,0,0,1,0,0,0,0],
                                   [0,0,1,0,0,1,0,0,0],
                                   [-1,0,0,1,0,0,0,0,0],
                                   [0,1,0,0,1,0,0,0,0],
                                   [0,0,1,0,0,1,0,0,0],
                                   [0,0,0,0,0,0,1,0,0],
                                   [0,0,0,0,0,0,0,1,0],
                                   [0,0,0,0,0,0,0,0,1]])
        self.ix=np.array([0,0,0,0,0,0,0,0,0]) #indirect KF

#		self._P=np.array([[100,0,0,0],[0,100,0,0],[0,0,100,0],[0,0,0,100]])
#		self._x=np.array([1,0,0,0]).reshape((4,1)) #LKF
#		self.bP=np.array([[100,0,0,0],[0,100,0,0],[0,0,100,0],[0,0,0,100]])
#		self.bx=np.array([0,0,0,0]).reshape((4,1)) #gyro_bias
#		self.baP=np.array([[100,0,0,0],[0,100,0,0],[0,0,100,0],[0,0,0,100]])
#		self.bax=np.array([0,0,0,0]).reshape((4,1)) #gyro_accel_bias


    def imu_callback(self,msg):
        self.imu_check=1
        anv=msg.angular_velocity
        lina=msg.linear_acceleration
        if self.prev_ti==0:
            self.prev_ti=time.time()
            return
        self.dti=time.time()-self.prev_ti
        self.prev_ti=time.time()


        self.p=anv.x
        self.q=anv.y
        self.r=anv.z
        if self.imu_check==1 and self.input_check==1:
            p=self.p
            q=self.q
            r=self.r
            ax=lina.x
            ay=lina.y
            az=lina.z
            ''' Filtering '''
            self.t_roll,self.t_pitch=accelerometer_arctan_estimator(ax,ay,az)

            #self._x,self._P=LKF_estimator([self.t_roll,self.t_pitch,self.euler_integrated_yaw],[p,q,r],self.dti,self._P,self._x)
            #self._xe=euler_from_quaternion([self._x[0][0],self._x[1][0],self._x[2][0],self._x[3][0]])
            self.x,self.P = EKF_estimator(np.array([self.t_roll,self.t_pitch,0]),[p,q,r],self.dti,self.P,self.x)
            self.ix,self.iP = indirect_estimator([ax,ay,az],[p,q,r],self.dti,self.iP,self.ix)
            #self.bx,self.bP=gyrobias_estimator(np.array([self.t_roll,self.t_pitch]),[p,q,r],self.dti,self.bP,self.bx)
            #self.bax,self.baP=gyro_accel_bias_estimator(np.array([self.t_roll,self.t_pitch]),[p,q,r],self.dti,self.baP,self.bax) #meaningless when R is big..

    def tf_callback(self,msg):
		idx=len(msg.transforms)-1
		if msg.transforms[idx].child_frame_id=="uav/imu":
		    self.truth=msg.transforms[idx].transform.translation
		    orientation_list = [msg.transforms[idx].transform.rotation.x, msg.transforms[idx].transform.rotation.y, msg.transforms[idx].transform.rotation.z, msg.transforms[idx].transform.rotation.w]
		    (self.roll, self.pitch, self.yaw) = euler_from_quaternion(orientation_list)

    def input_callback(self,msg):
		self.input_check=1

def indirect_estimator(accel,rates,dt,P,x): # only in NED,
    bias = [x[3], x[4], x[5]]
    prev_accel = [x[6], x[7], x[8]]
    nu=0.4

    '''Model part'''
    #1. delta psi
    p=(rates[0]-x[3])*dt # NWU->NED ,offset, dt
    q=(-rates[1]-x[4])*dt # NWU->NED ,offset, dt
    r=(-rates[2]-x[5])*dt # NWU->NED ,offset, dt

    #2. delta Q
    [qw,qx,qy,qz] = quaternion_from_euler(p,q,r) #current angular change
    [qw_p,qx_p,qy_p,qz_p] = quaternion_from_euler(x[0],x[1],x[2]) #previous angles
    #3. q- prediction
    [qw_u,qx_u,qy_u,qz_u]=quaternion_multiply([qw_p,qx_p,qy_p,qz_p],[qw,qx,qy,qz]) #updated orientation, prediction

    #4. Estimate g from orientation
    r_prior=quaternion_matrix([qw_u,qx_u,qy_u,qz_u])[:3,:3] # rotation matrix from predicted q_
    gravity_vector_from_orientation = r_prior[:,2].reshape(3,1)
    go=gravity_vector_from_orientation # to use easier

    #5. estimate g from accel
    gravity_vector_from_accel = np.array([accel[0]-x[6], -accel[1]-x[7], -accel[2]-x[8]]) #accel NWU(+) -> NED(+)
    z = gravity_vector_from_orientation - gravity_vector_from_accel.reshape(3,1) #error model


    ''' Kalman Equations '''
    H=np.array([[0, go[2], -go[1], 0, -go[2]*dt, go[1]*dt, 1, 0, 0], 
                [-go[2], 0, go[0], go[2]*dt, 0, -go[0]*dt, 0, 1, 0],
                [go[1], -go[0], 0, -go[1]*dt, go[0]*dt, 0, 0, 0, 1]]) #from observation model, kappa=1(DecimationFactor)/samplingrate == *dt


    kappa = dt*dt
    gyro_drift_noise = 0.001
    accel_drift_noise = 0.01
    R= 10000* (0.005 + accel_drift_noise + kappa*(gyro_drift_noise+0.003)) * np.array([[1,0,0],[0,1,0],[0,0,1]])# covariance of observation model noise * (gyroNoise + Drift + AccelNoise + Drift + sampling time)
    kappa = dt
    S= R + np.dot(H,np.dot(P,H.transpose()))
    S_ti = inv(S.transpose())
    K= np.dot(P,np.dot(H.transpose(),S_ti))
    x= np.dot(K,z.reshape(3,1))  

    P_k = P - np.dot(K,np.dot(H,P)) # updated Error Estimate covariance
    Q = 0.001*np.array([[sum(P_k[0,:])+kappa*kappa*P_k[2,0]+0.003+gyro_drift_noise, 0, 0, -kappa*(P_k[2,0]+gyro_drift_noise), 0, 0, 0, 0, 0],
                       [0, P_k[0,0]+kappa*kappa*P_k[3,0]+0.003+gyro_drift_noise, 0, 0, P_k[3,0]+gyro_drift_noise, 0, 0, 0, 0],
                       [0, 0, P_k[1,0]+kappa*kappa*P_k[4,0]+0.003+gyro_drift_noise, 0, 0, P_k[4,0]+gyro_drift_noise, 0, 0, 0],
                       [-kappa*(P_k[2,0]+gyro_drift_noise), 0, 0, P_k[2,0]+gyro_drift_noise, 0, 0, 0, 0, 0],
                       [0, P_k[3,0]+gyro_drift_noise, 0, 0, P_k[3,0]+gyro_drift_noise, 0, 0, 0, 0],
                       [0, 0, P_k[4,0]+gyro_drift_noise, 0, 0, P_k[4,0]+gyro_drift_noise, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, nu*nu*P_k[5,0]+accel_drift_noise, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, nu*nu*P_k[6,0]+accel_drift_noise, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, nu*nu*P_k[7,0]+accel_drift_noise]])

    '''correction'''
    x[0],x[1],x[2]= euler_from_quaternion(quaternion_multiply([qw_u,qx_u,qy_u,qz_u],quaternion_from_euler(x[0],x[1],x[2])))
    x[3] = bias[0] - x[3]
    x[4] = bias[1] - x[4]
    x[5] = bias[2] - x[5]
    x[6] = prev_accel[0]*nu - x[6]
    x[7] = prev_accel[1]*nu - x[7]
    x[8] = prev_accel[2]*nu - x[8]
    #print(p-x[3], q-x[4], r-x[5]) # angular velocities in NED
    return x, Q # Q is set to P (predicted P)

def accelerometer_arctan_estimator(ax,ay,az): # orientation must be checked before used.
	# roll=atan2(ay,az) #book - roll->pitch order
	# pitch=atan2(ax,sqrt(pow(ay,2)+pow(az,2)))
	roll=atan2(ay,sqrt(pow(ax,2)+pow(az,2))) # paper - roll->pitch order, different accel orientation.
	pitch=atan2(-ax,az)
	# roll=atan2(ay,sqrt(pow(ax,2)+pow(az,2))) #InHwan - pitch->roll order
	# pitch=atan2(ax,az)
	return roll,pitch

def LKF_estimator(z,rates,dt,P,x):
	p=rates[0]
	q=rates[1]
	r=rates[2]
	[qw,qx,qy,qz]=quaternion_from_euler(z[0],z[1],z[2])
	z=np.array([qw,qx,qy,qz]).reshape((4,1))

	H=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	Q=np.array([[0.0001,0,0,0],[0,0.0001,0,0],[0,0,0.0001,0],[0,0,0,0.0001]])
	R=np.array([[5000,0,0,0],[0,5000,0,0],[0,0,5000,0],[0,0,0,5000]])
	
	A=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) + \
	dt*0.5*np.array([[0,-p,-q,-r],[p,0,r,-q],[q,-r,0,p],[r,q,-p,0]])

	x_ = np.dot(A,x)
	P_ = np.dot(np.dot(A,P),A.transpose()) + Q
	K = np.dot(np.dot(P_,H.transpose()),inv(np.dot(np.dot(H,P_),H.transpose()) + R))

	x = x_ + np.dot(K,(z-np.dot(H,x_)))
	P = P_ - np.dot(np.dot(K,H),P_)
	return x,P

def EKF_estimator(z,rates,dt,P,x):
    p=rates[0]
    q=rates[1]
    r=rates[2]
    roll=x[0]
    pitch=x[1]
    yaw=x[2]
    z=z.reshape((3,1))

    H=np.array([[1,0,0],[0,1,0],[0,0,0]])
    Q=np.array([[0.0001,0,0],[0,0.0001,0],[0,0,0.0001]]) #Process noise
    R=np.array([[5000,0,0],[0,5000,0],[0,0,5000]]) #Sensor noise

    A=np.array([[1,0,0],[0,1,0],[0,0,1]]) + \
    dt * np.array([[q*cos(roll)*tan(pitch)-r*sin(roll)*tan(pitch), (q*sin(roll)+r*cos(roll))*pow(sec(pitch),2), 0], \
    [-q*sin(roll)-r*cos(roll), 0, 0], \
    [(q*cos(roll)-r*sin(roll))*sec(pitch), (q*sin(roll)+r*cos(roll))*sec(pitch)*tan(pitch), 0]])

    x_=x + dt*np.array([[p+(q*sin(roll)+r*cos(roll))*tan(pitch)], [q*cos(roll)-r*sin(roll)], [(q*sin(roll)+r*cos(roll))*sec(pitch)]])
    x_=x_.reshape((3,1))
    P_=np.dot(np.dot(A,P),A.transpose()) + Q

    K =np.dot(P_,H.transpose())*inv(np.dot(np.dot(H,P_),H.transpose()) + R)

    P =P_ - np.dot(np.dot(K,H),P_)
    if abs(np.amax(P))>1e+06:
        P=np.array([[100,0,0],[0,100,0],[0,0,100]])
        x=x
        return x,P
    x =x_ + np.dot(K,(z-np.dot(H,x_)))
    return x,P

def gyrobias_estimator(z,rates,dt,P,x): # x = [roll, roll_bias, pitch, pitch_bias]
	p=rates[0]
	q=rates[1]
	r=rates[2]
	# u=np.array([p*cos(x[2]),q,p*sin(x[2])+r])# as paper, too much approximated.
	u=np.array([p + tan(x[2])*(q*sin(x[0])+r*cos(x[0])) , q*cos(x[0])-r*sin(x[0])]).reshape((2,1)) # as conventional, universal

	z=z.reshape((2,1)) #roll, pitch

	A=np.array([[1,-dt,0,0],[0,1,0,0],[0,0,1,-dt],[0,0,0,1]])
	B=np.array([[dt,0],[0,0],[0,dt],[0,0]])

	H=np.array([[1,0,0,0],[0,0,1,0]]) # as paper, correct
	# H=np.array([[0,0,1,0],[-1,0,0,0]]) # as reference paper, typo?
	Q=np.array([[0.0001,0,0,0],[0,0.0001,0,0],[0,0,0.0001,0],[0,0,0,0.0001]]) #Process noise
	R=np.array([[5000,0],[0,5000]]) #Sensor noise
	# Q=np.dot(np.dot(B,R),B.transpose()) #Process noise as paper

	x_=np.dot(A,x)+np.dot(B,u)
	x_=x_.reshape((4,1))
	P_=np.dot(np.dot(A,P),A.transpose()) + Q

	K_temp = np.dot(H.transpose(),inv(np.dot(np.dot(H,P_),H.transpose()) + R))
	K =np.dot(P_,K_temp)

	P =P_ - np.dot(np.dot(K,H),P_)
	x =x_ + np.dot(K,(z-np.dot(H,x_)))
	
	return x,P


Ra=deque([0]*3) #residual
lamda=deque([0]*3) #eigen value
vector=deque([0]*3) #eigen vector
mu=deque([0]*3) # criteria for external accel
global check
check=0
# R is so big, accel_bias cannot be considered (already being so)
def gyro_accel_bias_estimator(z,rates,dt,P,x): # x = [roll, roll_bias, pitch, pitch_bias]
	global check
	p=rates[0]
	q=rates[1]
	r=rates[2]
	# u=np.array([p*cos(x[2]),q,p*sin(x[2])+r])# as paper, too much approximated.
	u=np.array([p + tan(x[2])*(q*sin(x[0])+r*cos(x[0])) , q*cos(x[0])-r*sin(x[0])]).reshape((2,1)) # as conventional, universal

	z=z.reshape((2,1)) #roll, pitch

	A=np.array([[1,-dt,0,0],[0,1,0,0],[0,0,1,-dt],[0,0,0,1]])
	B=np.array([[dt,0],[0,0],[0,dt],[0,0]])

	H=np.array([[1,0,0,0],[0,0,1,0]])
	Q=np.array([[0.0001,0,0,0],[0,0.0001,0,0],[0,0,0.0001,0],[0,0,0,0.0001]]) #Process noise
	R=np.array([[5000,0],[0,5000]]) #Sensor noise
	R_acc=np.array([[0,0],[0,0]]) # external acceleration 
	# Q=np.dot(np.dot(B,R),B.transpose())*0.1 #Process noise

	x_=np.dot(A,x)+np.dot(B,u)
	x_=x_.reshape((4,1))
	P_=np.dot(np.dot(A,P),A.transpose()) + Q


	Ra.popleft()
	Ra.append(z-np.dot(H,x_)) # residual
	Uk=np.array([[0,0],[0,0]])
	if type(Ra[0])!=int:
		for j in range(3):
			ra_temp=np.array(Ra[j]).reshape((2,1))
			Uk=Uk + np.dot(ra_temp,ra_temp.transpose())
		Uk=Uk/3
		lamda_temp, vector_temp=np.linalg.eig(Uk) # eigen value and eigen vector
		lamda.popleft()		
		lamda.append(lamda_temp)
		vector.popleft()		
		vector.append(vector_temp)

		if type(vector[0])!=int:
			mu.popleft()			
			mu.append([np.dot(np.dot(vector[2][0].transpose() ,(np.dot(np.dot(H,P_),H.transpose())) + R) , vector[2][0]), np.dot(np.dot(vector[2][1].transpose() ,(np.dot(np.dot(H,P_),H.transpose())) + R) , vector[2][1])])
			if type(mu[0])!=int:
				if np.amax([lamda[0][0]-mu[0][0],lamda[0][1]-mu[0][1], lamda[1][0]-mu[1][0],lamda[1][1]-mu[1][1], lamda[2][0]-mu[2][0],lamda[2][1]-mu[2][1]])<0.1:
					R_acc=np.array([[0,0],[0,0]])
				else :
					R_acc=max([lamda[2][0]-mu[2][0], 0])*np.dot(vector[2][0].reshape((2,1)),vector[2][0].reshape((2,1)).transpose()) + \
					max([lamda[2][1]-mu[2][1], 0])*np.dot(vector[2][1].reshape((2,1)),vector[2][1].reshape((2,1)).transpose())

	K_temp = np.dot(H.transpose(),inv(np.dot(np.dot(H,P_),H.transpose()) + R + R_acc))
	K =np.dot(P_,K_temp)

	P =P_ - np.dot(np.dot(K,H),P_)
	x =x_ + np.dot(K,z-np.dot(H,x_))
	
	return x,P

# def UKF_estimator(z,rates,dt,P,x):
# 	p=rates[0]
# 	q=rates[1]
# 	r=rates[2]
# 	roll=x[0]
# 	pitch=x[1]
# 	yaw=x[2]

# def CF_estimator():
# 	pass

def sec(x):
	return 1/cos(x)

#def quaternion_multiply(q1, q2):
#    a1, b1, c1, d1 = q1
#    a2, b2, c2, d2 = q2
#    return np.array([a1-a2 - b1*b2 - c1*c2 - d1*d2,
#                     a1*b2 + b1*a2 + c1*d2 - d1*c2,
#                     a1*c2 - b1*d2 + c1*a2 + d1*b2,
#                     a1*d2 + b1*c2 - c1*b2 + d1*a2], dtype=np.float64)

def graphupdate(i):
    global tg
    global prev_tg
    if prev_tg==0:
        prev_tg=time.time()
        return
    tg=tg+(time.time()-prev_tg)
    prev_tg=time.time()

    x_time.append(tg)
    x_time.popleft()
    roll_fig.set_xlim(np.amin(x_time),np.amax(x_time))
    pitch_fig.set_xlim(np.amin(x_time),np.amax(x_time))
	# yaw_fig.set_xlim(np.amin(x_time),np.amax(x_time))

    y_roll_truth.popleft()
    y_roll_truth.append(ekf.roll*r2d)
    roll_line1.set_data(x_time,y_roll_truth)

	# y_roll_accelero.popleft()
	# y_roll_accelero.append(ekf.t_roll*r2d)
	# roll_line2.set_data(x_time,y_roll_accelero)

	# y_roll_LKF.popleft()
	# y_roll_LKF.append(ekf._xe[0]*r2d)
	# roll_line3.set_data(x_time,y_roll_LKF)


    y_roll_indirect.popleft()
    y_roll_indirect.append(ekf.ix[0]*r2d)
    roll_line3.set_data(x_time,y_roll_indirect)

    y_roll_EKF.popleft()
    y_roll_EKF.append(ekf.x[0]*r2d)
    roll_line4.set_data(x_time,y_roll_EKF)

#	y_roll_bias.popleft()
#	y_roll_bias.append(ekf.bx[0]*r2d)
#	roll_line5.set_data(x_time,y_roll_bias)

    y_pitch_truth.popleft()
    y_pitch_truth.append(ekf.pitch*r2d) 
    pitch_line1.set_data(x_time,y_pitch_truth)

	# y_pitch_accelero.popleft()
	# y_pitch_accelero.append(ekf.t_pitch*r2d)
	# pitch_line2.set_data(x_time,y_pitch_accelero)

	# y_pitch_LKF.popleft()
	# y_pitch_LKF.append(ekf._xe[1]*r2d)
	# pitch_line3.set_data(x_time,y_pitch_LKF)

    y_pitch_indirect.popleft()
    y_pitch_indirect.append(-ekf.ix[1]*r2d) #since it is in NED
    pitch_line3.set_data(x_time,y_pitch_indirect)

    y_pitch_EKF.popleft()
    y_pitch_EKF.append(ekf.x[1]*r2d)
    pitch_line4.set_data(x_time,y_pitch_EKF)

#	y_pitch_bias.popleft()
#	y_pitch_bias.append(ekf.bx[2]*r2d)
#	pitch_line5.set_data(x_time,y_pitch_bias)

	# y_yaw_truth.popleft()
	# y_yaw_truth.append(ekf.yaw*r2d)
	# yaw_line1.set_data(x_time,y_yaw_truth)

	# y_yaw_accelero.popleft()
	# y_yaw_accelero.append(ekf.euler_integrated_yaw*r2d)
	# yaw_line2.set_data(x_time,y_yaw_accelero)

#	print("estimated roll: %.1f, pitch: %.1f, yaw: %.1f "%(ekf.t_roll*r2d,ekf.t_pitch*r2d,ekf.euler_integrated_yaw*r2d))
#	print("LKF###### roll: %.1f, pitch: %.1f, yaw: %.1f "%(ekf._xe[0]*r2d,ekf._xe[1]*r2d,ekf._xe[2]*r2d))
    print("##EKF#### roll: %.1f, pitch: %.1f, yaw: %.1f "%(ekf.x[0]*r2d, ekf.x[1]*r2d, ekf.x[2]*r2d))
#	print("###bias## roll: %.1f, pitch: %.1f, bias: %.1f ,  %.1f "%(ekf.bx[0]*r2d, ekf.bx[2]*r2d, ekf.bx[1]*r2d, ekf.bx[3]*r2d))
#	print("###bias## roll: %.1f, pitch: %.1f, bias: %.1f ,  %.1f "%(ekf.bax[0]*r2d, ekf.bax[2]*r2d, ekf.bax[1]*r2d, ekf.bax[3]*r2d))
    print("Indirect# roll: %.1f, pitch: %.1f, yaw: %.1f "%(ekf.ix[0]*r2d, -ekf.ix[1]*r2d, -ekf.ix[2]*r2d)) #since it is in NED
    print("#####TRUE roll: %.1f, pitch: %.1f, yaw: %.1f \n\n\n"%(ekf.roll*r2d, ekf.pitch*r2d, ekf.yaw*r2d))

if __name__ == '__main__':

    ekf=ekf()

    ''' For Graph '''
    fig=plt.figure(figsize=(8,10))

    roll_fig=fig.add_subplot(2,1,1)
    roll_fig.set_title('Roll')
    roll_fig.set_xlabel('seconds')
    roll_fig.set_ylabel('Degree')
    roll_fig.grid(color='gray',linestyle='dotted',alpha=0.8)
    roll_fig.set_ylim(-45,45)

    pitch_fig=fig.add_subplot(2,1,2)
    pitch_fig.set_ylim(-45,45)
    pitch_fig.set_title('Pitch')
    pitch_fig.set_xlabel('seconds')
    pitch_fig.set_ylabel('Degree')
    pitch_fig.grid(color='gray',linestyle='dotted',alpha=0.8)

    # yaw_fig=fig.add_subplot(3,1,3)
    # yaw_fig.set_ylim(-185,185)
    # yaw_fig.set_title('Yaw')
    # yaw_fig.set_xlabel('seconds')
    # yaw_fig.set_ylabel('Degree')
    # yaw_fig.grid(color='gray',linestyle='dotted')

    roll_line1,=roll_fig.plot([],[],color='blue',label='Truth')
    # roll_line2,=roll_fig.plot([],[],color='black',label='Accelero',linewidth=0.7,alpha=0.8)
    # roll_line3,=roll_fig.plot([],[],color='green',label='LKF',linewidth=0.7,alpha=0.8)
    roll_line3,=roll_fig.plot([],[],color='green',label='Indirect')
    roll_line4,=roll_fig.plot([],[],color='red',label='EKF')
    #	roll_line5,=roll_fig.plot([],[],color='magenta',label='Gyro Bias-LKF')

    pitch_line1,=pitch_fig.plot([],[],color='blue',label='Truth')
    # pitch_line2,=pitch_fig.plot([],[],color='black',label='Accelero',linewidth=0.7,alpha=0.8)
    # pitch_line3,=pitch_fig.plot([],[],color='green',label='LKF',linewidth=0.7,alpha=0.8)
    pitch_line3,=pitch_fig.plot([],[],color='green',label='Indirect')
    pitch_line4,=pitch_fig.plot([],[],color='red',label='EKF')
    #	pitch_line5,=pitch_fig.plot([],[],color='magenta',label='Gyro Bias-LKF')

    # yaw_line1,=yaw_fig.plot([],[],color='blue',label='Truth',linewidth=1,alpha=0.8)
    # yaw_line2,=yaw_fig.plot([],[],color='black',label='Euler_integrated',linewidth=1,alpha=0.8)

    roll_fig.legend(loc='upper left')
    pitch_fig.legend(loc='upper left')
    # yaw_fig.legend(loc='upper left')

    global t
    global width
    t=0
    width=100
    x_time=deque(np.linspace(3,0,num=width))
    y_roll_truth=deque([0]*width)
    y_pitch_truth=deque([0]*width)
#    y_yaw_truth=deque([0]*width)

    #	y_roll_accelero=deque([0]*width)
    #	y_pitch_accelero=deque([0]*width)
    #	y_yaw_accelero=deque([0]*width)

    #	y_roll_LKF=deque([0]*width)
    #	y_pitch_LKF=deque([0]*width)
    #	y_yaw_LKF=deque([0]*width)

    y_roll_indirect=deque([0]*width)
    y_pitch_indirect=deque([0]*width)

    y_roll_EKF=deque([0]*width)
    y_pitch_EKF=deque([0]*width)
#    y_yaw_EKF=deque([0]*width)

    #	y_roll_bias=deque([0]*width)
    #	y_pitch_bias=deque([0]*width)
    #	y_yaw_bias=deque([0]*width)

    while 1:
        try:
            if ekf.imu_check==1 and ekf.input_check==1:
                ''' Graph update & print '''
                animation=ani.FuncAnimation(fig,graphupdate,interval=150)
                plt.tight_layout()
                plt.show()
        except (rospy.ROSInterruptException, SystemExit, KeyboardInterrupt) :
            sys.exit(0)
        except :
            print("something's wrong")
