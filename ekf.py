#!/usr/bin/env python

#author: mason

import rospy
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage
from mav_msgs.msg import RateThrust

from math import atan2,pow,sqrt,sin,cos,tan
from tf.transformations import euler_from_quaternion,quaternion_from_euler

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

		self.P=np.array([[10,0,0],[0,10,0],[0,0,10]])
		self.x=np.array([0,0,0]).reshape((3,1)) #EKF
		self._P=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
		self._x=np.array([1,0,0,0]).reshape((4,1)) #LKF


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
			ax=FirstOrderLPF(self.prev_ax, lina.x, 200, 1/960);
			ay=FirstOrderLPF(self.prev_ay, lina.y, 200, 1/960);
			az=FirstOrderLPF(self.prev_az, lina.z, 200, 1/960);
			self.prev_ax=ax
			self.prev_ay=ay
			self.prev_az=az
			''' Filtering '''
			self.t_roll,self.t_pitch=accelerometer_arctan_estimator(ax,ay,az)

			self._x,self._P=LKF_estimator([self.t_roll,self.t_pitch,self.euler_integrated_yaw],[p,q,r],self.dti,self._P,self._x)
			self._xe=euler_from_quaternion([self._x[0][0],self._x[1][0],self._x[2][0],self._x[3][0]])

			self.x,self.P=EKF_estimator(np.array([self.t_roll,self.t_pitch,self.euler_integrated_yaw]),[p,q,r],self.dti,self.P,self.x)

    def tf_callback(self,msg):
		idx=len(msg.transforms)-1
		if msg.transforms[idx].child_frame_id=="uav/imu":
		    self.truth=msg.transforms[idx].transform.translation
		    orientation_list = [msg.transforms[idx].transform.rotation.x, msg.transforms[idx].transform.rotation.y, msg.transforms[idx].transform.rotation.z, msg.transforms[idx].transform.rotation.w]
		    (self.roll, self.pitch, self.yaw) = euler_from_quaternion(orientation_list)

    def input_callback(self,msg):
		self.input_check=1
		r=msg.angular_rates.z

		self.curr_t=time.time()
		if self.prev_t==0:
			self.prev_t=time.time()
			return
		self.euler_integrated_yaw=self.euler_integrated_yaw + r * (self.curr_t-self.prev_t) # * dt
		self.prev_t=self.curr_t

		if self.euler_integrated_yaw>np.pi:
			self.euler_integrated_yaw=self.euler_integrated_yaw-2*np.pi
		if self.euler_integrated_yaw<-np.pi:
			self.euler_integrated_yaw=self.euler_integrated_yaw+2*np.pi

def accelerometer_arctan_estimator(ax,ay,az):
	roll=atan2(ay,az)
	pitch=atan2(ax,sqrt(pow(ay,2)+pow(az,2)))
	# roll=atan2(-ay,sqrt(pow(ax,2)+pow(az,2)))
	# pitch=atan2(-ax,az)
	return roll,pitch

def LKF_estimator(z,rates,dt,P,x):
	p=rates[0]
	q=rates[1]
	r=rates[2]
	[qw,qx,qy,qz]=quaternion_from_euler(z[0],z[1],z[2])
	z=np.array([qw,qx,qy,qz]).reshape((4,1))

	H=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	Q=np.array([[0.001,0,0,0],[0,0.001,0,0],[0,0,0.001,0],[0,0,0,0.001]])
	R=np.array([[7000,0,0,0],[0,7000,0,0],[0,0,7000,0],[0,0,0,7000]])
	
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
	Q=np.array([[0.001,0,0],[0,0.001,0],[0,0,0.001]]) #Process noise
	R=np.array([[7000,0,0],[0,7000,0],[0,0,7000]]) #Sensor noise

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

def gyrobias_estimator(z,rates,dt,P,x):
	p=rates[0]
	q=rates[1]
	r=rates[2]
	roll=x[0]
	pitch=x[1]
	yaw=x[2]
	z=z.reshape((2,1)) #roll, pitch

	A=np.array([[1,-dt,0,0],[0,1,0,0],[0,0,1,-dt],[0,0,0,1]])
	B=np.array([[dt,0],[0,0],[0,dt],[0,0]])

	H=np.array([[1,0,0,0],[0,0,1,0]])
	Q=np.array([[0.001,0,0,0],[0,0.001,0,0],[0,0,0.001,0],[0,0,0,0.001]]) #Process noise
	# Q=np.dot(np.dot(B,R),R.transpose())
	R=np.array([[7000,0,0,0],[0,7000,0,0],[0,0,7000,0],[0,0,0,7000]]) #Sensor noise

	

	x_=np.dot(A,x)+np.dot(B,u)
	x_=x_.reshape((4,1))
	P_=np.dot(np.dot(A,P),A.transpose()) + Q

	K =np.dot(P_,H.transpose())*inv(np.dot(np.dot(H,P_),H.transpose()) + R)

	P =P_ - np.dot(np.dot(K,H),P_)
	if abs(np.amax(P))>1e+06:
		P=np.array([[100,0,0],[0,100,0],[0,0,100]])
		x=x
		return x,P
	x =x_ + np.dot(K,(z-np.dot(H,x_)))
	
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


def FirstOrderLPF(PtrPrevOutput, CurInput, CutOffFreq_hz, SamplingTime_sec):
    LPFGain = SamplingTime_sec/((1/(2*np.pi*CutOffFreq_hz))+SamplingTime_sec)
    LPFOutput = (1-LPFGain)*(PtrPrevOutput) + LPFGain*CurInput
    return LPFOutput

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
	yaw_fig.set_xlim(np.amin(x_time),np.amax(x_time))

	y_roll_truth.popleft()
	y_roll_truth.append(ekf.roll*r2d)
	roll_line1.set_data(x_time,y_roll_truth)

	y_roll_accelero.popleft()
	y_roll_accelero.append(ekf.t_roll*r2d)
	roll_line2.set_data(x_time,y_roll_accelero)

	y_roll_LKF.popleft()
	y_roll_LKF.append(ekf._xe[0]*r2d)
	roll_line3.set_data(x_time,y_roll_LKF)

	y_roll_EKF.popleft()
	y_roll_EKF.append(ekf.x[0]*r2d)
	roll_line4.set_data(x_time,y_roll_EKF)

	y_pitch_truth.popleft()
	y_pitch_truth.append(ekf.pitch*r2d)
	pitch_line1.set_data(x_time,y_pitch_truth)

	y_pitch_accelero.popleft()
	y_pitch_accelero.append(ekf.t_pitch*r2d)
	pitch_line2.set_data(x_time,y_pitch_accelero)

	y_pitch_LKF.popleft()
	y_pitch_LKF.append(ekf._xe[1]*r2d)
	pitch_line3.set_data(x_time,y_pitch_LKF)

	y_pitch_EKF.popleft()
	y_pitch_EKF.append(ekf.x[1]*r2d)
	pitch_line4.set_data(x_time,y_pitch_EKF)

	y_yaw_truth.popleft()
	y_yaw_truth.append(ekf.yaw*r2d)
	yaw_line1.set_data(x_time,y_yaw_truth)

	y_yaw_accelero.popleft()
	y_yaw_accelero.append(ekf.euler_integrated_yaw*r2d)
	yaw_line2.set_data(x_time,y_yaw_accelero)

	y_yaw_LKF.popleft()
	y_yaw_LKF.append(ekf._xe[2]*r2d)
	yaw_line3.set_data(x_time,y_yaw_LKF)

	y_yaw_EKF.popleft()
	y_yaw_EKF.append(ekf.x[2]*r2d)
	yaw_line4.set_data(x_time,y_yaw_EKF)

	print("estimated roll: %.1f, pitch: %.1f, yaw: %.1f "%(ekf.t_roll*r2d,ekf.t_pitch*r2d,ekf.euler_integrated_yaw*r2d))
	print("LKF###### roll: %.1f, pitch: %.1f, yaw: %.1f "%(ekf._xe[0]*r2d,ekf._xe[1]*r2d,ekf._xe[2]*r2d))
	print("###EKF### roll: %.1f, pitch: %.1f, yaw: %.1f "%(ekf.x[0]*r2d,ekf.x[1]*r2d,ekf.x[2]*r2d))
	print("#####TRUE roll: %.1f, pitch: %.1f, yaw: %.1f \n\n\n"%(ekf.roll*r2d, ekf.pitch*r2d, ekf.yaw*r2d))

if __name__ == '__main__':

    ekf=ekf()

    ''' For Graph '''
    fig=plt.figure(figsize=(8,10))

    roll_fig=fig.add_subplot(3,1,1)
    roll_fig.set_title('Roll')
    roll_fig.set_xlabel('seconds')
    roll_fig.set_ylabel('Degree')
    roll_fig.grid(color='gray',linestyle='dotted')
    roll_fig.set_ylim(-35,35)

    pitch_fig=fig.add_subplot(3,1,2)
    pitch_fig.set_ylim(-35,35)
    pitch_fig.set_title('Pitch')
    pitch_fig.set_xlabel('seconds')
    pitch_fig.set_ylabel('Degree')
    pitch_fig.grid(color='gray',linestyle='dotted')

    yaw_fig=fig.add_subplot(3,1,3)
    yaw_fig.set_ylim(-185,185)
    yaw_fig.set_title('Yaw')
    yaw_fig.set_xlabel('seconds')
    yaw_fig.set_ylabel('Degree')
    yaw_fig.grid(color='gray',linestyle='dotted')

    roll_line1,=roll_fig.plot([],[],color='blue',label='Truth')
    roll_line2,=roll_fig.plot([],[],color='black',label='Accelero')
    roll_line3,=roll_fig.plot([],[],color='green',label='LKF')
    roll_line4,=roll_fig.plot([],[],color='red',label='EKF')

    pitch_line1,=pitch_fig.plot([],[],color='blue',label='Truth')
    pitch_line2,=pitch_fig.plot([],[],color='black',label='Accelero')
    pitch_line3,=pitch_fig.plot([],[],color='green',label='LKF')
    pitch_line4,=pitch_fig.plot([],[],color='red',label='EKF')

    yaw_line1,=yaw_fig.plot([],[],color='blue',label='Truth')
    yaw_line2,=yaw_fig.plot([],[],color='black',label='Euler_integrated')
    yaw_line3,=yaw_fig.plot([],[],color='green',label='LKF')
    yaw_line4,=yaw_fig.plot([],[],color='red',label='EKF')

    roll_fig.legend(loc='upper left')
    pitch_fig.legend(loc='upper left')
    yaw_fig.legend(loc='upper left')

    global t
    global width
    t=0
    width=400
    x_time=deque(np.linspace(3,0,num=width))
    y_roll_truth=deque([0]*width)
    y_pitch_truth=deque([0]*width)
    y_yaw_truth=deque([0]*width)

    y_roll_accelero=deque([0]*width)
    y_pitch_accelero=deque([0]*width)
    y_yaw_accelero=deque([0]*width)

    y_roll_LKF=deque([0]*width)
    y_pitch_LKF=deque([0]*width)
    y_yaw_LKF=deque([0]*width)

    y_roll_EKF=deque([0]*width)
    y_pitch_EKF=deque([0]*width)
    y_yaw_EKF=deque([0]*width)


    while 1:
		try:
			if ekf.imu_check==1 and ekf.input_check==1:
				''' Graph update & print '''
				animation=ani.FuncAnimation(fig,graphupdate,interval=100)
				plt.tight_layout()
				plt.show()

		except (rospy.ROSInterruptException, SystemExit, KeyboardInterrupt) :
			sys.exit(0)
		except :
		    print("something's wrong")
