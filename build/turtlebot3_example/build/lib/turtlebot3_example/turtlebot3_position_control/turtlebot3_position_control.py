#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Du Yong, Wang Yubin

import math
import numpy
import numpy as np
import sys
import termios
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random 
import rclpy

import numpy.matlib
import time
from numpy import linalg as LA
from scipy.integrate import odeint
from scipy.optimize import minimize
from casadi import *

#from tkinter import Button
from matplotlib.widgets import Button
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from turtlebot3_msgs.msg import SensorState

from turtlebot3_example.turtlebot3_position_control.turtlebot3_path import Turtlebot3Path

terminal_msg = """
Turtlebot3 Position Control
------------------------------------------------------
From the current pose,
x: goal position x (unit: m)
y: goal position y (unit: m)
theta: goal orientation (range: -180 ~ 180, unit: deg)
------------------------------------------------------
"""

class Turtlebot3PositionControl(Node):

    def __init__(self):
        super().__init__('turtlebot3_position_control')

        """************************************************************
        ** Initialise variables
        ************************************************************"""
        self.form_num=1     
        self.node_num=6 # default 6
        self.trajectory_len=79
        self.time=0.1
        self.buttonstart=0

        ######## initialize mpc ##########
        self.T = 0.3  # [s] # 0.3
        self.N = 35  
        self.m = 6
        self.M = int(self.m * (self.m -1)/2)
        self.dmin = 0.3 + 0.2                                                                     # prediction horizon
        self.rob_diam = 0.15

        self.v_max = +0.22; self.v_min = -self.v_max  ## max_vel
        self.omega_max = +2.84; self.omega_min = -self.omega_max

        ##casadi attributes
        x1 = SX.sym('x1'); y1 = SX.sym('y1'); theta1 = SX.sym('theta1')
        x2 = SX.sym('x2'); y2 = SX.sym('y2'); theta2 = SX.sym('theta2')
        x3 = SX.sym('x3'); y3 = SX.sym('y3'); theta3 = SX.sym('theta3')
        x4 = SX.sym('x4'); y4 = SX.sym('y4'); theta4 = SX.sym('theta4')
        x5 = SX.sym('x5'); y5 = SX.sym('y5'); theta5 = SX.sym('theta5')
        x6 = SX.sym('x6'); y6 = SX.sym('y6'); theta6 = SX.sym('theta6')

        states = np.array([[x1], [y1], [theta1], [x2], [y2], [theta2], [x3], [y3], [theta3], [x4], [y4], [theta4], [x5], [y5], [theta5], [x6], [y6], [theta6]]); self.n_states = len(states)

        v1 = SX.sym('v1'); omega1 = SX.sym('omega1')
        v2 = SX.sym('v2'); omega2 = SX.sym('omega2')
        v3 = SX.sym('v3'); omega3 = SX.sym('omega3')
        v4 = SX.sym('v4'); omega4 = SX.sym('omega4')
        v5 = SX.sym('v5'); omega5 = SX.sym('omega5')
        v6 = SX.sym('v6'); omega6 = SX.sym('omega6')
   
        controls = np.array([[v1],[omega1], [v2], [omega2], [v3], [omega3], [v4], [omega4], [v5], [omega5] , [v6], [omega6]]); self.n_controls = len(controls)
        rhs = np.array([[v1*np.cos(theta1)],[v1*np.sin(theta1)],[omega1], [v2*np.cos(theta2)],[v2*np.sin(theta2)],[omega2], [v3*np.cos(theta3)],[v3*np.sin(theta3)],[omega3], [v4*np.cos(theta4)],[v4*np.sin(theta4)],[omega4], [v5*np.cos(theta5)],[v5*np.sin(theta5)],[omega5], \
        [v6*np.cos(theta6)],[v6*np.sin(theta6)],[omega6]])                   # system r.h.s
        # print("rhs: ", rhs)

        self.f = Function('f',[states,controls],[rhs])                                       # nonlinear mapping function f(x,u)
        #print("Function :", f)

        self.U = SX.sym('U',self.n_controls,self.N) # ;                                                   # Decision variables (controls)
        self.P = SX.sym('P',2*self.n_states) #print("U: ", U) ; #print("P: ", P)                                            # parameters (which include the initial state and the reference state)
        
        self.X = SX.sym('X',self.n_states,(self.N+1)) #;# A vector that represents the states over the optimization problem.      
        #print("X: ", X)

        self.obj = 0                                                                        # Objective function
                                                                                # constraints vector
        self.Q = np.zeros((self.n_states,self.n_states)) # weighing matrices (states)
        self.Q[0,0] = 1; self.Q[1,1] = 5; self.Q[2,2] = 0.1
        self.Q[3,3] = 1; self.Q[4,4] = 5; self.Q[5,5] = 0.1   #0.1                           # weighing matrices (states)
        self.Q[6,6] = 1; self.Q[7,7] = 5; self.Q[8,8] = 0.1
        self.Q[9,9] = 1; self.Q[10,10] = 5; self.Q[11,11] = 0.1   #0.1                           # weighing matrices (states)
        self.Q[12,12] = 1; self.Q[13,13] = 5; self.Q[14,14] = 0.1
        self.Q[15,15] = 1; self.Q[16,16] = 5; self.Q[17,17] = 0.1

        self.R = np.zeros((self.n_controls,self.n_controls))
        self.R[0,0] = 0.5; self.R[1,1] = 0.05
        self.R[2,2] = 0.5; self.R[3,3] = 0.05       #0.05                                  # weighing matrices (controls)
        self.R[4,4] = 0.5; self.R[5,5] = 0.05
        self.R[6,6] = 0.5; self.R[7,7] = 0.05       #0.05
        self.R[8,8] = 0.5; self.R[9,9] = 0.05
        self.R[10,10] = 0.5; self.R[11,11] = 0.05

        #print("Q: ", Q)
        #print("R: ", R)


        self.st  = self.X[:,0]                                                                    # initial state
        #print("st_type: ", self.st.type)
        self.g = vertcat(self.st-self.P[0:self.n_states], np.matlib.repmat(np.array([3.5]), self.M, 1))   
        #print("g: ", g)

        ######## initialize plot #########
        
        self.x_cur_trajectory=np.array([[0.0 for x in range(self.trajectory_len+1)] for y in range(self.node_num)]) 
        self.y_cur_trajectory=np.array([[0.0 for x in range(self.trajectory_len+1)] for y in range(self.node_num)]) 

        self.x_head=np.array([0.0 for x in range(self.node_num)]) 
        self.y_head=np.array([0.0 for x in range(self.node_num)])

        self.x_left=np.array([0.0 for x in range(self.node_num)])
        self.y_left=np.array([0.0 for x in range(self.node_num)])
    
        self.x_right=np.array([0.0 for x in range(self.node_num)])
        self.y_right=np.array([0.0 for x in range(self.node_num)])

        self.x_curl=np.array([[0.0 for x in range(2)] for y in range(self.node_num)])
        self.y_curl=np.array([[0.0 for x in range(2)] for y in range(self.node_num)])
        
        self.x_curr=np.array([[0.0 for x in range(2)] for y in range(self.node_num)])
        self.y_curr=np.array([[0.0 for x in range(2)] for y in range(self.node_num)])
        
        self.x_body=[0.0 for x in range(self.node_num)]
        self.y_body=[0.0 for x in range(self.node_num)]

        self.odom = Odometry()
        self.twisttb30 = Twist()
        #self.twisttb31 = Twist()
        #self.twisttb32 = Twist()
        #self.twisttb33 = Twist()
        #self.twisttb34 = Twist()
        #self.twisttb35 = Twist()
        '''
        self.x_cur=0     #[2.0,2.0,2.0,2.0,2.0,2.0]
        self.y_cur=0 
        self.d_cur=0
        self.thr = self.d_cur
        self.Xr = np.array([[self.x_cur], [self.y_cur], [self.thr]])
        '''
        self.x_cur=np.array([0.0 for x in range(self.node_num)])
        self.y_cur=np.array([0.0 for x in range(self.node_num)])
        self.theta_cur=np.array([0.0 for x in range(self.node_num)])
        
##########################################################################################################
        self.colorArr = ['r','g','b','c','m','k','gray','tan','pink','navy','r','g','b','c','m','k','gray','tan','pink','navy']
        
###########################################################################################################
        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialise publishers
        
        self.tb30_cmd_vel_pub = self.create_publisher(Twist,'tb3_0/cmd_vel', qos)
        self.tb31_cmd_vel_pub = self.create_publisher(Twist,'tb3_1/cmd_vel', qos)
        self.tb32_cmd_vel_pub = self.create_publisher(Twist,'tb3_2/cmd_vel', qos)
        self.tb33_cmd_vel_pub = self.create_publisher(Twist,'tb3_3/cmd_vel', qos)
        self.tb34_cmd_vel_pub = self.create_publisher(Twist,'tb3_4/cmd_vel', qos)
        self.tb35_cmd_vel_pub = self.create_publisher(Twist,'tb3_5/cmd_vel', qos)
        

        # Initialise subscribers
        '''
        self.odom_sub = self.create_subscription(Odometry, 'odom',self.odom_callback,qos)

        #####################position################################
        
        self.optitracktb30_sub = self.create_subscription(PoseStamped,'vrpn_client_node/tb3_0/pose',self.optitracktb30_callback,qos)
        self.optitracktb31_sub = self.create_subscription(PoseStamped,'vrpn_client_node/tb3_1/pose',self.optitracktb31_callback,qos)
        self.optitracktb32_sub = self.create_subscription(PoseStamped,'vrpn_client_node/tb3_2/pose',self.optitracktb32_callback,qos)
        self.optitracktb33_sub = self.create_subscription(PoseStamped,'vrpn_client_node/tb3_3/pose',self.optitracktb33_callback,qos)
        self.optitracktb34_sub = self.create_subscription(PoseStamped,'vrpn_client_node/tb3_4/pose',self.optitracktb34_callback,qos)
        self.optitracktb35_sub = self.create_subscription(PoseStamped,'vrpn_client_node/tb3_5/pose',self.optitracktb35_callback,qos)
        '''
   
        """************************************************************
        ** Initialise timers
        ************************************************************"""
        self.update_timer = self.create_timer(self.time, self.update_callback)  # unit: s
        self.get_logger().info("MPC simulation starts.")

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    #####################position callback#################################### 

    def optitracktb30_callback(self, msg):
        self.x_cur[0] = msg.pose.position.x
        self.y_cur[0]= msg.pose.position.y
        _, _, self.d_cur[0] = self.euler_from_quaternion(msg.pose.orientation)
        self.pose_state_cur[0] = True

    def optitracktb31_callback(self, msg):
        self.x_cur[1] = msg.pose.position.x
        self.y_cur[1]= msg.pose.position.y
        _, _, self.d_cur[1] = self.euler_from_quaternion(msg.pose.orientation)
        self.pose_state_cur[1] = True

    def optitracktb32_callback(self, msg):
        self.x_cur[2] = msg.pose.position.x
        self.y_cur[2]= msg.pose.position.y
        _, _, self.d_cur[2] = self.euler_from_quaternion(msg.pose.orientation)
        self.pose_state_cur[2] = True

    def optitracktb33_callback(self, msg):
        self.x_cur[3] = msg.pose.position.x
        self.y_cur[3]= msg.pose.position.y
        _, _, self.d_cur[3] = self.euler_from_quaternion(msg.pose.orientation)
        self.pose_state_cur[3] = True

    def optitracktb34_callback(self, msg):
        self.x_cur[4] = msg.pose.position.x
        self.y_cur[4]= msg.pose.position.y
        _, _, self.d_cur[4] = self.euler_from_quaternion(msg.pose.orientation)
        self.pose_state_cur[4] = True

    def optitracktb35_callback(self, msg):
        self.x_cur[5] = msg.pose.position.x
        self.y_cur[5]= msg.pose.position.y
        _, _, self.d_cur[5] = self.euler_from_quaternion(msg.pose.orientation)
        self.pose_state_cur[5] = True

    
    #####################update callback   10ms    ####################################  
    
    def update_callback(self):      
    
        #self.transition()
        self.mpc_implement()

        self.twisttb30.linear.x=self.v_cur[0]
        self.twisttb30.angular.z=self.w_cur[0]

        '''
        self.tb30_cmd_vel_pub.publish(self.twisttb30)
        ''' 
        
        #self.twisttb31.linear.x=self.v_cur[1]
        #self.twisttb31.angular.z=self.w_cur[1]
        '''
        self.tb31_cmd_vel_pub.publish(self.twisttb31)
        '''

        #self.twisttb32.linear.x=self.v_cur[2]
        #self.twisttb32.angular.z=self.w_cur[2]
        '''
        self.tb32_cmd_vel_pub.publish(self.twisttb32)
        '''
        

        #self.twisttb33.linear.x=self.v_cur[3]
        #self.twisttb33.angular.z=self.w_cur[3]
        '''
        self.tb33_cmd_vel_pub.publish(self.twisttb33)
        '''
        
        #self.twisttb34.linear.x=self.v_cur[4]
        #self.twisttb34.angular.z=self.w_cur[4]
        '''
        self.tb34_cmd_vel_pub.publish(self.twisttb34)
        '''

        #self.twisttb35.linear.x=self.v_cur[5]
        #self.twisttb35.angular.z=self.w_cur[5]
        '''
        self.tb35_cmd_vel_pub.publish(self.twisttb35)
        '''
        
    def pre_mpc(self):

        print('prepare mpc')

        for k in range(self.N):
            self.st = self.X[0:,k];  self.con = self.U[:,k]

            self.d12 = (self.st[0]-self.st[3])**2 + (self.st[1]-self.st[4])**2
            self.d13 = (self.st[0]-self.st[6])**2 + (self.st[1]-self.st[7])**2
            self.d14 = (self.st[0]-self.st[9])**2 + (self.st[1]-self.st[10])**2
            self.d15 = (self.st[0]-self.st[12])**2 + (self.st[1]-self.st[13])**2
            self.d16 = (self.st[0]-self.st[15])**2 + (self.st[1]-self.st[16])**2
            #
            self.d23 = (self.st[3]-self.st[6])**2 + (self.st[4]-self.st[7])**2
            self.d24 = (self.st[3]-self.st[9])**2 + (self.st[4]-self.st[10])**2
            self.d25 = (self.st[3]-self.st[12])**2 + (self.st[4]-self.st[13])**2
            self.d26 = (self.st[3]-self.st[15])**2 + (self.st[4]-self.st[16])**2
            #
            self.d34 = (self.st[6]-self.st[9])**2 + (self.st[7]-self.st[10])**2
            self.d35 = (self.st[6]-self.st[12])**2 + (self.st[7]-self.st[13])**2
            self.d36 = (self.st[6]-self.st[15])**2 + (self.st[7]-self.st[16])**2
            #
            self.d45 = (self.st[9]-self.st[12])**2 + (self.st[10]-self.st[13])**2
            self.d46 = (self.st[9]-self.st[15])**2 + (self.st[10]-self.st[16])**2
            #
            self.d56 = (self.st[12]-self.st[15])**2 + (self.st[13]-self.st[16])**2
            
            self.obj = self.obj  +  mtimes((self.st-self.P[self.n_states:]).T,mtimes(self.Q,(self.st-self.P[self.n_states:]))) + mtimes(self.con.T, mtimes(self.R, self.con))               # calculate obj
            #print("Obj: ", obj)
            self.st_next = self.X[0:,k+1] #;
            #print("st_next: ", st_next)
            f_value = self.f(self.st,self.con) #;
            #print("f_value: ", f_value)
            self.st_next_euler = self.st + (self.T*f_value)
            #print("st_next_euler: ", st_next_euler)
            self.g = vertcat(self.g, vertcat(vertcat(vertcat(\
    vertcat(vertcat(vertcat(vertcat(vertcat( \
    vertcat(vertcat(vertcat(vertcat(vertcat(vertcat(self.st_next-self.st_next_euler, self.d12), self.d13), \
    self.d14), self.d15), self.d16), self.d23), self.d24), self.d25), self.d26), \
    self.d34), self.d35), self.d36), self.d45), self.d46), \
    self.d56)
    # compute constraints
 
        # make the decision variable one column  vector
        self.OPT_variables = vertcat(reshape(self.X, self.n_states*(self.N+1),1), reshape(self.U, self.n_controls*self.N, 1))
        # print("OPT: ", OPT_variables)
        nlp_prob = {'f':self.obj, 'x':self.OPT_variables, 'g':self.g, 'p':self.P}
        # print("NLP: ", nlp_prob)

        opts = {'print_time': 0, 'ipopt':{'max_iter':2000, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6}}
        self.solver = nlpsol('solver','ipopt', nlp_prob, opts)

        self.args = {'lbg': np.matlib.repmat(horzcat(np.zeros((1,self.n_states)), np.matlib.repmat(np.array([self.dmin*self.dmin]), 1, self.M)), 1, self.N+1), \
        'ubg': np.matlib.repmat(horzcat(np.zeros((1,self.n_states)), np.matlib.repmat(np.array([np.Inf]), 1, self.M)), 1, self.N+1), \
        'lbx': np.concatenate((np.matlib.repmat(np.array([[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf]]),self.N+1,1),np.matlib.repmat(np.array([[self.v_min],[self.omega_min],[self.v_min],[self.omega_min], [self.v_min],[self.omega_min], [self.v_min],[self.omega_min], [self.v_min],[self.omega_min],[self.v_min],[self.omega_min]]),self.N,1)), axis=0), \
        'ubx': np.concatenate((np.matlib.repmat(np.array([[+10],[+10],[+np.Inf],[+10],[+10],[+np.Inf], [+10],[+10],[+np.Inf], [+10],[+10],[+np.Inf], [+10],[+10],[+np.Inf],[+10],[+10],[+np.Inf]]),self.N+1,1),np.matlib.repmat(np.array([[self.v_max],[self.omega_max],[self.v_max],[self.omega_max],[self.v_max],[self.omega_max],[self.v_max],[self.omega_max],[self.v_max],[self.omega_max],[self.v_max],[self.omega_max]]),self.N,1)), axis=0)}
        # print("args: ", args)

        self.sim_tim = 1000
        self.t0 = 0 #;
        self.x0 = np.array([[+0.866], [+0.5], [-2.618], [+0.0], [+1.0], [-1.57], [-0.866], [0.5], [-0.523],\
        [-0.866], [-0.5], [0.523], [0.0], [-1.0], [1.57], [+0.866], [-0.5], [+2.618]])
        self.Xr1 = np.array([[+0.866], [+0.5], [-2.618]])
        self.Xr2 = np.array([[+0.0], [+1.0], [-1.57]])                                                            # initial condition.
        self.Xr3 = np.array([[-0.866], [+0.5], [-0.523]])
        self.Xr4 = np.array([[-0.866], [-0.5], [+0.523]])                                                            # initial condition.
        self.Xr5 = np.array([[0.0], [-1.0], [1.57]])
        self.Xr6 = np.array([[0.866], [-0.5], [2.618]])                                                          # initial condition.

        #
        # print("Xr1: ", Xr1)
        # print("Xr2: ", Xr2)
        # print("x0: ", x0)
        #self.xs = np.array([[+1.0], [+1.0], [0.0]])                                                        # Reference posture.
        self.xs = np.array([[-0.866], [-0.5], [-2.618], [+0.0], [-1.0], [-1.57], [+0.866], [-0.5], [-0.523],\
        [+0.866], [+0.5], [+0.523], [0.0], [+1.0], [1.57] ,\
        [-0.866], [+0.5], [2.618]]) 

        self.xx = np.zeros((self.n_states, int(self.sim_tim/self.T)))
        # print("xx: ", xx[:,0:1])
        self.xx[:,0:1] = self.x0                                                                    # xx contains the history of states
        self.t = np.zeros(int(self.sim_tim/self.T))
        # print("t: ", np.shape(t))
        self.t[0] = self.t0

        self.u0 = np.zeros((self.n_controls,self.N));                                                             # two control inputs for each robot
        # print("u0: ", u0)
        self.X0 = np.transpose(repmat(self.x0,1,self.N+1))                                                         # initialization of the states decision variables
        #print("X0", self.X0)

                                                                           # Maximum simulation time
    def start_mpc(self):
        self.number = 0
        num = self.number
        print('start MPC', num)
        # Start MPC
        mpciter = 0
        self.xx1 = np.zeros((self.N+1,self.n_states,int(self.sim_tim/self.T)))
        self.u_cl = np.zeros((int(self.sim_tim/self.T),self.n_controls))

        #---
        # the main simulaton loop... it works as long as the error is greater
        # than 10^-6 and the number of mpc steps is less than its maximum
        # value.
        # main_loop = tic;
        #

        # Main Loop
        while (LA.norm(self.x0-self.xs) > 1e-1):
        # and (mpciter < sim_tim / T)

            self.args['p'] = np.concatenate((self.x0, self.xs), axis=0)                                # set the values of the parameters vector
            # print("args.p: ", args['p'])

            # initial value of the optimization variables
            self.args['x0']  = np.concatenate((reshape(np.transpose(self.X0),self.n_states*(self.N+1),1), reshape(np.transpose(self.u0),self.n_controls*self.N,1)), axis=0)
            # print("args: ", args['x0'])
            # print("args: ", args)

            self.sol = self.solver(x0=self.args['x0'], p=self.args['p'], lbx=self.args['lbx'], ubx=self.args['ubx'], lbg=self.args['lbg'], ubg=self.args['ubg'])
            # print("sol: ", sol['x'])

            self.solu = self.sol['x'][self.n_states*(self.N+1):]; self.solu_full = np.transpose(self.solu.full())
            self.u = np.transpose(reshape(self.solu_full, self.n_controls,self.N))                                    # get controls only from the solution
            #print("u: ", self.u)

            self.solx = self.sol['x'][0:self.n_states*(self.N+1)]; self.solx_full = np.transpose(self.solx.full())
            self.xx1[0:,0:self.n_states,mpciter] = np.transpose(reshape(self.solx_full, self.n_states,self.N+1))                               # get solution TRAJECTORY
            # print("xx1: ", xx1[:,0:3,mpciter])

            self.u_cl[mpciter,0:] = self.u[0:1,0:]
            #print("u_cl: ", self.u_cl[mpciter,0:])

            self.t[mpciter] = self.t0 #;

            # Apply the control and shift the solution
            self.t0, self.x0, self.u0 = shift(self.T, self.t0, self.x0, self.u, self.f)
            print(self.x0.shape)

            self.xx[0:,mpciter+1:mpciter+2] = self.x0
            #print("xx: ", self.xx)

            self.solX0 = self.sol['x'][0:self.n_states*(self.N+1)]; self.solX0_full = np.transpose(self.solX0.full())
            self.X0 = np.transpose(reshape(self.solX0_full, self.n_states,self.N+1))                                # get solution TRAJECTORY

            #print("u: ", self.u)
            # Shift trajectory to initialize the next step
            self.X0 = np.concatenate((self.X0[1:,0:self.n_states+1], self.X0[self.N-1:self.N,0:self.n_states+1]), axis=0)

            '''
            # Move Robot

            self.Xr[0]=self.Xr[0]+self.T*self.u_cl[mpciter,0]*math.cos(self.Xr[2])
            self.Xr[1]=self.Xr[1]+self.T*self.u_cl[mpciter,0]*math.sin(self.Xr[2])
            self.Xr[2]=self.Xr[2]+self.T*self.u_cl[mpciter,1]
            '''
            #self.x0 = np.array([self.Xr[0], self.Xr[1], self.Xr[2]])   

            #print(time.clock(), self.u_cl[mpciter,0], self.u_cl[mpciter,1], self.Xr[0], self.Xr[1], self.Xr[2])
            self.ne = LA.norm(self.x0-self.xs)
            # Stop Condition
            if self.ne < 0.1:
                print("mpciter, error: ", mpciter, self.ne)
                print("Robot has arrived to GOAL point!")
                self.save_gif()
                exit()

            print("mpciter, error: ", mpciter, self.ne)
            self.unicycle_simulation()
            mpciter = mpciter + 1
            num = num + 1
            self.number = num
        # main_loop_time = toc(main_loop);
        #self.ss_error = LA.norm(self.x0-self.xs)
        #print("Steady State Error: ", self.ss_error)
        #print("Closed-Loop Control: ", self.u_cl)
        # average_mpc_time = main_loop_time/(mpciter+1)
        # Draw_MPC_point_stabilization_v1 (t,xx,xx1,u_cl,xs,N,rob_diam)


        # tt = 0
        # tc0 = time.clock()
        # tc = 0.0
        # print("time: ", tc0)
    '''
    def mpc_implement(self):
        
        plt.figure(1)

        if self.buttonstart==0:
            axcut = plt.axes([0.0, 0.0, 1.0, 1.0])
            bcut = Button(axcut, 'MPC-based Collision-Free Point-To-Point Transiction', color='pink', hovercolor='pink')
            bcut.on_clicked(self.start)
            plt.pause(3) # default 5
            
        
        if self.buttonstart==1:
            
            self.pre_mpc()
            self.unicycle_simulation()
            self.start_mpc() 
             
    ''' 
    def mpc_implement(self):
        self.fig = plt.figure()

        self.ims = []

        self.pre_mpc()
        self.start_mpc()  
    
    def start(self,event):
        if self.buttonstart==0:
            self.buttonstart=1
        else :
            self.buttonstart=0 
        print(self.buttonstart)    

##########################################################################################################################          
    def unicycle_simulation(self):


        r = 0.15
     
        theta = np.arange(0, 2*np.pi, 0.01)
        self.x_cur[0] = self.x0[0]; self.x_cur[1] = self.x0[3]; self.x_cur[2] = self.x0[6]; 
        self.x_cur[3] = self.x0[9]; self.x_cur[4] = self.x0[12]; self.x_cur[5] = self.x0[15]; 
        self.y_cur[0] = self.x0[1]; self.y_cur[1] = self.x0[4]; self.y_cur[2] = self.x0[7]; 
        self.y_cur[3] = self.x0[10]; self.y_cur[4] = self.x0[13]; self.y_cur[5] = self.x0[16]; 
        self.theta_cur[0] = self.x0[2]; self.theta_cur[1] = self.x0[5]; self.theta_cur[2] = self.x0[8]; 
        self.theta_cur[3] = self.x0[11]; self.theta_cur[4] = self.x0[14]; self.theta_cur[5] = self.x0[17]; 
        #print(self.x_cur)
        #print(self.y_cur)

        for i in range(0, self.node_num, 1):
            self.x_body[i] = self.x_cur[i] + r * np.cos(theta)
            self.y_body[i] = self.y_cur[i] + r * np.sin(theta)
            
            self.x_head[i]=self.x_cur[i]+r*math.cos(self.theta_cur[i])
            self.y_head[i]=self.y_cur[i]+r*math.sin(self.theta_cur[i])

            self.x_left[i]=self.x_cur[i]-r*math.cos(self.theta_cur[i]+1.57)
            self.y_left[i]=self.y_cur[i]-r*math.sin(self.theta_cur[i]+1.57)

            self.x_right[i]=self.x_cur[i]-r*math.cos(self.theta_cur[i]-1.57)
            self.y_right[i]=self.y_cur[i]-r*math.sin(self.theta_cur[i]-1.57)
            
            self.x_curl[i][0]=self.x_head[i]
            self.x_curl[i][1]=self.x_cur[i]

            self.y_curl[i][0]=self.y_head[i]
            self.y_curl[i][1]=self.y_cur[i]
            
        
        for i in range(0, self.node_num, 1):
            
            plt.plot(self.x_head[i],self.y_head[i],self.colorArr[i],self.x_left[i],self.y_left[i],self.colorArr[i],self.x_right[i],self.y_right[i],self.colorArr[i],self.x_cur[i],self.y_cur[i],self.colorArr[i],marker='.')
            plt.plot(self.x_body[i],self.y_body[i],self.colorArr[i])
            ''''''
            #im = plt.plot(self.x_curl[i], self.y_curl[i], self.colorArr[i],self.x_body[i],self.y_body[i],self.colorArr[i], self.x_head[i],self.y_head[i],self.x_left[i],self.y_left[i],self.x_right[i],self.y_right[i],self.x_cur[i],self.y_cur[i],self.colorArr[i],marker='.')
            #self.ims.append(im)
        self.axis=1
        plt.plot(self.axis*3,self.axis*2,self.colorArr[i],marker='*')
        plt.plot(self.axis*3,-self.axis*2,self.colorArr[i],marker='*')
        plt.plot(-self.axis*3,self.axis*2,self.colorArr[i],marker='*')
        plt.plot(-self.axis*3,-self.axis*2,self.colorArr[i],marker='*')  

        plt.grid(True)
        plt.axis('equal')
        plt.axis([-self.axis*3,self.axis*3,-self.axis*2,self.axis*2])
        if self.number<=10000:
            plt.pause(0.01)
            plt.clf()
        else :
            plt.show()   

    def save_gif(self):
        self.ani = animation.ArtistAnimation(self.fig, self.ims, interval=200, repeat_delay=1000)
        #self.ani.save("~/gifs/test.gif",writer='pillow')
        self.ani.save("~/gifs/test.gif", fps=30) 
        print('save to ~/gifs/test.gif')

#####################################################
   
######################################################################################################################## 
 
########################################################################################################################       

    def get_key(self):
        # Print terminal message and get inputs
        print(terminal_msg)
        input_x = float(input("Input x: "))
        input_y = float(input("Input y: "))
        input_theta = float(input("Input theta: "))
        while input_theta > 180 or input_theta < -180:
            self.get_logger().info("Enter a value for theta between -180 and 180")
            input_theta = input("Input theta: ")
        input_theta = numpy.deg2rad(input_theta)  # Convert [deg] to [rad]

        settings = termios.tcgetattr(sys.stdin)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

        return input_x, input_y, input_theta

    """*******************************************************************************
    ** Below should be replaced when porting for ROS 2 Python tf_conversions is done.
    *******************************************************************************"""
    def euler_from_quaternion(self, quat):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quat = [x, y, z, w]
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w*y - z*x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def callback_odom(odom):
    global Xr
    xr = odom.pose.pose.position.x
    yr = odom.pose.pose.position.y
    qz = odom.pose.pose.orientation.z
    qw = odom.pose.pose.orientation.w
    th = 2*np.arcsin(qz)
    thr = modify(th)
    Xr = np.array([[xr], [yr], [thr]])
    # Xr = np.array([[yr], [xr], [2*np.arcsin(qz)]])

def modify(th):
    if th >= -math.pi and th < 0:
        th_modified = th + 2*math.pi
    else:
        th_modified = th
    return th_modified

def shift(T, t0, x0, u, f):
    st = x0
    #print(x0)
    con = np.transpose(u[0:1,0:])
    f_value = f(st, con)
    st = st + (T*f_value)
    x0 = st.full()
    #print(x0)
    t0 = t0 + T
    ushape = np.shape(u)
    u0 = np.concatenate(( u[1:ushape[0],0:],  u[ushape[0]-1:ushape[0],0:]), axis=0)
    return t0, x0 ,u0
'''
def shift(T, t0, u):
    # st = x0
    con = np.transpose(u[0:1,0:])
    # f_value = f(st, con)
    # st = st + (T*f_value)
    # x0 = st.full()
    t0 = t0 + T
    ushape = np.shape(u)
    u0 = np.concatenate(( u[1:ushape[0],0:],  u[ushape[0]-1:ushape[0],0:]), axis=0)
    return t0, u0
'''