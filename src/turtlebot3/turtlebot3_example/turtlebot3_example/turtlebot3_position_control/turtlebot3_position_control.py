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
        self.count=0.0
        self.position_init=1

        self.axis=5
        self.vw=self.axis/5

        self.Kv=0.22/2*self.vw
        self.Kw=2.84/2*self.vw/math.pi
        self.B_oz=0.0
        self.B_of=0.0
        #self.B_safed=0.1
        self.B_safed=0.08
        self.buttonstart=0

        ######## initialize mpc ##########
        self.T = 0.1  # [s]
        self.N = 100                                                                          # prediction horizon
        self.rob_diam = 0.15

        self.v_max = +0.2; self.v_min = -self.v_max  ## max_vel
        self.omega_max = +math.pi/4; self.omega_min = -self.omega_max

        ##casadi attributes
        x = SX.sym('x'); y = SX.sym('y'); theta = SX.sym('theta')
        states = np.array([[x], [y], [theta]]); self.n_states = len(states)         #print("states: ", states)

        v = SX.sym('v'); omega = SX.sym('omega')
        controls = np.array([[v],[omega]]); self.n_controls = len(controls)
        rhs = np.array([[v*np.cos(theta)],[v*np.sin(theta)],[omega]])                   # system r.h.s
        #print("rhs: ", rhs)

        self.f = Function('f',[states,controls],[rhs])                                       # nonlinear mapping function f(x,u)
        #print("Function :", f)

        self.U = SX.sym('U',self.n_controls,self.N) # ;                                                   # Decision variables (controls)
        self.P = SX.sym('P',self.n_states + self.n_states) #print("U: ", U) ; #print("P: ", P)                                            # parameters (which include the initial state and the reference state)
        
        self.X = SX.sym('X',self.n_states,(self.N+1)) #;# A vector that represents the states over the optimization problem.      
        #print("X: ", X)

        self.obj = 0                                                                        # Objective function
                                                                                # constraints vector
        self.Q = np.zeros((3,3)); self.Q[0,0] = 1; self.Q[1,1] = 5; self.Q[2,2] = 0.1                           # weighing matrices (states)
        self.R = np.zeros((2,2)); self.R[0,0] = 0.5; self.R[1,1] = 0.05                                  # weighing matrices (controls)
        #print("Q: ", Q)
        #print("R: ", R)


        # obstacle info
        self.obs1_r = 0.15   # obstacle's diameter
        self.obs1_x = 0.4
        self.obs1_y = 1.1

        self.st  = self.X[:,0]                                                                    # initial state
        #print("st: ", st)

        self.g = self.st-self.P[0:3]                                                                   # initial condition constraints
        #print("g: ", g)

        ######## initialize plot #########
        self.r_o=np.array([0.2 for x in range(self.node_num)])
        self.d_d=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        self.d_1=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        self.d_2=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        self.d_t=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        
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

        self.x_o_body=[0.0 for x in range(self.node_num)]
        self.y_o_body=[0.0 for x in range(self.node_num)]

        self.x_tar_body=[0.0 for x in range(self.node_num)]
        self.y_tar_body=[0.0 for x in range(self.node_num)]


        self.v_cur=np.array([0.0 for x in range(self.node_num)])
        self.w_cur=np.array([0.0 for x in range(self.node_num)])

        
        
        self.pose_state_cur=np.array([0.0 for x in range(self.node_num)])

        self.odom = Odometry()
        self.twisttb30 = Twist()
        #self.twisttb31 = Twist()
        #self.twisttb32 = Twist()
        #self.twisttb33 = Twist()
        #self.twisttb34 = Twist()
        #self.twisttb35 = Twist()

        self.gama=np.array([1.000 for x in range(self.node_num)])
       
        self.x_target=np.array([0.0 for x in range(self.node_num)])   
        self.y_target=np.array([0.0 for x in range(self.node_num)])
        self.d_target=np.array([0.0 for x in range(self.node_num)])


        self.x_cur=0     #[2.0,2.0,2.0,2.0,2.0,2.0]
        self.y_cur=0 
        self.d_cur=1
        self.thr = self.d_cur
        self.Xr = np.array([[self.x_cur], [self.y_cur], [self.thr]])

        
##########################################################################################################
        self.colorArr = ['r','g','b','c','m','k','gray','tan','pink','navy','r','g','b','c','m','k','gray','tan','pink','navy']
        
###########################################################################################################
        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialise publishers
        '''
        self.tb30_cmd_vel_pub = self.create_publisher(Twist,'tb3_0/cmd_vel', qos)
        self.tb31_cmd_vel_pub = self.create_publisher(Twist,'tb3_1/cmd_vel', qos)
        self.tb32_cmd_vel_pub = self.create_publisher(Twist,'tb3_2/cmd_vel', qos)
        self.tb33_cmd_vel_pub = self.create_publisher(Twist,'tb3_3/cmd_vel', qos)
        self.tb34_cmd_vel_pub = self.create_publisher(Twist,'tb3_4/cmd_vel', qos)
        self.tb35_cmd_vel_pub = self.create_publisher(Twist,'tb3_5/cmd_vel', qos)
        '''

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
            self.st = self.X[:,k];  self.con = self.U[:,k]
            #print("st: ", st); print("con: ", con); #print("R: ", R)
            
            self.obj = self.obj  +  mtimes((self.st-self.P[3:6]).T,mtimes(self.Q,(self.st-self.P[3:6]))) + mtimes(self.con.T, mtimes(self.R, self.con))               # calculate obj
            #print("Obj: ", obj)
            self.st_next = self.X[:,k+1] #;
            #print("st_next: ", st_next)
            f_value = self.f(self.st,self.con) #;
            #print("f_value: ", f_value)
            self.st_next_euler = self.st + (self.T*f_value)
            #print("st_next_euler: ", st_next_euler)
            self.g = vertcat(self.g, self.st_next-self.st_next_euler)  # compute constraints
            # construct obstacle constraints
            self.g = vertcat(self.g, sqrt((self.X[0,k] - self.obs1_x)**2 + (self.X[1,k] - self.obs1_y)**2)-self.rob_diam-self.obs1_r)  
        # print("g: ", g); print(g.shape
 
        # make the decision variable one column  vector
        self.OPT_variables = vertcat(reshape(self.X, 3*(self.N+1),1), reshape(self.U, 2*self.N, 1))
        # print("OPT: ", OPT_variables)
        nlp_prob = {'f':self.obj, 'x':self.OPT_variables, 'g':self.g, 'p':self.P}
        # print("NLP: ", nlp_prob)

        opts = {'print_time': 0, 'ipopt':{'max_iter':2000, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6}}
        self.solver = nlpsol('solver','ipopt', nlp_prob, opts)

        self.args = {'lbg': np.concatenate((np.array([[0.0],[0.0],[0.0]]), np.matlib.repmat(np.array([[0.0],[0.0],[0.0],[0.05]]),self.N,1)), axis=0), 'ubg': np.concatenate((np.array([[0.0],[0.0],[0.0]]), np.matlib.repmat(np.array([[0.0],[0.0],[0.0],[np.Inf]]),self.N,1)), axis=0), \
        'lbx': np.concatenate((np.matlib.repmat(np.array([[-10],[-10],[-2*math.pi]]),self.N+1,1),np.matlib.repmat(np.array([[self.v_min],[self.omega_min]]),self.N,1)), axis=0), \
        'ubx': np.concatenate((np.matlib.repmat(np.array([[+10],[+10],[+2*math.pi]]),self.N+1,1),np.matlib.repmat(np.array([[self.v_max],[self.omega_max]]),self.N,1)), axis=0)}
        # print("args: ", args) print(args['lbg'].shape)

        self.sim_tim = 1000
        self.t0 = 0 #;
        self.x0 = np.array([[0.0], [0.0], [0.0]])                                                             # initial condition.

        # print("x0: ", x0)
        #self.xs = np.array([[+1.0], [+1.0], [0.0]])                                                        # Reference posture.

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
        num = 0
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
        goal_idx = 1
        ng = 10
        mpc_start = True

        # Main Loop
        while mpc_start == True:
        # and (mpciter < sim_tim / T)
             # Goal points
            if goal_idx == 1:
                self.xs = np.array([[1.5], [+1.5], [0.0]]) # reference pose
                #print('design form')
            elif goal_idx == 2:
                self.xs = np.array([[+0.0], [0.75], [-1.57]]) # reference pose
                #print('design form')
            elif goal_idx == 3:
                self.xs = np.array([[-0.5], [+0.5], [3.14]]) # reference pose
                #print('design form')
            elif goal_idx == 4:
                self.xs = np.array([[-0.5], [-0.75], [+0.785]]) # reference pose
                #print('design form')
            elif goal_idx == 5:
                self.xs = np.array([[0.75], [-0.75], [-0.785]]) # reference pose
                #print('design form')
            elif goal_idx == 6:
                self.xs = np.array([[0.0], [0.0], [0.0]]) # reference pose
                #print('design form')

            self.args['p'] = np.concatenate((self.x0, self.xs), axis=0)                                # set the values of the parameters vector
            # print("args.p: ", args['p'])

            # initial value of the optimization variables
            self.args['x0']  = np.concatenate((reshape(np.transpose(self.X0),3*(self.N+1),1), reshape(np.transpose(self.u0),2*self.N,1)), axis=0)
            # print("args: ", args['x0'])
            # print("args: ", args)

            self.sol = self.solver(x0=self.args['x0'], p=self.args['p'], lbx=self.args['lbx'], ubx=self.args['ubx'], lbg=self.args['lbg'], ubg=self.args['ubg'])
            # print("sol: ", sol['x'])

            self.solu = self.sol['x'][3*(self.N+1):]; self.solu_full = np.transpose(self.solu.full())
            self.u = np.transpose(reshape(self.solu_full, 2,self.N))                                    # get controls only from the solution
            #print("u: ", self.u)

            self.solx = self.sol['x'][0:3*(self.N+1)]; self.solx_full = np.transpose(self.solx.full())
            self.xx1[0:,0:3,mpciter] = np.transpose(reshape(self.solx_full, 3,self.N+1))                               # get solution TRAJECTORY
            # print("xx1: ", xx1[:,0:3,mpciter])

            self.u_cl[mpciter,0:] = self.u[0:1,0:]
            #print("u_cl: ", self.u_cl[mpciter,0:])

            self.t[mpciter] = self.t0 #;

            # Apply the control and shift the solution
            self.t0, self.x0, self.u0 = shift(self.T, self.t0, self.x0, self.u, self.f)
            print(self.x0)

            self.xx[0:,mpciter+1:mpciter+2] = self.x0
            #print("xx: ", self.xx)

            self.solX0 = self.sol['x'][0:3*(self.N+1)]; self.solX0_full = np.transpose(self.solX0.full())
            self.X0 = np.transpose(reshape(self.solX0_full, 3,self.N+1))                                # get solution TRAJECTORY

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
                goal_idx = goal_idx + 1
                if goal_idx > ng:
                    mpc_start = False
                    print("Robot has arrived to GOAL point!")

            print("mpciter, error: ", mpciter, self.ne)
            mpciter = mpciter + 1
            num = num + 1

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
        self.pre_mpc()
        self.start_mpc() 
        self.unicycle_simulation() 
    '''
    def start(self,event):
        if self.buttonstart==0:
            self.buttonstart=1
        else :
            self.buttonstart=0 
        print(self.buttonstart)    

##########################################################################################################################          
    def unicycle_simulation(self):
        roo=np.array([0.0])
        r = 0.2
        ro=self.B_safed+r
        sumx=0
        sumy=0
        
        theta = np.arange(0, 2*np.pi, 0.01)
        self.x_cur = self.x0[0]
        self.y_cur = self.x0[1]

        self.x_body = self.x_cur + r * np.cos(theta)
        self.y_body = self.y_cur + r * np.sin(theta)

        self.x_tar_body = self.x_target + 0.2 * np.cos(theta)
        self.y_tar_body = self.y_target + 0.2 * np.sin(theta)

        self.x_o_body = self.x_cur + roo * np.cos(theta)
        self.y_o_body = self.y_cur + roo * np.sin(theta)

            
        self.x_head=self.x_cur+r*math.cos(self.d_cur)
        self.y_head=self.y_cur+r*math.sin(self.d_cur)

        self.x_left=self.x_cur-r*math.cos(self.d_cur+1.57)
        self.y_left=self.y_cur-r*math.sin(self.d_cur+1.57)

        self.x_right=self.x_cur-r*math.cos(self.d_cur-1.57)
        self.y_right=self.y_cur-r*math.sin(self.d_cur-1.57)

        self.x_curl[0]=self.x_head
        self.x_curl[1]=self.x_cur

        self.y_curl[0]=self.y_head
        self.y_curl[1]=self.y_cur


        plt.plot(self.x_curl, self.y_curl,self.colorArr)
        
        plt.plot(self.x_head,self.y_head,self.colorArr,self.x_left,self.y_left,self.colorArr,self.x_right,self.y_right,self.colorArr,self.x_cur,self.y_cur,self.colorArr,marker='.')
        
        plt.plot(self.x_body,self.y_body,self.colorArr,self.x_tar_body,self.y_tar_body,self.colorArr)

        plt.plot(self.x_target,self.y_target,self.colorArr)

        plt.plot(self.axis*3,self.axis*2,self.colorArr,marker='*')
        plt.plot(self.axis*3,-self.axis*2,self.colorArr,marker='*')
        plt.plot(-self.axis*3,self.axis*2,self.colorArr,marker='*')
        plt.plot(-self.axis*3,-self.axis*2,self.colorArr,marker='*')
            
        #plt.plot(self.x_cur_trajectory[i],self.y_cur_trajectory[i],self.colorArr[i])       
        '''
        plt.text(self.x_cur[i], self.y_cur[i]-self.axis/10, '%.2f' %self.x_cur[i], ha='center', va= 'bottom',fontsize=6,color = self.colorArr[i])
        plt.text(self.x_cur[i], self.y_cur[i]-self.axis/5, '%.2f' %self.y_cur[i], ha='center', va= 'bottom',fontsize=6,color = self.colorArr[i])
        '''
        x=-self.axis*3
        y=self.axis*2
        d1=self.axis/2
        d2=self.axis/10 #0.5
        
        fsize=9
       
        x=-self.axis*3
        y=0
        d3=self.axis/10
        fsize=10

        
        plt.grid(True)
        plt.axis('equal')
        plt.axis([-self.axis*3,self.axis*3,-self.axis*2,self.axis*2])

        if self.count<=10000:
            plt.pause(0.01)
            plt.clf()
            self.count=self.count+1.0
        else :
            plt.show()   

             
########################################################################################################################
   
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
    print(x0)
    con = np.transpose(u[0:1,0:])
    f_value = f(st, con)
    st = st + (T*f_value)
    x0 = st.full()
    print(x0)
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