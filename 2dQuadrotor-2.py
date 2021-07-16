#!/usr/bin/env python
# coding: utf-8

# In[4]:


# setup matplotlib for nice display in Jupyter
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.animation as animation
import IPython

np.set_printoptions(precision=5,linewidth=120,suppress=True)

import cvxopt

def animate_2dquadrotor(z, dt):
    """
    This function makes an animation showing the behavior of a 2d quadrotor
    takes as input the result of a simulation (with dt=0.01s)
    """
    
    min_dt = 0.1
    if(dt < min_dt):
        steps = int(min_dt/dt)
        use_dt = int(min_dt * 1000)
    else:
        steps = 1
        use_dt = int(dt * 1000)
    
    #what we need to plot
    plotz = z[:,::steps]
    
    fig = mp.figure.Figure(figsize=[8.5,6.5])
    mp.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=[-10,10], ylim=[-8,8])
    ax.grid()
    
    list_of_lines = []
    
    #create the simplified quadrotor
    line, = ax.plot([], [], 'k', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'ro', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'go', lw=2)
    list_of_lines.append(line)

    
    quad_radius = 0.2
    
    def animate(i):
        for l in list_of_lines: #reset all lines
            l.set_data([],[])
        
        x_quad1 = plotz[0,i] + np.cos(plotz[3,i])
        y_quad1 = plotz[1,i] - np.sin(plotz[3,i])
        x_quad2 = plotz[0,i] - np.cos(plotz[3,i])
        y_quad2 = plotz[1,i] + np.sin(plotz[3,i])
        
               
        list_of_lines[0].set_data([x_quad1, x_quad2], [y_quad1, y_quad2])
        list_of_lines[1].set_data([x_quad1,x_quad1], [y_quad1,y_quad1])
        list_of_lines[2].set_data([x_quad2, x_quad2], [y_quad2, y_quad2])
        
        return list_of_lines
    
    def init():
        return animate(0)


    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(plotz[0,:])),
        interval=use_dt, blit=True, init_func=init)
    plt.close(fig)
    plt.close(ani._fig)
    IPython.display.display_html(IPython.core.display.HTML(ani.to_html5_video()))

class Quadrotor2D:
    """
    This class describes a 2D quadrotor model 
    """
    
    def __init__(self):
        """
        constructor of the class, takes as input desired discretization number
        for x (angle), v (angular velocity) and u (control) and the maximum control
        """
        #gravity constant
        self.g=9.81

        #integration step
        self.dt = 0.01
        
        #we define radius and masses
        self.r = 0.15
        self.m = 5.0
        
        # Define constant A, B for system dynamic equation
        self.A = np.array([[1.,0.,0.,self.dt,0.,0.],[0.,1.,0.,0.,self.dt,0.],[0.,0.,1.,0.,0.,self.dt],[0.,0.,-self.dt*self.g,1.,0.,0.],
                   [0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,1.]])
        self.B = np.array([0.,0.,0.,0.,0.,0.,0.,0.,self.dt/self.m,0.,0,self.dt/(self.m*self.r**2)])
        self.B = self.B.reshape(6,2)
            
    def next_state(self,z,u):
        """
        Inputs:
        z: state of the 2d quadrotor syste as a numpy array (x,y,theta,v_x,v_y,omega)
        u: thrust forces as a numpy array (u1,u2)
        
        Output:
        the new state of the pendulum as a numpy array
        """
        x = z[0]
        y = z[1]
        th = z[2]
        v_x = z[3]
        v_y = z[4]
        om = z[5]
#         x_next = x + self.dt * v_x + 0.5 * self.dt**2 * ((u[0]* np.sin(th))/self.m)
#         y_next = y + self.dt * v_y + 0.5 * self.dt**2 * (-self.g + (u[0]*np.cos(th))/self.m)
#         th_next = th + self.dt * om + 0.5 * self.dt**2 * u[1]/(self.m*self.r**2)
        x_next = x + self.dt * v_x 
        y_next = y + self.dt * v_y 
        th_next = th + self.dt * om 
        v_x_next = v_x - self.dt * ((u[0]* np.sin(th))/self.m)
        v_y_next = v_y + self.dt * (-self.g + (u[0]*np.cos(th))/self.m)
        w_next = om + self.dt * u[1]/(self.m*self.r**2)
        z = np.array([x_next,y_next,th_next,v_x_next,v_y_next,w_next])
        return z
    
    def simulate(self, z0, K, uff, horizon_length):
        """
        This function simulates the quadrotor of horizon_length steps from initial state x0
        
        Inputs:
        z0: the initial conditions of the 2D quadrotor as a numpy array (x,y,theta,v_x,v_y,omega)
        controller: a function that takes a state z as argument and index i of the time step and returns a control u
        horizon_length: the horizon length
        
        Output:
        z[6x(time_horizon+1)] and u[2,time_horizon] containing the time evolution of states and controls
        """
        uoff = np.array([[self.g * self.m, 0.]]) # To simulate original quadrotor dynamics remember to add u0 = [mg 0]!
        z=np.empty([6, horizon_length+1])
        z[:,0] = z0
        u=np.empty([2,horizon_length])
        for i in range(horizon_length):
            u[:,i] = uoff + K[i].dot(z[:,i]) + uff[i]
#             u[:,i] = uoff + K[i].dot(z[:,i]) 
#             u[:,i] = controller(z[:,i],i)
            z[:,i+1] = self.next_state(z[:,i], u[:,i])
        return z, u

    def LQ_simulate(self, z0, K, uff, horizon_length):
        """
        This function simulates the quadrotor of horizon_length steps from initial state x0
        
        Inputs:
        z0: the initial conditions of the 2D quadrotor as a numpy array (x,y,theta,v_x,v_y,omega)
        controller: a function that takes a state z as argument and index i of the time step and returns a control u
        horizon_length: the horizon length
        
        Output:
        z[6x(time_horizon+1)] and u[2,time_horizon] containing the time evolution of states and controls
        """
        uoff = np.array([[self.g * self.m, 0.]]) # To simulate original quadrotor dynamics remember to add u0 = [mg 0]!
        z=np.empty([6, horizon_length+1])
        z[:,0] = z0
        u=np.empty([2,horizon_length])
        for i in range(horizon_length):
            u[:,i] = uoff + K[i].dot(z[:,i]) + uff[i]
            z[:,i+1] = self.A.dot(z[:,i]) + self.B.dot(u[:,i]) #here use approximation system dynamics to simulate
        return z, u
    
def solve_ricatti_equations(A,B,Q,R,horizon_length):
    """
    This function solves the backward Riccatti equations for regulator problems of the form
    min sum(xQx + uRu) + xQx subject to xn+1 = Axn + Bun
    
    Arguments:
    A, B, Q, R: numpy arrays defining the problem
    horizon_length: length of the horizon
    
    Returns:
    P: list of numpy arrays containing Pn from 0 to N
    K: list of numpy arrays containing Kn from 0 to N-1
    """
    P = [] #will contain the list of Ps from N to 0
    K = [] #will contain the list of Ks from N-1 to 0

    P.append(Q) #PN
    
    for i in range(horizon_length):
        Knew = -1.0 * np.linalg.inv(B.transpose().dot(P[i]).dot(B) + R).dot(B.transpose()).dot(P[i]).dot(A)
        Pnew = Q + A.transpose().dot(P[i]).dot(A) + A.transpose().dot(P[i]).dot(B).dot(Knew)
        K.append(Knew)
        P.append(Pnew)
    
    # since we went backward we return reverted lists
    return P[::-1],K[::-1]


def check_controllability(A,B):
    """
    This function check  the controllabilitystate for system
    c=[B AB A^2B A^3B A^4B A^5B]
    """
    c=np.concatenate([B, np.dot(A, B), np.dot(A, A).dot(B),np.dot(A, A.dot(A)).dot(B), np.dot(A.dot(A), A.dot(A)).dot(B), np.dot(A.dot(A), A.dot(A).dot(A)).dot(B)], axis=1)
    R=np.linalg.matrix_rank(c)
    print('rank is',R)
    if R < np.linalg.matrix_rank(A):
        print('is not controllable')
    else:print('is controllable')

        
def plot_results(z,u,K,animate):
    """
    This function plots the results. It displays (plane & angular) positions and velocities of the quadrotorvfor all states.
    Then it displays the evolution of two thrusts as control inputs. 
    It's optional to display the feedback control gains K at each step when applying LQR controllers.
    Finally it shows an animation of the quadrotor's motions.
    """
    
    t = np.linspace(0,quadrotor.dt*(horizon_length),horizon_length+1)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t,z[0,:])
    plt.ylabel('horizontal position x')
    plt.title('Position')
    plt.subplot(3,1,2)
    plt.plot(t,z[1,:])
    plt.ylabel('vertical position y')
    plt.subplot(3,1,3)
    plt.plot(t,z[2,:])
    plt.ylabel('angular position theta')
    plt.xlabel('Time')
   

    plt.figure() 
    plt.subplot(3,1,1)
    plt.plot(t,z[3,:])
    plt.ylabel('$v_x$')
    plt.title('Velocity')
    plt.subplot(3,1,2)
    plt.plot(t,z[4,:])
    plt.ylabel('$v_y$')
    plt.subplot(3,1,3)
    plt.plot(t,z[5,:])
    plt.ylabel(r'$\omega$')
    plt.xlabel('Time')

    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t[:-1],u[0,:])
    plt.ylabel('$u_1$')
    plt.title('Control thrusts')
    plt.subplot(2,1,2)
    plt.plot(t[:-1],u[1,:])
    plt.ylabel('$u_2$')
    plt.xlabel('Time')
   

    # we can plot the computed gain
    # Note that in this case K is a 2 by 6 matrix with 12 elements
    K = np.array(K)         
    plt.figure(figsize=[8.5,4.5])
    # first row of K
    plt.subplot(1,2,1)
    plt.plot(K[:,0,0],'-o', markersize=6)
    plt.plot(K[:,0,1],'-o', markersize=6)
    plt.plot(K[:,0,2],'-o', markersize=6)
    plt.plot(K[:,0,3],'-o', markersize=6)
    plt.plot(K[:,0,4],'-o', markersize=6)
    plt.plot(K[:,0,5],'-o', markersize=6)
    plt.legend(["K00","K01","K02","K03","K04","K05"],fontsize = 6,frameon = False,loc ='best')
    # plt.legend(["K11"],fontsize = 6,frameon = False,loc ='best')
    plt.ylabel('K1')
    plt.xlabel('Time')
    plt.title('Feedback gains ')

    # second row of K
    plt.subplot(1,2,2)
    plt.plot(K[:,1,0],'-o', markersize=6)
    plt.plot(K[:,1,1],'-o', markersize=6)
    plt.plot(K[:,1,2],'-o', markersize=6)
    plt.plot(K[:,1,3],'-o', markersize=6)
    plt.plot(K[:,1,4],'-o', markersize=6)
    plt.plot(K[:,1,5],'-o', markersize=6)
    plt.legend(["K10","K11","K12","K13","K14","K15"],fontsize = 6,frameon = False,loc ='best')
    # plt.legend(["K11"],fontsize = 6,frameon = False,loc ='best')
    plt.ylabel('K2')
    plt.xlabel('Time')
    plt.title('Feedback gains ')
    
    if animate:
        animate_2dquadrotor(z, quadrotor.dt)


# In[5]:


# we show an example on how to simulate the cartpole and display its behavior

# create a 2dquadrotor
quadrotor = Quadrotor2D()
check_controllability(quadrotor.A, quadrotor.B)

# we simulate the 2dqudrotor when do nothing
z0 = np.array([0,0,0,0.,0.,0])
horizon_length = 1000
# useless controller
k = np.array([[0,0,0,0.,0.,0],[0,0,0,0.,0.,0]])
uf = np.array([0.,0.])
K = []
uff = []
for i in range(horizon_length):
    K.append(k)
    uff.append(uf)

z,u = quadrotor.simulate(z0, K, uff, horizon_length)
plot_results(z, u, K, animate=True)


# # make an animation of the cart-pole
# animate_2dquadrotor(z,quadrotor.dt)


# In[6]:


## First we use LQR to stabilize the quadrotor system around the fixed poiont(all zeroes)
#initial state
z0_1 = np.array([5, 6, 0.5, 0.,0.,0])
horizon_length = 1000

Q=100*np.eye(6)
R=0.1*np.eye(2)

P1,K1 = solve_ricatti_equations(quadrotor.A, quadrotor.B, Q, R, horizon_length)

# here we need to add the feedfordward command uff computed due to the change of variables
# in this case since the stabilized point z0 is a zero vector
uff1 = []
for i in range(horizon_length):
    uff1.append(-K1[i].dot(np.array([0,0,0,0.,0.,0]))) 
    
# def feedforward_controller(z,i):
#     u = K1[i].dot(z) + uff[i] 
#     return u

z_1,u_1 = quadrotor.simulate(z0_1, K1, uff1, horizon_length)

plot_results(z_1,u_1,K1,animate=True)


# In[7]:


## Here use Quadratic Program Solver to compute the optimal control thrusts for the quadrotor

def QPsolver(A,B,Q,R,x0,horizon_length, u_len, u_max, z_max, z_des):
    """
    Here we want to find the optimal control path following a designated trajectory using a QP solver
    
    Inputs: 
    A,B system dynamics
    Q,R: numpy arrays for the quadratic cost
    z_des: the desired trajectory
    u_max: the thrusts as controls u = [u1 u2].T
    horizon_length: the number of steps
    
    returns: the state and control trajectories
    """    
    
    # the lenght of the state vector for the full trajectory is
    num_states = z0.size*horizon_length
    # the length of the control vector for the full trajectory is
    num_control = u_len*horizon_length
    # the total number of variables is
    num_vars = num_states + num_control

    Qtilde = np.zeros([num_vars, num_vars])
    
    Atilde = np.zeros([num_states, num_vars])
    btilde = np.zeros([num_states])
    btilde[0:z0.size] = -A.dot(z0)
    
    qtilde = np.zeros([num_vars])
    for i in range(horizon_length):
        Qtilde[z0.size * i:z0.size * (i + 1), z0.size * i:z0.size * (i + 1)] = Q
        Qtilde[num_states + u_len * i:num_states + u_len * (i + 1), 
               num_states + u_len * i:num_states + u_len * (i + 1)] = R
        qtilde[i * z0.size:(i + 1) * z0.size] = -z_des[:, i].dot(Q)
        Atilde[z0.size * i:z0.size * (i + 1), num_states + u_len * i:num_states + u_len * (i + 1)] = B

        if i > 0:
            Atilde[z0.size * i:z0.size * (i + 1), z0.size * (i - 1):z0.size * (i + 1)] = np.hstack((A, -np.eye(z0.size)))
        else:
            Atilde[z0.size * i:z0.size * (i + 1), z0.size * i:z0.size * (i + 1)] = -np.eye(z0.size)
 
    # transform numpy arrays into cvxopt compatible matrices
    P = cvxopt.matrix(Qtilde)
    q = cvxopt.matrix(qtilde)
                    
    A = cvxopt.matrix(Atilde)
    b = cvxopt.matrix(btilde)

    # max positions
    G1 = np.zeros([2*num_states,num_vars])
    G1[:,0:num_states] = np.vstack((np.eye(num_states),-np.eye(num_states)))
    h1 = np.vstack((np.tile(z_max,horizon_length),np.tile(z_max,horizon_length))).flatten()
#     G = np.zeros([num_vars,num_vars])
#     G1 = np.eye(num_states)
#     print(G1)
#     G[:,0:num_states] = np.vstack((G1,np.zeros([num_control,num_states])))
#     g = np.vstack(np.tile(z_max,num_states)).flatten()

    # max forces(controls)
    G2 = np.zeros([2*num_control,num_vars])
    G2[:,num_states:] = np.vstack((np.eye(num_control),-np.eye(num_control)))
    h2 = np.vstack((np.tile(u_max,horizon_length),np.tile(u_max,horizon_length))).flatten()
    G = np.vstack((G1,G2))
    h = np.hstack((h1.T,h2.T))     # 1 by   2*(num_states + num_control) row vector
#     print(btilde.shape)
#     print(np.size(h1,0))
#     print(h2.shape)
#     print(G.shape)
#     print(h.shape)

#     G2 = np.eye(num_control)
#     G[:,num_states:] = np.vstack((np.zeros([num_states,num_control]),G2))
#     tt = np.vstack(np.tile(u_max,num_control)).flatten()
#     h = np.hstack((g.T,tt.T))
    G = G.astype('float')
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
  
    sol = cvxopt.solvers.qp(P,q,G,h,A,b)      
#     sol = cvxopt.solvers.qp(P,q,A,b)
    z = np.array(sol['x'])
    
    ## we assume that the problem was constructed with the states first (x0,x1,...)
    ## and then the control inputs (u0, u1, ...)
    
#     # we extract the control trajectory as a 2 * horizon_length array
    u = (z[num_states:].reshape(horizon_length,u_len)).transpose() 
#     u = x[num_states:]
    print(u.size)
    # we extract the state trajectory and add the initial condition
    z = z[0:num_states]
    z = np.vstack((z0, z.reshape([horizon_length, z0.size]))).transpose()
       
    print(z.size)
    return z, u


def animate_2Dquadrotor_2(z,z_des,dt):
    """
    This function makes an animation showing the behavior of the cart-pole
    takes as input the result of a simulation (with dt=0.01s)
    """
    
    min_dt = 0.1
    if(dt < min_dt):
        steps = int(min_dt/dt)
        use_dt = int(min_dt * 1000)
    else:
        steps = 1
        use_dt = int(dt * 1000)
    
    #what we need to plot
    zdes = np.hstack((z_des,(z_des[:,-1]).reshape(6,1)))
    plotz = z[:,::steps]
    plotzd = zdes[:,::steps]
    
    fig = mp.figure.Figure(figsize=[8.5,6.5])
    mp.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=[-2,30], ylim=[-6,8])
    ax.grid()
    
    list_of_lines = []
    
    #create the quadrotor
    line, = ax.plot([], [], 'k', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'ro', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'go', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'bo', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'b', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'C1o', lw=2)
    list_of_lines.append(line)
    line, = ax.plot([], [], 'C1--', lw=2)
    list_of_lines.append(line)
    
#     quad_radius = 0.2
    x_data,y_data = [],[]
    x_goal,y_goal = [],[]
    
    def animate(i):
        for l in list_of_lines: #reset all lines
            l.set_data([],[])
        
        x_quad1 = plotz[0,i] + np.cos(plotz[3,i])
        y_quad1 = plotz[1,i] - np.sin(plotz[3,i])
        x_quad2 = plotz[0,i] - np.cos(plotz[3,i])
        y_quad2 = plotz[1,i] + np.sin(plotz[3,i])
        x_com = plotz[0,i]
        y_com = plotz[1,i]
        
        x_data.append(x_com)
        y_data.append(y_com)
        x_goal.append(plotzd[0,i])
        y_goal.append(plotzd[1,i])
               
        list_of_lines[0].set_data([x_quad1, x_quad2], [y_quad1, y_quad2])
        list_of_lines[1].set_data([x_quad1,x_quad1], [y_quad1,y_quad1])
        list_of_lines[2].set_data([x_quad2, x_quad2], [y_quad2, y_quad2])
        list_of_lines[3].set_data([x_com,x_com], [y_com, y_com])
        list_of_lines[4].set_data(x_data,y_data)
        list_of_lines[5].set_data([plotzd[0,i],plotzd[0,i]],[plotzd[1,i],plotzd[1,i]])
        list_of_lines[6].set_data(x_goal,y_goal)
        
        return list_of_lines
    
    
    def init():
        return animate(0)

    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(plotz[0,:])),
        interval=use_dt, blit=True, init_func=init)
    plt.close(fig)
    plt.close(ani._fig)
    IPython.display.display_html(IPython.core.display.HTML(ani.to_html5_video()))


def QPtrackResults(z,u,z_des,animate):
    
    t = np.linspace(0,quadrotor.dt*(horizon_length),horizon_length+1)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t,z[0,:],label = 'Simulated Horizontal Pos')
    plt.plot(t[:-1],z_des[0,:],label = 'Planned Horizontal Pos')
    plt.ylabel('x')
    plt.title('Position')
    plt.subplot(3,1,2)
    plt.plot(t,z[1,:],label = 'Simulated Vertical Pos')
    plt.plot(t[:-1],z_des[1,:],label ='Planned Vertical Pos')
    plt.ylabel('y')
    plt.subplot(3,1,3)
    plt.plot(t,z[2,:])
    plt.ylabel(r'$\theta$')
    plt.xlabel('Time')
   

    plt.figure() 
    plt.subplot(3,1,1)
    plt.plot(t,z[3,:],label='Simulated Horizontal Vel')
    plt.plot(t[:-1],z_des[3,:],label = 'Planned Horizontal Vel')
    plt.ylabel('$v_x$')
    plt.title('Velocity')
    plt.subplot(3,1,2)
    plt.plot(t,z[4,:],label='Simulated Vertical Vel')
    plt.plot(t[:-1],z_des[4,:],label = 'Planned Vertical Vel')
    plt.ylabel('$v_y$')
    plt.subplot(3,1,3)
    plt.plot(t,z[5,:])
    plt.ylabel(r'$\omega$')
    plt.xlabel('Time')

    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t[:-1],u[0,:])
    plt.ylabel('$u_1$')
    plt.title('Control thrusts')
    plt.subplot(2,1,2)
    plt.plot(t[:-1],u[1,:])
    plt.ylabel('$u_2$')
    plt.xlabel('Time')
   
    if animate:
        animate_2Dquadrotor_2(z,z_des,quadrotor.dt)


# In[8]:


## First create a sine wave trjectory

horizon_length = 1500
det_t = 0.01

# 16s simulation in total: 12s tracking plus 3s keeping stable
simu_time = 12
terminal = int(simu_time/det_t)

z_des_1 = np.empty([6, horizon_length])
for i in range(horizon_length):
    if i <= terminal:
        z_des_1[:,i] = np.array([1.5 + 1.85*det_t*i, 1.5+ 5*np.sin(0.33*np.pi*det_t*i), 0., 1.85, 0.33*np.pi*5*np.cos(0.33*np.pi*det_t*i), 0.])
    else:
        z_des_1[:,i] = z_des_1[:,terminal]
        z_des_1[:,i][2] = 0.
        z_des_1[:,i][3] = z_des_1[:,terminal][3]*np.exp(-3*(10**-3)*i)
        z_des_1[:,i][4] = z_des_1[:,terminal][4]*np.exp(-5*(10**-3)*i)
        z_des_1[:,i][5] = 0.
        
#initial state
z0_2 = np.array([1, 2, 0.3, 0.,0.,0])

Q2=100*np.eye(6)
R2=0.1*np.eye(2)

u_len = 2

#constraints
u_max = np.array([10.**6,10.**6])
z_max = np.array([50.,10,10.**6,10.**6,10.**6,10.**6])

# z_2,u_2 = solve_ricatti_equations(quadrotor.A, quadrotor.B, Q, R, horizon_length)
z_21,u_21 = QPsolver(quadrotor.A,quadrotor.B,Q2,R2,z0_2, horizon_length, u_len, u_max, z_max, z_des_1)

QPtrackResults(z_21,u_21,z_des_1,animate=True)


# In[9]:


# Then if we add restrictions on velocities and control thrusts
# constraints
horizon_length = 1500
u_max_2 = np.array([200.,200.])
z_max_2 = np.array([30.,10.,10.**6,5.,5.,10.**6])
z_22,u_22 = QPsolver(quadrotor.A,quadrotor.B,Q2,R2,z0_2, horizon_length, u_len, u_max_2, z_max_2, z_des_1)

QPtrackResults(z_22,u_22,z_des_1,animate = True)


# In[ ]:


# Create the 2nd trjectory which is more unregular
# 15s simulation in total: 12s tracking plus 3s slowing down(keeping stable)

simu_time = 12
terminal = int(simu_time/det_t)
z_des_2 = np.empty([6, horizon_length])

TR = lambda t: 0.5*np.sin(0.8*np.pi*t) + 1.2*np.cos(0.05*t) + 2.8*np.sin(0.3*t)        # lambda function defining R(t)
dTR = lambda t: 0.4*np.pi*np.cos(0.8*np.pi*t) - 0.06*np.sin(0.05*t) + 0.84*np.cos(0.3*t)        # lambda function defining dTR/dt

for i in range(horizon_length):
    if i <= terminal:
        z_des_2[:,i] = np.array([1.5 + 0.18*(det_t*i)**2, 1.5 + TR(det_t*i), 0., 0.36*(det_t*i), dTR(det_t*i), 0.])
    else:
        z_des_2[:,i] = z_des_2[:,terminal]
        z_des_2[:,i][2] = 0.
        z_des_2[:,i][3] = z_des_2[:,terminal][3]*np.exp(-1.2*(10**-3)*i)
        z_des_2[:,i][4] = z_des_2[:,terminal][4]*np.exp(-1.5*(10**-3)*i)
        z_des_2[:,i][5] = 0.
        

#initial state
z0_3 = np.array([-1, 3, -0.5, 0.,0.1,0.2])

Q3=1000*np.eye(6)
R3=0.01*np.eye(2)

u_len = 2

# constraints
u_max_3 = np.array([100.,100.])
z_max_3 = np.array([50.,10,0.3,5.,5.,10.**6])

z_23,u_23 = QPsolver(quadrotor.A,quadrotor.B, Q3, R3, z0_3, horizon_length, u_len, u_max_3, z_max_3, z_des_2)

QPtrackResults(z_23,u_23,z_des_2,animate=True)


# In[11]:


# Create the 3rd trjectory(ellipse)
z_des_3 = np.empty([6, horizon_length])

# 20s simulation in total: 15s tracking plus 3s slowing down(keeping stable)
horizon_length = 1800
simu_time = 15
terminal = int(simu_time/det_t)


for i in range(horizon_length):
    if i <= terminal:
        z_des_3[:,i] = np.array([12.+ 10.*np.cos(det_t*i + np.pi), 1.+ 5 *np.sin(det_t*i + np.pi), 0., -10.*np.sin(det_t*i + np.pi), 5.*np.cos(det_t*i + np.pi), 0.])
    else:
        z_des_3[:,i] = z_des_3[:,terminal]
        z_des_3[:,i][2] = 0.
        z_des_3[:,i][3] = z_des_3[:,terminal][3]*np.exp(-1.2*(10**-3)*i)
        z_des_3[:,i][4] = z_des_3[:,terminal][4]*np.exp(-1.5*(10**-3)*i)
        z_des_3[:,i][5] = 0.

#initial state
z0_4 = np.array([-1, 3, -0.5, 0.,0.1,0.2])

Q4=1000*np.eye(6)
R4=0.001*np.eye(2)

u_len = 2

# constraints
u_max_4 = np.array([1000.,1000.])
z_max_4 = np.array([50.,10,10.**6,10.,10.,10.**6])

z_24,u_24 = QPsolver(quadrotor.A,quadrotor.B, Q4, R4, z0_4, horizon_length, u_len, u_max_4, z_max_4, z_des_3)

QPtrackResults(z_24,u_24,z_des_3,animate=True)


# In[ ]:


# Create the 4th trjectory(Lemniscate/Bernoulli curve: A "8" curve)
z_des_4 = np.empty([6, horizon_length])

# 15s simulation in total: 15s tracking plus 3s slowing down(keeping stable)
horizon_length = 1800
simu_time = 15
terminal = int(simu_time/det_t)

# Define Lemniscate of Bernoulli
Lx = lambda t: 8.*np.sqrt(2)*np.cos(t)/(np.sin(t)**2 + 1)   # lambda function defining Lx(t)
Ly = lambda t: 8.*np.sqrt(2)*np.cos(t)*np.sin(t)/(np.sin(t)**2 + 1)   # lambda function defining Ly(t)
dLx = lambda t: -8.*np.sqrt(2)*(np.sin(t)**3 + np.sin(t) + np.sin(2*t))/(np.sin(t)**2 + 1)**2 
dLy = lambda t: 8.*np.sqrt(2)*(np.cos(2*t)*(np.sin(t)**2 + 1) - np.sin(t)*np.sin(2*t))/(np.sin(t)**2 + 1)**2 

for i in range(horizon_length):
    if i <= terminal:
        z_des_4[:,i] = np.array([12.+ Lx(det_t*i+0.75*np.pi), 1.+ Ly(det_t*i+0.75*np.pi),0., dLx(det_t*i+0.75*np.pi), dLy(det_t*i+0.75*np.pi), 0.])
    else:
        z_des_4[:,i] = z_des_3[:,terminal]
        z_des_4[:,i][2] = 0.
        z_des_4[:,i][3] = z_des_3[:,terminal][3]*np.exp(-1.2*(10**-3)*i)
        z_des_4[:,i][4] = z_des_3[:,terminal][4]*np.exp(-1.5*(10**-3)*i)
        z_des_4[:,i][5] = 0.

#initial state
z0_5 = np.array([-1, 3, -0.5, 0.,0.1,0.2])

Q5=1000*np.eye(6)
R5=0.001*np.eye(2)

u_len = 2

# constraints
u_max_5 = np.array([10.**3,10.**3])
z_max_5 = np.array([100.,50.,10.**6,10.**6,10.**6,10.**6])

z_25,u_25 = QPsolver(quadrotor.A,quadrotor.B, Q5, R5, z0_5, horizon_length, u_len, u_max_5, z_max_5, z_des_4)

QPtrackResults(z_25,u_25,z_des_4,animate=True)


# In[ ]:


## Instead of using Quadratic Programs with constraints,
## next we try to use feedforward LQR controllers to track above trjectories and obtain global optimal controls

def solve_LQtracking(A,B,Q,R,z_des,horizon_length):
    """
    This function solves the backward Riccatti equations for regulator problems of the form
    min sum(xQx + uRu + qx) + xQx + qx subject to xn+1 = Axn + Bun
    
    Arguments:
    A, B, Q, R: numpy arrays defining the problem
    horizon_length: length of the horizon
    
    Returns:
    P: list of numpy arrays containing Pn from 0 to N
    K: list of numpy arrays containing Kn from 0 to N-1
    p: list of numpy arrays containing pn from 0 to N
    kf: list of numpy arrays containing kn from 0 to N-1
    """
    P = [] #will contain the list of Ps from N to 0
    K = [] #will contain the list of Ks from N-1 to 0
    p = [] #will contain the list of ps from N to 0
    kf = [] #will contain the list of ks from N-1 to 0
    
    P.append(Q) #PN
    p.append(-Q.dot(z_des[:,-1]))  #pN
    q = 0
    
    for i in range(horizon_length):
        q = - Q.dot(z_des[:,i])
        Knew = -1.0 * np.linalg.inv(B.transpose().dot(P[i]).dot(B) + R).dot(B.transpose()).dot(P[i]).dot(A)
        Pnew = Q + A.transpose().dot(P[i]).dot(A) + A.transpose().dot(P[i]).dot(B).dot(Knew)
        knew = -1.0 * np.linalg.inv(B.transpose().dot(P[i]).dot(B) + R).dot(B.transpose()).dot(p[i])
        pnew = q + A.transpose().dot(p[i]) + A.transpose().dot(P[i]).dot(B).dot(knew)
        K.append(Knew)
        P.append(Pnew)
        kf.append(knew)
        p.append(pnew)
    
    # since we went backward we return reverted lists
    return P[::-1],K[::-1],p[::-1],kf[::-1]


def LQTrackResults(z,u,z_des,K, uff, animate):
    
    t = np.linspace(0,quadrotor.dt*(horizon_length),horizon_length+1)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t,z[0,:],label = 'Simulated Horizontal Pos')
    plt.plot(t[:-1],z_des[0,:],label = 'Planned Horizontal Pos')
    plt.ylabel('x')
    plt.title('Position')
    plt.subplot(3,1,2)
    plt.plot(t,z[1,:],label = 'Simulated Vertical Pos')
    plt.plot(t[:-1],z_des[1,:],label ='Planned Vertical Pos')
    plt.ylabel('y')
    plt.subplot(3,1,3)
    plt.plot(t,z[2,:])
    plt.ylabel('$\theta$')
    plt.xlabel('Time')
   

    plt.figure() 
    plt.subplot(3,1,1)
    plt.plot(t,z[3,:],label='Simulated Horizontal Vel')
    plt.plot(t[:-1],z_des[3,:],label = 'Planned Horizontal Vel')
    plt.ylabel('$v_x$')
    plt.title('Velocity')
    plt.subplot(3,1,2)
    plt.plot(t,z[4,:],label='Simulated Vertical Vel')
    plt.plot(t[:-1],z_des[4,:],label = 'Planned Vertical Vel')
    plt.ylabel('$v_y$')
    plt.subplot(3,1,3)
    plt.plot(t,z[5,:])
    plt.ylabel(r'$\omega$')
    plt.xlabel('Time')

    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t[:-1],u[0,:])
    plt.ylabel('$u_1$')
    plt.title('Control thrusts')
    plt.subplot(2,1,2)
    plt.plot(t[:-1],u[1,:])
    plt.ylabel('$u_2$')
    plt.xlabel('Time')
    
    K = np.array(K)         
    plt.figure(figsize=[8.5,4.5])
    # first row of K
    plt.subplot(1,2,1)
    plt.plot(K[:,0,0],'-o', markersize=6)
    plt.plot(K[:,0,1],'-o', markersize=6)
    plt.plot(K[:,0,2],'-o', markersize=6)
    plt.plot(K[:,0,3],'-o', markersize=6)
    plt.plot(K[:,0,4],'-o', markersize=6)
    plt.plot(K[:,0,5],'-o', markersize=6)
    plt.legend(["K00","K01","K02","K03","K04","K05"],fontsize = 6,frameon = False,loc ='best')
    # plt.legend(["K11"],fontsize = 6,frameon = False,loc ='best')
    plt.ylabel('K1')
    plt.xlabel('Time')
    plt.title('Feedback gains ')

    # second row of K
    plt.subplot(1,2,2)
    plt.plot(K[:,1,0],'-o', markersize=6)
    plt.plot(K[:,1,1],'-o', markersize=6)
    plt.plot(K[:,1,2],'-o', markersize=6)
    plt.plot(K[:,1,3],'-o', markersize=6)
    plt.plot(K[:,1,4],'-o', markersize=6)
    plt.plot(K[:,1,5],'-o', markersize=6)
    plt.legend(["K10","K11","K12","K13","K14","K15"],fontsize = 6,frameon = False,loc ='best')
    # plt.legend(["K11"],fontsize = 6,frameon = False,loc ='best')
    plt.ylabel('K2')
    plt.xlabel('Time')
    plt.title('Feedback gains ')

    # Here kf is a 2 by 1 vector with two elements
    # plt.figure(figsize=[8.5,4.5])
    plt.figure()
    plt.plot(uff,'-o', markersize=6)
    # plt.plot(kf31[:,1],'-o', markersize=6)
    plt.legend(["uf1","uf2"],fontsize = 6,frameon = False,loc ='best')
    # plt.legend(["K11"],fontsize = 6,frameon = False,loc ='best')
    plt.ylabel('uff')
    plt.xlabel('Time')
    plt.title('Feedforwards')

    if animate:
        animate_2Dquadrotor_2(z,z_des,quadrotor.dt)


# In[ ]:


## Use LQR to track the 1st sine wave path
horizon_length = 1500

Q31=1000*np.eye(6)
R31=0.01*np.eye(2)

P31,K31,p31,kf31 = solve_LQtracking(quadrotor.A, quadrotor.B, Q31, R31, z_des_1 ,horizon_length)

z_31,u_31 = quadrotor.LQ_simulate(z0_3, K31, kf31, horizon_length)

LQTrackResults(z_31, u_31, z_des_1,K31,kf31,animate=True)


# In[ ]:


## Straight line path
horizon_length = 1500
z_line = np.empty([6, horizon_length])

# 15s simulation in total: 12s tracking plus 3s keeping stable
simu_time = 12
terminal = int(simu_time/det_t)

for i in range(horizon_length):
    if i <= terminal:
        z_line[:,i] = np.array([0. +(det_t*i), 0. + 0.667*(det_t*i),0., 1., 0.667, 0.])
    else:
        z_line[:,i] = z_line[:,terminal]
        z_line[:,i][2] = 0.
        z_line[:,i][3] = z_line[:,terminal][3]*np.exp(-1.2*(10**-3)*i)
        z_line[:,i][4] = z_line[:,terminal][4]*np.exp(-1.5*(10**-3)*i)
        z_line[:,i][5] = 0.

z_l0 = z_line[:,0]
Q31=1000*np.eye(6)
R31=0.01*np.eye(2)


P31,K31,p31,kf31 = solve_LQtracking(quadrotor.A, quadrotor.B, Q31, R31, z_line, horizon_length)

z_31,u_31 = quadrotor.LQ_simulate(z_l0, K31, kf31, horizon_length)

LQTrackResults(z_31, u_31, z_line, K31,kf31,animate=True)        


# In[ ]:




