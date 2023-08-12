# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:25:59 2023

@author: thoma
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import matplotlib.animation as animation
from sklearn import preprocessing


def init():
    ax.clear()


def animate(i):
    ax.clear()
    ax.plot(X[:i,0], X[:i,1], X[:i,2], '--r', linewidth = 1.0)
    ax.plot(X[:i,6], X[:i,7], X[:i,8], '--b', linewidth = 1.0)

    ti, xi = T[i], X[i,:]
    vec_RT = np.array(xi[0:3])
    vec_VT = np.array(xi[3:6]) 
    vec_RP = np.array(xi[6:9])
    vec_VP = np.array(xi[9:]) 
    
    arrow_len = 2500
    head_size = 1000
    norm = np.linalg.norm(vec_VP)
    dx = vec_VP[0]/norm*arrow_len
    dy = vec_VP[1]/norm*arrow_len
    dz = vec_VP[2]/norm*arrow_len
    #print(dx,dy,dz)
    
    VP = np.linalg.norm(vec_VP)
    
    #ax.plot([xi[0], xi[6]],[xi[1],xi[7]],[xi[2],xi[8]], '--',color ='gray')
    ax.plot(xi[0], xi[1], xi[2], 'or',markersize=2., label = 'Target')
    ax.plot(xi[6], xi[7], xi[8], 'ob',markersize=2, label = 'Interceptor')
    
    #ax.quiver(vec_RP[0], vec_RP[1], vec_RP[2], vec_VP[0], vec_VP[1], vec_VP[2], length=1, arrow_length_ratio=1e-2, color = 'black')
    #ax.quiver(vec_RP[0], vec_RP[1], vec_RP[2], dx, dy, dz, length=1, arrow_length_ratio=1e-2, color = 'black')
    
    ax.set_xlim(-80,80)
    ax.set_ylim(0,50000)
    ax.set_zlim(10000,10200)
    ax.text(0, 0, 9800, f't={round(ti,2)} s')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    #ax.set_xlabel('$X$')
    #ax.set_ylabel('$Y$')
    #ax.set_zlabel(r'$Altitude$')
    ax.view_init(elev=20, azim=-30)  # Elevação: 30 graus, Azimute: 45 graus
    legend = ax.legend(loc = 'center left')
    legend.get_frame().set_edgecolor('none')
    ax.set_box_aspect([1, 5, 1])
    plt.tight_layout()
    
    #plt.pause(1e-2)


def ZeroEffortMiss(t,x):
    '''

    Parameters
    ----------
    t : scalar
        time instant.
    x : state vector wrinten in Inertial Coordinate System
        beta: Target flight path angle
        RTx, RTz: Target position
        RPx, RPz: Persuer position
        VTx, VTz: Target velocity
        VPx, VPz: Persuer velocity
    Returns
    -------
    dxdt

    '''
    # Input state
    RT1, RT2, RT3, VT1, VT2, VT3, RP1, RP2, RP3, VP1, VP2, VP3 = x
    tgo = tf-t
    # Velocity vectors
    vec_VT = np.array([VT1, VT2, VT3])
    VT = np.linalg.norm(vec_VT)    
    vec_VP = np.array([VP1, VP2, VP3])
    VP = np.linalg.norm(vec_VP)
    # Position vectors
    vec_RT = np.array([RT1, RT2, RT3])
    RT = np.linalg.norm(vec_RT)
    vec_RP = np.array([RP1, RP2, RP3])
    RP = np.linalg.norm(vec_RP)
    # Relative position
    vec_RTP = np.subtract(vec_RT,vec_RP)
    RTP = np.linalg.norm(vec_RTP)
    unit_vec_RTP = vec_RTP/RTP
    # Relative velocity
    vec_VTP = np.subtract(vec_VT,vec_VP)
    VTP = np.linalg.norm(vec_VTP)
    VTPx = vec_VTP[0]
    VTPz = vec_VTP[1]
    # Zero Effort Miss
    vec_ZEM = vec_RTP + vec_VTP*tgo
    vec_ZEM_r = np.dot(vec_ZEM,unit_vec_RTP)*unit_vec_RTP
    vec_ZEM_n = np.subtract(vec_ZEM,vec_ZEM_r)
    # ProNav
    aP = Np * vec_ZEM_n / tgo**2
    # Kinematic equations
    aTi = fun_aT(t)
    vTi = fun_VT(t)
    # x = RT1, RT2, RT3, VT1, VT2, VT3, RP1, RP2, RP3, VP1, VP2, VP3
    dx = np.zeros(x.size)
    dx[0]  =  vTi[0]
    dx[1]  =  vTi[1]
    dx[2]  =  vTi[2]
    dx[3]  =  aTi[0]
    dx[4]  =  aTi[1]
    dx[5]  =  aTi[2]
    dx[6]  =  VP1
    dx[7]  =  VP2
    dx[8]  =  VP3
    dx[9]  =  aP[0]
    dx[10] =  aP[1]
    dx[11] =  aP[2]
    
    return dx.reshape(x0.shape)


#%% TARGET KINEMATICS
aT = 6 * 9.81
omeg = 1 # rad/s
fun_VT = lambda t : np.array([-aT*np.cos(omeg*t), 1000, aT*np.sin(omeg*t)])
fun_aT = lambda t : np.array([aT*np.sin(omeg*t), 0, aT*np.cos(omeg*t)])
#%% INITIAL CONDITIONS
RP0 = np.array([0, 0, 10000]) # Peruer initial position (inertial reference frame) [ft]
VP0 = np.array([0, 2400, 0])  # Peruer initial velocity (persuer reference frame) [ft/s]

RT0 = np.array([0, 30000, 10000])  # Target initial position (inertial reference frame) 
VT0 = np.array([-aT/omeg, 1000, 0])                 # Target initial velocity (target reference frame)

HE = 0*np.pi/180
Np = 3

x0 = np.hstack((RT0,VT0,RP0,VP0))
#%% SIMULATION PARAMETERS
t0 = 0

VC = VP0[1]-VT0[1]
RTM2 = RT0[1]-RP0[1]
tf = RTM2/VC
h = 1e-1

N = int((tf-t0)/h)

T = np.zeros((N+1))
X = np.zeros((N+1,x0.size))

X[0,:] = x0

for i in range(0,N):
    X[i+1,:] = X[i,:] + h*ZeroEffortMiss(T[i],X[i,:])
    T[i+1] = T[i] + h
    

#normalized_matrix = preprocessing.normalize(matrix, axis=0)

sol = list(zip(T, X))
n_plots = 30

font = {'family' : 'Times New Roman',
        'size'   : 20}

fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111, projection='3d')

#step = len(T) // n_plots
#for i in range(1,n_plots+1):
    # 215
ani = animation.FuncAnimation(fig, animate, frames=60, init_func=init, interval=1e-3)
#ani.show()

ani.save('interception_animation.gif', writer='imagemagick',dpi =150)
