import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

data=np.loadtxt('/home/pyong/ijiang/cpp/pic3d').reshape(64,64,64)

print(np.shape(data))

'''
fig=plt.figure('x-y')
ax=Axes3D(fig)
for i in range(64):
    ax.clear()
    ax.set_title(i)

    X = np.linspace(0,1,64)
    Y = np.linspace(0,1,64)
    Y,X = np.meshgrid(Y,X)
    Z=data[:,:,i]
    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)

    plt.pause(0.01)

'''
fig=plt.figure('x-y')
ax=Axes3D(fig)
for i in range(64):
    ax.clear()
    ax.set_title(i)

    X = np.linspace(0,1,64)
    Z = np.linspace(0,1,64)
    Z,X = np.meshgrid(Z,X)
    Y=data[:,i,:]
    
    surf = ax.plot_surface(X, Z,Y, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)

    plt.pause(0.01)



