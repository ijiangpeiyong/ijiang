import numpy as np
from scipy.fftpack import dstn, idstn, irfft, rfft, dst, idst, dct, idct, fft, ifft
import time
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#  10000 对应　１０ｓ


A=np.random.random((64,64,64))
A[0,:,:]=0
A[-1,:,:]=0
A[:,0,:]=0
A[:,-1,:]=0

tic=time.time()
for i in range(10000):
    A[1:-1,1:-1,1:-1]=(A[0:-2,1:-1,1:-1]+A[2::,1:-1,1:-1]+A[1:-1,0:-2,1:-1]+A[1:-1,2::,1:-1]+A[1:-1,1:-1,0:-2]+A[1:-1,1:-1,2::])/6.
    A[:,:,0]=A[:,:,-2]
    A[:,:,-1]=A[:,:,1]

toc=time.time()

print(toc-tic)

fig=plt.figure('x-y')
ax=Axes3D(fig)
for i in range(64):
    ax.clear()
    ax.set_title(i)

    X = np.linspace(0,1,64)
    Y = np.linspace(0,1,64)
    Y,X = np.meshgrid(Y,X)
    Z=A[:,:,i]
    
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)

    plt.pause(0.01)



fig=plt.figure('x-y')
ax=Axes3D(fig)
for i in range(64):
    ax.clear()
    ax.set_title(i)

    X = np.linspace(0,1,64)
    Z = np.linspace(0,1,64)
    Z,X = np.meshgrid(Z,X)
    Y=A[:,i,:]
    
    surf = ax.plot_surface(X, Z,Y, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)

    plt.pause(0.01)
