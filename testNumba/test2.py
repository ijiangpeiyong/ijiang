import numpy as np
from scipy.fftpack import dstn, idstn, irfft, rfft, dst, idst, dct, idct, fft, ifft
import time
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numba as nb

def Test(A,B,num):
    for i in range(num):
        C=A-B
    return C

@nb.jit()
def nbTest(A,B,num):
    for i in nb.prange(num):
        C=A-B
    return C

@nb.jit()
def nbTest2(C,num):
    
    for i in nb.prange(num):
        C[0:64,0:64,0:64]-=C[64::,64::,64::]
    return C


@nb.jit(nopython=True,parallel=True)
def nbTest3(A):
    for i in nb.prange(10000):
        A[1:-1,1:-1,1:-1]=(A[0:-2,1:-1,1:-1]+A[2::,1:-1,1:-1]+A[1:-1,0:-2,1:-1]+A[1:-1,2::,1:-1]+A[1:-1,1:-1,0:-2]+A[1:-1,1:-1,2::])/6.
        A[:,:,0]=A[:,:,-2]
        A[:,:,-1]=A[:,:,1]
    return A


if __name__=='__main__':
    myShape=(64,64,64)
    A=np.random.random(myShape)
    B=np.random.random(myShape)
    C=np.hstack((A,B))
    
    num=int(1e4)

    t1=time.time()
    #C=Test(A,B,num)
    # t2=time.time()
    # C=nbTest2(C,num)
    # t3=time.time()
    # C=nbTest2(C,num)
    # t4=time.time()

    t2=time.time()
    nbTest3(A)
    t3=time.time()
    nbTest3(A)
    t4=time.time()


    print(t4-t3,t2-t1)



