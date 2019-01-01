import numpy as np 
import numba as nb 
import time
from scipy.fftpack import dstn, idstn,fft,ifft

num=128


data=np.random.random((num,num,num))

def npFFT(data):
    
    data=dstn(data,axes=[0,1],type=1,overwrite_x=True)
    data=fft(data,axis=-1,overwrite_x=True)

    data=ifft(data,axis=-1,overwrite_x=True)
    data=idstn(data,axes=[0,1],type=1,overwrite_x=True)
    data/=4*(num+1)**2

#@nb.vectorize(["void(float32)"], target='cuda')
#@nb.jit(["void(float32)"], target='parallel')



@nb.jit()
def nbFFT(data):
    
    data=dstn(data,axes=[0,1],type=1,overwrite_x=True)
    data=fft(data,axis=-1,overwrite_x=True)

    data=ifft(data,axis=-1,overwrite_x=True)
    data=idstn(data,axes=[0,1],type=1,overwrite_x=True)
    data/=4*(num+1)**2
    


if __name__=='__main__':
    data=np.random.random((num,num,num))
    tic1=time.time()
    npFFT(data)
    tic2=time.time()
    nbFFT(data)
    tic3=time.time()
    nbFFT(data)
    tic4=time.time()



    print(tic2-tic1,tic3-tic2,tic4-tic3)



