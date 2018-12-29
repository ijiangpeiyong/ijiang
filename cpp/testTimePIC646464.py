import numpy as np
from scipy.fftpack import dstn, idstn, irfft, rfft, dst, idst, dct, idct, fft, ifft
import time

# 对应０．０2ｓ

A=np.random.random((64,64,64))

tic=time.time()
A=dstn(A,axes=[0,1],type=1,overwrite_x=True)
A=fft(A,axis=-1,overwrite_x=True)

#mpWeighGridOnX=mpWeighGridOnX/fftK2
B=np.random.random((64,64,64))
A/=B

A=np.real(ifft(A,axis=-1,overwrite_x=True))

A=idstn(A,axes=[0,1],type=1,overwrite_x=True)

toc=time.time()

print(toc-tic)
