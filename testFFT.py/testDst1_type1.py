from scipy.fftpack import dst, dct,idst,idct,rfft,irfft
import numpy as np

n=6
data=np.random.random((n))

print(data)
#print(dst(data,type=1))
print(idst(dst(data,type=1),type=1)/(2.*(n+1.)))
print(irfft(rfft(data)))



