from scipy.fftpack import dst, dct,idst,idct
import numpy as np

data=np.random.random((6))

print(data)
print(dst(data,norm='ortho'))
print(idst(dst(data,norm='ortho'),norm='ortho'))







