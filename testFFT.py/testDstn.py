from scipy.fftpack import dstn, dctn,idstn,idctn
import numpy as np

data=np.random.random((2,3,4))

print(data)
#print(dstn(data,norm='ortho'))
print('-'*50)
#print(idstn(dstn(data,norm='ortho',axes=[0,1]),norm='ortho',axes=[0,1]))

print(idstn(dstn(data,norm='ortho',axes=[0,1]),norm='ortho'))
