from scipy.fftpack import dstn, dctn,idstn,idctn
import numpy as np

dataC=np.random.random((2,2))
dataO=np.zeros((2,2))
data1=np.hstack((dataO,dataO,dataO))
data2=np.hstack((dataO,dataC,dataO))

data=np.vstack((data1,data2,data1))

dataS2=idstn(dstn(data,norm='ortho',axes=[0,1]),norm='ortho',axes=[0,1])
dataS2[dataS2<1e-6]=0


dataS3=idstn(dstn(data,norm='ortho'),norm='ortho')
dataS3[dataS2<1e-6]=0

print(data)
#print(dstn(data,norm='ortho'))
print('-'*50)
print(dataS2)
print('-'*50)
print(dataS3)

kx=









