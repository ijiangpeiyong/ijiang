import numpy as np 
import dask.array as da
import time

numMat=int(1e5)

tic_1=time.time()
a= np.random.normal(size= (1000,10000))
b= np.random.normal(size= (10000,1000))
result_1=a.dot(b)

tic_2=time.time()

c=da.random.normal(size= (1000,10000),chunks=1000)
d=da.random.normal(size= (10000,1000),chunks=1000)
result_2=c.dot(d).compute()

tic_3=time.time()



print(tic_2-tic_1,tic_3-tic_2)
