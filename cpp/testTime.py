import numpy as np 
import time 

t1=time.time()
x,y,z,c,b,n=np.random.randn(np.int(1e7),6).T
t2=time.time()
print(t2-t1)