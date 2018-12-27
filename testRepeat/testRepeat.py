
import numpy as np 

a=np.random.random((4))

'''
b=np.tile(a,(3,2,1))
print(a)
print('-'*10)
print(b)

print(np.shape(b))

print('*'*50)
'''

a=np.array([1,2,3,4])

c=np.repeat(a[:,np.newaxis],2,axis=1)[:,:,np.newaxis]

d=np.repeat(c,3,axis=2)

print(c)
print(np.shape(c),np.shape(d))

for i in range(2):
    for j in range(3):
        print(d[:,i,j])

