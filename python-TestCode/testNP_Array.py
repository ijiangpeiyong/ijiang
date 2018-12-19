import numpy as np 

A=np.random.random((3,4))
B=np.random.random((1,4))

print(A)
print(B)

A=np.vstack((A,B))
print(A)

print('-'*10)

for i in A:
    print(i)