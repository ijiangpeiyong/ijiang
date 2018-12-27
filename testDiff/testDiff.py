import numpy as np 

a=np.round(np.random.random((4,6))*100)

b=np.arange(24).reshape((4,6))

print(b)

print('-')

d=np.gradient(b)

print(d[0])

print(d[1])


