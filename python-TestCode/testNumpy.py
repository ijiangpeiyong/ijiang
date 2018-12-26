import numpy as np

numPart=10000
dataRandom=np.random.random((numPart,4))*2.-1.
r=dataRandom[:,0]**2+dataRandom[:,1]**2+dataRandom[:,2]**2+dataRandom[:,3]**2
indexR=r<1.

A=(dataRandom[indexR,:])[0:250,:]

A1=A[:,0]

print(np.shape(A1))

'''
print(A.shape)
print(numPart/3.5)
print(16.*2/np.pi**2*1.1)


x1,x2,x3,x4=np.random.random((2,4))
print(x1,x2,x3,x4)
'''

from scipy import constants as const

print(const.c)

print(dataRandom[:,0].std())

print(dataRandom[:,0].var())

print(eval('dataRandom[:,0].var()'))



print(11.1//3.05)

print(np.full((2),np.nan))

A=11.1
B=3.5
print(A / B, A//B, A%B, A-(A//B)*B)
