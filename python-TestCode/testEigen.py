import numpy as np 

beta=np.random.random()+1.
alpha=np.random.random()
gamma=(1.+alpha**2)/beta

mat=np.array([[beta,-alpha],[-alpha,gamma]])
print(mat)

matEig,matVec=np.linalg.eig(mat)

matEigDiag=np.diag(matEig)
print(matEigDiag)


print(matVec)
print(matVec.T)
print(np.linalg.inv(matVec))

print('-'*10)
print(mat)
print(np.dot(np.dot(matVec,matEigDiag),matVec.T))
print(np.dot(np.dot(matVec.T,matEigDiag),matVec))

x1,x2=np.random.random((6,2)).T 

print(x1)
print(x2)

x=[x1,x2]

y=np.dot(matEigDiag,x)

print(x)
print(y)

z=np.vstack((x1,x2))
print(z)
y2=np.dot(matEigDiag,z)
print(y2)

print('-'*10)
print(y)
print(y2)

y21,y22=np.dot(matEigDiag,[x1,x2])

print('-'*10)
print(y2)
print(y21)
print(y22)

print(np.cov([y21.T,y22.T]))

print(np.cov([y21,y22]))





