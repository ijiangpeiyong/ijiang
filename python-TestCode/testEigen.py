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

'''

b1,b2=np.linalg.eig(a)

print(b1)
print(b2)

c=np.diag(b1)

print(b2.T.dot(c).dot(b2))
'''