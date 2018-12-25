from scipy import sparse
row = [2,2,3,2]
col = [3,4,2,3]
data=[1,2,3,6]
c = sparse.coo_matrix((data,(row,col)),shape=(5,6))

print(c.toarray())
print(c)

d=c.toarray()
print(d)