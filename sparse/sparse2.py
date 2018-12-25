from scipy import sparse
row = [2,2,3,2]
col = [3,4,2,3]
page=[3,3,3,3]
data=[1,2,3,6]
c = sparse.coo_matrix((data,(row,col,page)),shape=(5,6,5))

print(c.toarray())
print(c)

d=c.toarray()
print(d)