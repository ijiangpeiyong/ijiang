import time
import numpy as np
import scipy.sparse as sp

def calculate(vector, matrix = None):
    for i in range(50):
        v = matrix.dot(vector)
    return v

if __name__ == '__main__':
    N = np.int(1e6)
    matrix = sp.rand(N, N, density = 1e-5, format = 'csr')
    t = time.time()
    res = []
    for i in range(10):
        res.append(calculate(np.random.rand(N), matrix = matrix))    
    print(time.time() - t)