import time
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
from functools import partial

def calculate(vector, matrix = None):
    for i in range(50):
        v = matrix.dot(vector)
    return v

if __name__ == '__main__':
    N = np.int(1e6)
    matrix = sp.rand(N, N, density = 1e-5, format = 'csr')

    t = time.time()
    input = []
    for i in range(10):
        input.append(np.random.rand(N))
    mp = partial(calculate, matrix = matrix)
    p = Pool(2)
    res = p.map(mp, input)
    print(time.time() - t)