
import numpy as np
import multiprocessing as mp


def f(x):
    n=1000
    x=np.random.random((n,n))
    while 1:
       x**=2
       print(x)


import multiprocessing as mp
n_thread = mp.cpu_count()
with mp.Pool(n_thread) as p:
    p.map(f, range(n_thread))


print('END')
