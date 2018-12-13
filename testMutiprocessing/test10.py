import numpy as np
import multiprocessing as mp

a=0


def f(a):
    a+=1
    print(a)


import multiprocessing as mp
n_thread = mp.cpu_count()
with mp.Pool(n_thread) as p:
    p.map(f, range(n_thread))


print('END')