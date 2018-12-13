
import numpy as np
import multiprocessing as mp


def f(x):
    n=1
    x=np.random.random(1)
    while 1:
       x**=2
       print(n)
       n+=1
       if n>100000:
           break



import multiprocessing as mp
n_thread = mp.cpu_count()
with mp.Pool(n_thread) as p:
    p.map(f, range(n_thread))