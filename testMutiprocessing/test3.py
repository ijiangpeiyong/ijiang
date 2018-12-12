# 网上的一个例子，100%,超级牛逼

import numpy as np
import multiprocessing as mp


def f(x):
    while 1:
        if (x * x) ^ 1 + 1 >= 1:  # run all parts
            pass  # but it is just pointless

import multiprocessing as mp
n_thread = mp.cpu_count()
with mp.Pool(n_thread) as p:
    p.map(f, range(n_thread))