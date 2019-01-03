from numba import jit
import random
import time

def monte_carlo_pi(nsamples):
    acc=0
    for i in range(nsamples):
        x=random.random()
        y=random.random()
        if (x**2+y**2)<1.0:
            acc+=1
    return 4.0*acc/nsamples



@jit()
def jit_monte_carlo_pi(nsamples):
    acc=0
    for i in range(nsamples):
        x=random.random()
        y=random.random()
        if (x**2+y**2)<1.0:
            acc+=1
    return 4.0*acc/nsamples

@jit(nopython=True)
def nopython_monte_carlo_pi(nsamples):
    acc=0
    for i in range(nsamples):
        x=random.random()
        y=random.random()
        if (x**2+y**2)<1.0:
            acc+=1
    return 4.0*acc/nsamples


@jit(nogil=True)
def nogil_monte_carlo_pi(nsamples):
    acc=0
    for i in range(nsamples):
        x=random.random()
        y=random.random()
        if (x**2+y**2)<1.0:
            acc+=1
    return 4.0*acc/nsamples


@jit(nogil=True,nopython=True)
def nogil_nopython_monte_carlo_pi(nsamples):
    acc=0
    for i in range(nsamples):
        x=random.random()
        y=random.random()
        if (x**2+y**2)<1.0:
            acc+=1
    return 4.0*acc/nsamples


if __name__ == "__main__":
    nsamples=int(1e5)
    tic=time.time()
    monte_carlo_pi(nsamples)
    toc=time.time()
    print('nb  ',toc-tic)

    jit_monte_carlo_pi(nsamples)
    tic_jit_monte_carlo_pi=time.time()
    jit_monte_carlo_pi(nsamples)
    toc_jit_monte_carlo_pi=time.time()
    print('nb  ',toc_jit_monte_carlo_pi-tic_jit_monte_carlo_pi)
 

    nopython_monte_carlo_pi(nsamples)
    tic_nopython_monte_carlo_pi=time.time()
    nopython_monte_carlo_pi(nsamples)
    toc_nopython_monte_carlo_pi=time.time()
    print('nb  ',toc_nopython_monte_carlo_pi-tic_nopython_monte_carlo_pi)
 
    nogil_monte_carlo_pi(nsamples)
    ticNogil=time.time()
    nogil_monte_carlo_pi(nsamples)
    tocNogil=time.time()
    print('nb  ',tocNogil-ticNogil)
 
    nogil_nopython_monte_carlo_pi(nsamples)
    tic_nogil_nopython_monte_carlo_pi=time.time()
    nogil_nopython_monte_carlo_pi(nsamples)
    toc_nogil_nopython_monte_carlo_pi=time.time()
    print('nb  ',toc_nogil_nopython_monte_carlo_pi-tic_nogil_nopython_monte_carlo_pi)
  