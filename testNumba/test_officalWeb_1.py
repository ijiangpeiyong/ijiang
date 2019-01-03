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



@jit(nopython=True)
def nb_monte_carlo_pi(nsamples):
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

    nb_monte_carlo_pi(nsamples)
    ticNB=time.time()
    nb_monte_carlo_pi(nsamples)
    tocNB=time.time()
    print('nb  ',tocNB-ticNB)
    print('-/nb=',(toc-tic)/(tocNB-ticNB))    
