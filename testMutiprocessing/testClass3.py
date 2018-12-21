#Author:Peiyong Jiang
import multiprocessing as mp 
import numpy as np

class Test():
    def __init__(self):
        self.all=np.arange(10)
        print(self.all)
    
    def aAdd(self,a):
        now=self.all[a]
        #print(now)
        return now

if __name__=="__main__":
    t=Test()


    #-------------------
    numCPU=6
    with mp.Pool(processes=numCPU) as p:
        resPool=p.map(t.aAdd,range(numCPU))

    print(resPool)

    res=np.hstack(resPool)

    print(res)

