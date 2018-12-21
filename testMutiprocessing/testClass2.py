#Author:Peiyong Jiang
import multiprocessing as mp 
import numpy as np

class Test():
    def __init__(self):
        self.all=np.zeros((1))
    
    def aAdd(self,a):
        self.all=np.append(self.all,a)
        #print(self.all)
        return self.all

if __name__=="__main__":
    t=Test()


    #-------------------
    numCPU=6
    with mp.Pool(processes=numCPU) as p:
        resPool=p.map(t.aAdd,range(numCPU))

    print(resPool)

    res=np.hstack(resPool)

    print(res)

