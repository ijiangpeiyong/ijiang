import numpy as np
import multiprocessing as mp 
import time
from multiprocessing import Process, Array,Value,Lock

import matplotlib.pyplot as plt

import time

class Beam:
    def __init__(self):
        
        self.numPart=np.int(1e7)
        self.x=Array('d',self.numPart)
        self.xp=Array('d',self.numPart)

        self.y=np.empty((self.numPart))
        self.yp=np.empty((self.numPart))

    def SetCPU(self,numCPU):
        self.numCPU=np.int(numCPU)


    def Allocation(self):
        allocPart=[self.numPart//self.numCPU]*self.numCPU
        allocPartCum=np.cumsum(allocPart)

        self.allocPart=Array('i',allocPart)
        self.allocPartCum=Array('i',allocPartCum)
        #print(self.allocPartCum[::])

            


    def Worker(self,iCPU,lock):
        tic_1=time.time()
        num=self.allocPart[iCPU]
        numCum=self.allocPartCum[iCPU]
        numPart=int(num*4./3.14*1.15)
        #print(numPart)
        tic_2=time.time()
        x=np.random.random((numPart))*2.-1
        xp=np.random.random((numPart))*2.-1
        tic_3=time.time()
        r=x**2+xp**2
        indexR=r<1.
        tic_4=time.time()
        
        lock.acquire()
        self.x[numCum-num:numCum]=x[indexR][0:num]
        self.xp[numCum-num:numCum]=xp[indexR][0:num]
        tic_5=time.time()
        lock.release()
        #print('local time:  ',tic_2-tic_1,tic_3-tic_2,tic_4-tic_3,tic_5-tic_4)


    def WorkerS(self):
        numPart=int(self.numPart*4./3.14*1.15)
        x=np.random.random((numPart))*2.-1
        xp=np.random.random((numPart))*2.-1
        r=x**2+xp**2
        indexR=r<1.
        self.y=x[indexR][0:self.allocPart[iCPU]]
        self.yp=xp[indexR][0:self.allocPart[iCPU]]        


if __name__ == "__main__":
    lock=Lock()
    beam=Beam()
    for i in range(1,5):
        tic_1=time.time()
        beam.SetCPU(i)
        beam.Allocation()

        
        for iCPU in range(beam.numCPU):
            p = Process(target=beam.Worker,args=(iCPU,lock,))
            p.start()
            p.join()
        


        tic_2=time.time()
        beam.WorkerS()
        tic_3=time.time()


        
        print((tic_2-tic_1)/(tic_3-tic_2),tic_2-tic_1,tic_3-tic_2)

    '''
    plt.figure(1)
    plt.plot(beam.x,beam.xp,'.')


    plt.figure(2)
    plt.plot(beam.y,beam.yp,'.')
    plt.show()
    '''
