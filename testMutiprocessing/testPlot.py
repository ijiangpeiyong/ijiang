import multiprocessing as mp

import matplotlib.pyplot as plt

import numpy as np

from queue import Queue
import time


def GenData(nData):
    data=np.arange(nData,500000000)
    #print(data)
    return data

def CalData(nData):
    data=GenData(nData)
    sumData=np.sum(np.sin(data))
    print(sumData)
    return sumData

def workCal(nData):
    time1=time.time()
    sumData=CalData(nData)
    time2=time.time()
    timed=time2-time1
    return sumData

    


if __name__=='__main__':
    numCount = 6
    with mp.Pool(processes=numCount) as p:
        res=p.map(workCal,range(numCount))
    
    #pool=mp.Pool(processes=numCount)
    #res=pool.map(workCal,range(numCount))

    print(res)



    print('END')
