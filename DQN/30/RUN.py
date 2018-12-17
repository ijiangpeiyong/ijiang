# Author: Peiyong Jiang

from ENV import ENV
from BRAIN import BRAIN

import numpy as np

from multiprocessing import Process

numEpisode=1000

env=ENV()
brain=BRAIN()

counterStep=0
iEpisode=0
numPreStoreMemory=10

env.Reset()

iTest=0
while True:
    stateNow,counterRunNow=env.GetState()
    actionNow=brain.SelAction(stateNow,counterRunNow)

    stateNext,rewardNow, doneNow,counterRunNow=env.UpdateState(stateNow,actionNow)
   
    #print(stateNow,actionNow,stateNext)
   
    counterRunNext=counterRunNow+1

    brain.StoreMemory(stateNow,counterRunNow,actionNow,rewardNow,stateNext,counterRunNext)

    if counterStep > numPreStoreMemory and (counterStep % 5==0):
        p = Process(target=brain.Learn)
        p.start()
        p.join()
        #brain.Learn()

    counterStep+=1

    if doneNow:
        env.Reset()
        stateNext,counterRunNext=env.GetState()

        iEpisode+=1
        #print(iEpisode,brain.factorGreedyEpsilon)
        if iEpisode>numEpisode:
            break

    env.SetState(stateNext)

    brain.PlotLoss()














print('END@DQN')
