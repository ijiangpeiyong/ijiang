# Author: Peiyong Jiang

from ENV import ENV
from BRAIN import BRAIN

import numpy as np

numEpisode=1000

env=ENV()
brain=BRAIN()

counterStep=0
iEpisode=0
numPreStoreMemory=10

env.Reset()

iTest=0
while True:
    stateNow=env.GetState()
    actionNow=brain.SelAction(stateNow)

    stateNext,rewardNow, doneNow=env.UpdateState(stateNow,actionNow)
   
    #print(stateNow,actionNow,stateNext)
   
    brain.StoreMemory(stateNow,actionNow,rewardNow,stateNext)

    if counterStep > numPreStoreMemory and (counterStep % 5==0):
        brain.Learn()

    counterStep+=1

    if doneNow:
        env.Reset()
        stateNext=env.GetState()

        iEpisode+=1
        #print(iEpisode,brain.factorGreedyEpsilon)
        if iEpisode>numEpisode:
            break

    env.SetState(stateNext)

    #brain.PlotLoss()














print('END@DQN')
