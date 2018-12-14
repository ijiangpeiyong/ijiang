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



    
    print(stateNow)         

    iTest+=1
    if iTest==1:
        break
    



    '''
    actionNow=brain.SelAction(stateNow)

    stateNext,rewardNow,doneNow=env.UpdateState(stateNow,actionNow)
    brain.StoreMemory(stateNow,actionNow,rewardNow,stateNext)

    if counterStep > numPreStoreMemory and (counterStep % 5==0):
        brain.Learn()

    counterStep+=1

    env.SetState(stateNext)


    if doneNow:
        env.Reset()
        iEpisode+=1
        print(iEpisode,brain.factorGreedyEpsilon)
        if iEpisode>numEpisode:
            break




    #brain.PlotLoss()

    '''












print('END@DQN')
