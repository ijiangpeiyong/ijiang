# Author: Peiyong Jiang

from ENV import ENV
from BRAIN import BRAIN

import numpy as np


numEpisode=100

env=ENV()
brain=BRAIN()

counterStep=0
numPreStoreMemory=200

for iEpisode in range(numEpisode):
    env.Reset()
    while True:
        stateNow=env.GetState()

        actionNow=brain.SelAction(stateNow)

        stateNext,rewardNow,doneNow=env.UpdateState(stateNow,actionNow)
        brain.StoreMemory(stateNow,actionNow,rewardNow,stateNext)

        if counterStep > numPreStoreMemory and (counterStep % 5==0):
            brain.Learn()

        env.SetState(stateNext)

        if doneNow:
            print(doneNow)
            break
        counterStep+=1


    #brain.PlotLoss()














print('END@DQN')
