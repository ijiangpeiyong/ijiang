# Author: Peiyong Jiang

import numpy as np
import time
import matplotlib.pyplot as plt

def BuildEnv(numRow,numCol,barrier,girl):
        env=[0]*(numCol*numRow)
        for iBarrier in barrier:
            env[iBarrier]=-1
        for iGirl in girl:
            env[iGirl]=1
        return env


def Print(env,boy,numRow,numCol,counterRun):
        plt.figure(printName)

        plt.clf()
        
        iRow,iCol=0,0
        for iEnv in env: 
            if iEnv==0:
                plt.plot(iCol,iRow,'ys',markersize='40')
            elif iEnv==-1:
                plt.plot(iCol,iRow,'ks',markersize='40')
            elif iEnv==1:
                plt.plot(iCol,iRow,'rs',markersize='40')         

            iCol+=1
            if iCol==numCol:
                iRow+=1
                iCol=0

        iRow=boy // numCol
        iCol=boy % numCol
        plt.plot(iCol,iRow,'bo',markersize='30')    

        plt.axis('equal') 
        plt.axis([-1,7,-1,5])


        pltTitle=counterRun
        plt.title(pltTitle)

        plt.pause(timeFresh)

        #plt.show()    

def Reset(env,boy,numRow,numCol,counterRun):
    boy=0
    counterRun=0
    Print(env,boy,numRow,numCol,counterRun)

def UpdateState(stateNow,actionNow,boy,numRow,numCol,counterRun):

        numState=numCol*numRow
        stateNext=stateNow
        
        if actionNow==0:    # 下
            if stateNext>=numCol:
                stateNext-=numCol
                #print('下')
        if actionNow==1:      # 上
            if stateNext<numState-numCol:
                stateNext+=numCol
                #print('上')
        if actionNow==2:    # 左
            if (stateNext % numCol) > 0:
                stateNext-=1
                #print('左')
        if actionNow==3:     # 右
            if (stateNext % numCol) < numCol-1:
                stateNext+=1
                #print('右')
        
        if stateNext in barrier:
            rewardNow=-1
            doneNow=True
            
        elif stateNext in girl:
            rewardNow=1
            doneNow=True
        else:
            rewardNow=0
            doneNow=False

        if counterRun==numRun:
            doneNow=True
        
        rewardNow-=0.05


        boy=stateNext

        Print()

        counterRun+=1

        counterRun=counterRun

        return stateNext,rewardNow,doneNow,counterRun







actionSpace=[0,1,2,3]   # 上，下，左，右
numAction=len(actionSpace)
numCol=7
numRow=5
barrier=[34]
girl=[17]
boy=0

timeFresh=0.005

numRun=100
counterRun=0

printName='env'



env=BuildEnv(numRow,numCol,barrier,girl)
Print(env,boy,numRow,numCol,counterRun)
Reset(env,boy,numRow,numCol,counterRun)









print('End@myDQN')
