# Author : Peiyong Jiang
import numpy as np
import time
import matplotlib.pyplot as plt

class ENV:
    def __init__(self):
        self.actionSpace=[0,1,2,3]   # 上，下，左，右
        self.numAction=len(self.actionSpace)
        self.numFeature=2
        self.numCol=7
        self.numRow=5
        self.barrier=[[1,0],[1,2],[1,3],[1,4],[2,2]]
        self.girl=[2,3]
        self.boy=[0,0]

        self.timeFresh=0.1

        self.printName='env'

        self.BuildEnv()
        self.Print()
        self.Reset()


    def BuildEnv(self):
        self.env=np.zeros((self.numRow,self.numCol))
        for iBarrier in self.barrier:
            self.env[iBarrier[0],iBarrier[1]]=-1
        self.env[self.girl[0],self.girl[1]]=1


    def Print(self):
        plt.figure(self.printName)
        plt.clf()
        for iRow in range(self.numRow):
            for iCol in range(self.numCol):
                if self.env[iRow,iCol]==0:
                    plt.plot(iCol,self.numRow-iRow,'ys',markersize='50')

                elif self.env[iRow,iCol]==-1:
                    plt.plot(iCol,self.numRow-iRow,'ks',markersize='50')
                elif self.env[iRow,iCol]==1:
                    plt.plot(iCol,self.numRow-iRow,'rs',markersize='50')

        plt.plot(self.boy[0],self.numRow-self.boy[1],'bo',markersize='30')
        plt.axis('equal')
        plt.pause(self.timeFresh)


    def Reset(self):
        self.boy=[0,0]

    def UpdateState(self,stateNow,actionNow):
        stateNext=stateNow
        if actionNow==0:    # 上
            stateNext[0]-=1
            if stateNext[0]<0:
                stateNext[0]=0

        elif actionNow==1:   # 下
            stateNext[0]+=1
            if stateNext[0]>self.numRow:
                stateNext[0]=self.numRow


        elif actionNow==2:  #  左
            stateNext[1]-=1
            if stateNext[1]<0:
                stateNext[1]=0

        elif actionNow==3:  # 右
            stateNext[1]+=1
            if stateNext[1]>self.numCol:
                stateNext[1]=self.numCol

        if stateNext in self.barrier:
            rewardNow=-1
            doneNow=True
        elif stateNext in self.girl:
            rewardNow=1
            doneNow=True
        else:
            rewardNow=0
            doneNow=False

        return stateNext,rewardNow,doneNow

    def GetState(self):
        return self.boy
    def SetState(self,state):
        self.boy=state



if __name__=="__main__":
    env=ENV()


print('END@ENV')
