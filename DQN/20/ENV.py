# Author : Peiyong Jiang
import numpy as np
import time
import matplotlib.pyplot as plt

class ENV:
    def __init__(self):
        self.actionSpace=[0,1,2,3]   # 上，下，左，右
        self.numAction=len(self.actionSpace)
        self.numCol=7
        self.numRow=5
        self.barrier=[9]
        self.girl=[17]

        self.timeFresh=0.00001

        self.numRun=30
        self.counterRun=0

        self.printName='env'

        self.BuildEnv()
        #self.Print()
        #self.Reset()


    def List2Array(self):
        self.barriernumRow
        self.barriernumRow
        self.



    def BuildEnv(self):
        self.env=[0]*self.numCol*self.numRow
        self.env[self.barrier]=-1

        print(self.env)

    def Print(self,pltTitle='self.counterRun',doneNow=False):
        plt.figure(self.printName)
        plt.clf()
        for iRow in range(self.numRow):
            for iCol in range(self.numCol):
                if self.env[iRow,iCol]==0:
                    plt.plot(iCol,iRow,'ys',markersize='50')

                elif self.env[iRow,iCol]==-1:
                    plt.plot(iCol,iRow,'ks',markersize='50')
                elif self.env[iRow,iCol]==1:
                    plt.plot(iCol,iRow,'rs',markersize='50')

        plt.plot(self.boy[1],self.boy[0],'bo',markersize='30')
        plt.axis('equal')

        if pltTitle=='self.counterRun':
            pltTitle=self.counterRun

        plt.title(pltTitle)
        plt.pause(self.timeFresh)

        if doneNow:
            plt.close(self.printName)

        #plt.show()


    def Reset(self):
        self.boy=[0,0]
        self.counterRun=0

    def UpdateState(self,stateNow,actionNow):
        if not isinstance(stateNow,list):
            stateNow=stateNow.tolist()

        stateNext=stateNow
        if actionNow==0:    # 下
            stateNext[0]-=1
            if stateNext[0]<0:
                stateNext[0]=0

        elif actionNow==1:   # 上
            stateNext[0]+=1
            if stateNext[0]>self.numRow-1:
                stateNext[0]=self.numRow-1

        elif actionNow==2:  #  左
            stateNext[1]-=1
            if stateNext[1]<0:
                stateNext[1]=0

        elif actionNow==3:  # 右
            stateNext[1]+=1
            if stateNext[1]>self.numCol-1:
                stateNext[1]=self.numCol-1

        if stateNext in self.barrier:
            rewardNow=-1
            doneNow=1
        elif stateNext in self.girl:
            rewardNow=1
            doneNow=2
        else:
            rewardNow=0
            doneNow=False

        rewardNow-=0.1

        if self.counterRun>=self.numRun:
            rewardNow-=0.5
            doneNow=3


        self.Print(doneNow=doneNow)
        self.counterRun+=1



        if doneNow:
            print(rewardNow)
        return stateNext,rewardNow,doneNow

    def GetState(self):
        return np.array(self.boy)
    def SetState(self,state):
        self.boy=state



if __name__=="__main__":
    env=ENV()


    print('END@ENV')
