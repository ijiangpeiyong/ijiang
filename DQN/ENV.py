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
        self.girl=[[2,3]]

        self.timeFresh=0.00001

        self.counterRun=0

        self.printName='env'

        self.BuildEnv()
        #self.Print()
        self.Reset()


    def BuildEnv(self):
        self.env=np.zeros((self.numRow,self.numCol))

        for iBarrier in self.barrier:
            self.env[iBarrier[0],iBarrier[1]]=-1

        for iGirl in self.girl:
            self.env[iGirl[0],iGirl[1]]=1

            #print(self.env)

    def Print(self,pltTitle='self.counterRun'):
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
            doneNow=True
        elif stateNext in self.girl:
            rewardNow=1
            doneNow=True
        else:
            rewardNow=0
            doneNow=False

        self.Print()
        self.counterRun+=1
        return stateNext,rewardNow,doneNow

    def GetState(self):
        return np.array(self.boy)
    def SetState(self,state):
        self.boy=state



if __name__=="__main__":
    env=ENV()
    stateNow=[0,0]
    actionNow=-1
    env.SetState(stateNow)
    env.UpdateState(stateNow,actionNow)


    print('END@ENV')
