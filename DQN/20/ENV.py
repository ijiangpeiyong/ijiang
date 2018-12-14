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
        self.barrier=[7,9,10,11,16]
        self.girl=[17]
        self.boy=0

        self.timeFresh=0.00001

        self.numRun=30
        self.counterRun=0

        self.printName='env'

        self.BuildEnv()
        self.Print()
        self.Reset()


    def BuildEnv(self):
        self.env=[0]*(self.numCol*self.numRow)
        for iBarrier in self.barrier:
            self.env[iBarrier]=-1
        for iGirl in self.girl:
            self.env[iGirl]=1
        self.env[self.boy]=2        

    def Print(self,pltTitle='self.counterRun'):
        plt.figure(self.printName)
        plt.clf()
        
        iRow,iCol=0,0
        for iEnv in self.env: 
            if iEnv==0:
                plt.plot(iCol,iRow,'ys',markersize='40')
            elif iEnv==-1:
                plt.plot(iCol,iRow,'ks',markersize='40')
            elif iEnv==1:
                plt.plot(iCol,iRow,'rs',markersize='40')
            elif iEnv==2:
                plt.plot(iCol,iRow,'bo',markersize='30')            

            iCol+=1
            if iCol==self.numCol:
                iRow+=1
                iCol=0

        plt.axis('equal') 
        plt.axis([-1,7,-1,5])

        if pltTitle=='self.counterRun':
            pltTitle=self.counterRun
        plt.title(pltTitle)

        plt.pause(self.timeFresh)

        #plt.show()
        
    def Reset(self):
        self.boy=0
        self.counterRun=0

    def UpdateState(self,stateNow,actionNow):

        numState=self.numCol*self.numRow
        stateNext=stateNow
        
        
        if actionNow==0:    # 下
            if stateNext<self.numCol:
                stateNext-=self.numCol
        if actionNow==1:      # 上
            if stateNext>numState-numState-1:
                stateNext+=self.numCol
        if actionNow==2:    # 左
            if stateNext > 0:
                stateNext-=1
        if actionNow==3:     # 右
            if stateNow<numState-1:
                stateNext+=1


        if stateNext in self.barrier:
            rewardNow=-1
            doneNow='barrier'
        elif stateNext in self.girl:
            rewardNow=1
            doneNow='girl'
        else:
            rewardNow-=0.1
            doneNow='continue'


        self.Print()
        self.counterRun+=1

        return stateNext,rewardNow,doneNow

    def GetState(self):
        return self.boy
    def SetState(self,state):
        self.boy=state



if __name__=="__main__":
    env=ENV()


    print('END@ENV')
