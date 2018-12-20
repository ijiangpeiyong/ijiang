# Author : Peiyong Jiang
# jiangpeiyong@impcas.ac.cn
#

import os
import time
from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DQN:
    def __init__(self):
        self.numENV=6
        self.numBrain=6

        # ------- 环境 ---------------
        self.actionSpace = [0, 1, 2, 3]   # 上，下，左，右
        self.numAction = len(self.actionSpace)
        self.numCol = 7
        self.numRow = 5
        self.barrier = [7, 9, 10, 11, 16]
        self.aim = [17]

        self.rewardNow=-1e10
        self.rewardGlobal=-1e10
        self.storeMemoryGreedy=0.8

        self.timeFresh = 0.005

        self.numRunMax = 300
        self.counterRun = 0

        self.printName = 'env'

        # ------- 初始化 ： 环境 ---------------
        self.BuildEnv()
        self.InitEnv()



        
        # --------- 大脑 ------------
        self.numEpisode=100000
        
        self.numAssignTE=500

        self.sizeMemory=3000
        self.sizeBatch=128

        self.factorGreedyEpsilon=0.7
        self.factorGreedyEpsilonInc=0.001
        self.factorGreedyEpsilonMax=1.

        self.factorRewardDecayGamma=0.9

        self.factorLearningRate=0.0001

        self.outputNNGraph=True

        self.memoryLong=np.zeros((self.sizeMemory,self.numMemoryPiece))

        self.counterMemory=0
        self.counterLearn=0

        self.histLoss=[]

        self.Reset()


        '''
        # ------- 初始化 ： 大脑 ---------------

        self.BuildNet()


        # Get all variables in the netTarget and netEval
        paramsTarget=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='netTarget')
        paramsEval=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='netEval')

        # 替换神经网络的参数
        with tf.variable_scope('assignTE'):
            self.assignTE=[tf.assign(t,e) for t,e in zip(paramsTarget,paramsEval)]

        # 配置Session
        self.sess=tf.Session(config=tf.ConfigProto(
        device_count={"CPU":self.numBrain},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        ))
        self.sess.run(tf.global_variables_initializer())  

        # 可视化
        if self.outputNNGraph:
            os.system('rm -fr ./logs/*')
            tf.summary.FileWriter("logs/",self.sess.graph)
    '''
    #----------------- 环境 ----------------------------

    def InputNN(self):
        self.inputNow=np.array([self.stateNow])
        self.unputNext=np.array([self.stateNext])

    
    def BuildEnv(self):
        self.env = [0]*(self.numCol*self.numRow)
        for iBarrier in self.barrier:
            self.env[iBarrier] = -1
        for iaim in self.aim:
            self.env[iaim] = 1
    
    def InitEnv(self):
        self.stateNow = 0
        self.stateNext=0
        self.counterRun = 0

        self.InputNN()
        self.numFeature=len(self.inputNow)

        self.numMemoryPiece=self.numFeature*2+2


    def Print(self, pltTitle='self.counterRun'):
        plt.figure(self.printName)

        plt.clf()

        iRow, iCol = 0, 0
        for iEnv in self.env:
            if iEnv == 0:
                plt.plot(iCol, iRow, 'ys', markersize='40')
            elif iEnv == -1:
                plt.plot(iCol, iRow, 'ks', markersize='40')
            elif iEnv == 1:
                plt.plot(iCol, iRow, 'rs', markersize='40')

            iCol += 1
            if iCol == self.numCol:
                iRow += 1
                iCol = 0

        iRow = self.stateNow // self.numCol
        iCol = self.stateNow % self.numCol
        plt.plot(iCol, iRow, 'bo', markersize='30')

        plt.axis('equal')
        plt.axis([-1, 7, -1, 5])

        if pltTitle == 'self.counterRun':
            pltTitle = self.counterRun
        plt.title(pltTitle)

        plt.pause(self.timeFresh)

        # plt.show()

    def Reset(self):
        self.memoryShort=np.zeros((0,self.numMemoryPiece))
        self.stateNow = 0
        self.stateNext=0
        self.counterRun = 0
        self.Print()

        if (self.rewardNow>=self.rewardGlobal) or (np.random.uniform()>self.storeMemoryGreedy):
            self.MemoryLong()

    '''

    def UpdateState(self):

        numState = self.numCol*self.numRow
        self.stateNow=self.stateNext

        if self.actionNow == 0:    # 下
            if self.stateNext >= self.numCol:
                self.stateNext -= self.numCol
                # print('下')
        if self.actionNow == 1:      # 上
            if self.stateNext < numState-self.numCol:
                self.stateNext += self.numCol
                # print('上')
        if self.actionNow == 2:    # 左
            if (self.stateNext % self.numCol) > 0:
                self.stateNext -= 1
                # print('左')
        if self.actionNow == 3:     # 右
            if (self.stateNext % self.numCol) < self.numCol-1:
                self.stateNext += 1
                # print('右')

        if self.stateNext in self.barrier:
            self.rewardNow = -1
            self.doneNow = True

        elif self.stateNext in self.aim:
            self.rewardNow = 1
            self.doneNow = True
        else:
            self.rewardNow = 0
            self.doneNow = False

        if self.counterRun == self.numRunMax:
            self.rewardNow += 0
            self.doneNow = True

        self.rewardNow += 0.003

        self.Print()

        self.counterRun += 1


    #----------------- 大脑 ----------------------------

    def BuildNet(self):
        # 整体思路：

        # 输入
        self.stateNow=tf.placeholder(tf.float32,[None,self.numFeature],name='stateNow')
        self.stateNext=tf.placeholder(tf.float32,[None,self.numFeature],name='stateNext')
        self.rewardNow=tf.placeholder(tf.float32,[None,],name='rewardNow')
        self.actionNow=tf.placeholder(tf.int32,[None,],name='actionNow')

        # 初始化
        initializeW,initializeB=tf.random_normal_initializer(0,0.3),tf.constant_initializer(0.1)


        # build evaluate net:
        with tf.variable_scope('netEval'):
            netEval_1=tf.layers.dense(self.stateNow,20,tf.nn.relu,
            kernel_initializer=initializeW,bias_initializer=initializeB,name='netEval_1')

            self.netEval=tf.layers.dense(netEval_1,self.numAction,
            kernel_initializer=initializeW,bias_initializer=initializeB,name='netEval')

        # Build target net:
        with tf.variable_scope('netTarget'):
            netTarget_1=tf.layers.dense(self.stateNext,20,tf.nn.relu,
            kernel_initializer=initializeW,bias_initializer=initializeB,name='netTarget_1')

            self.netTarget=tf.layers.dense(netTarget_1,self.numAction,
            kernel_initializer=initializeW,bias_initializer=initializeB,name='netTarget')

        # qTarget：未来的
        # qTarget = r + gamma * qMaxS_
        with tf.variable_scope('qTarget'):
            qTarget=self.rewardNow+self.factorRewardDecayGamma*tf.reduce_max(self.netTarget,axis=1,name='qTarget')
            self.qTarget=tf.stop_gradient(qTarget)


        # qEval: 现在的
        with tf.variable_scope('qEval'):
            indexAction=tf.stack([tf.range(tf.shape(self.actionNow)[0],dtype=tf.int32),self.actionNow],axis=1)
            self.qEval=tf.gather_nd(params=self.netEval,indices=indexAction)

        # loss
        with tf.variable_scope('loss'):
            self.loss=tf.reduce_mean(tf.squared_difference(self.qTarget,self.qEval,name='TD_Error'),name='loss')

        # train:
        with tf.variable_scope('train'):
            self.train=tf.train.RMSPropOptimizer(self.factorLearningRate).minimize(self.loss)

    '''

    def MemoryPiece(self):
        self.memoryPiece=np.hstack((self.stateNow,self.actionNow,self.rewardNow,self.stateNext))

    def MemoryShort(self):
        self.memoryShort=np.vstack((self.memoryShort,self.memoryPiece))

    def MemoryLong(self):
        for iShort in self.memoryShort:
            indexMemory=self.counterMemory % self.sizeMemory
            self.memoryLong[indexMemory,:]=iShort
            self.counterMemory+=1

    '''

    def SelSamples(self):
        if self.counterMemory>self.sizeMemory:
            indexSample=np.random.choice(self.sizeMemory,size=self.sizeBatch)
        else:
            indexSample=np.random.choice(self.counterMemory,size=self.sizeBatch)

        self.memoryBatch=self.memoryLong[indexSample,:]
        

    def SelAction(self):
        stateNow=np.array([stateNow])
        stateNow=stateNow[np.newaxis,:]

        if np.random.uniform()<self.factorGreedyEpsilon:
            qActionNow=self.sess.run(self.netEval,feed_dict={self.stateNow:stateNow})
            actionNow=np.argmax(qActionNow)
            #print('贪婪')
        else:
            actionNow=np.random.randint(0,self.numAction)
            #print('随机')
        return actionNow


    def Learn(self):
        if self.counterLearn % self.numAssignTE==0:
            #print('AssignTE')
            self.sess.run(self.assignTE)

        memoryBatch=self.SelSamples()

        _,lossNow=self.sess.run([self.train,self.loss],feed_dict={
        self.stateNow:memoryBatch[:,:self.numFeature],
        self.actionNow:memoryBatch[:,self.numFeature],
        self.rewardNow:memoryBatch[:,self.numFeature+1],
        self.stateNext:memoryBatch[:,-self.numFeature:],
        })

        self.histLoss.append(lossNow)

        self.factorGreedyEpsilon=self.factorGreedyEpsilon+self.factorGreedyEpsilonInc if self.factorGreedyEpsilon<self.factorGreedyEpsilonMax else self.factorGreedyEpsilonMax

        self.counterLearn+=1

    def PlotLoss(self):
        plt.figure('loss')
        plt.clf()
        #print(self.histLoss)
        #plt.plot(np.arange(len(self.histLoss)),self.histLoss)
        numPrint=100
        if len(self.histLoss)>numPrint:
            plt.plot(self.histLoss[-numPrint:])
        else:
            plt.plot(self.histLoss)
        plt.ylabel('loss')
        plt.xlabel('training step')
        plt.title(self.factorGreedyEpsilon)
        plt.pause(0.001)
        #plt.show()
    '''










if __name__ == '__main__':

    myDQN = DQN()



    '''
    numEpisode=1000000

    counterStep=0
    iEpisode=0
    numPreStoreMemory=10

    myDQN.Reset()

    iTest=0
    while True:
        stateNow,counterRunNow=myDQN.GetState()
        actionNow=brain.SelAction(stateNow,counterRunNow)

        stateNext,rewardNow, doneNow,counterRunNow=env.UpdateState(stateNow,actionNow)
    
        #print(stateNow,actionNow,stateNext)
    
        counterRunNext=counterRunNow+1

        brain.StoreMemory(stateNow,counterRunNow,actionNow,rewardNow,stateNext,counterRunNext)

        if counterStep > numPreStoreMemory and (counterStep % 5==0):
            brain.Learn()

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
    '''


    print('END')
