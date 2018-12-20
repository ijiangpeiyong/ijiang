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

class DQN():
    def __init__(self):
        self.ParameterGlobal()
        self.ParameterENV()
        self.ParameterBrain()

        self.InitENV()
        self.InitBrain()

        self.BuildEnv()
        self.BuildNet()

        self.InitNet()

        self.ResetEnv()


    def Input4NN(self):
        self.inputNow=np.array([self.stateNow,self.counterNow])
        self.inputNext=np.array([self.stateNext,self.counterNow+1])        


    def ParameterGlobal(self):
        self.numENV=6
        self.numBrain=6

    def ParameterENV(self):
        self.actionSpace = [0, 1, 2, 3]   # 上，下，左，右
        
        self.numCol = 7
        self.numRow = 5
        self.barrier = [7, 9, 10, 11, 16]
        #self.barrier = [34]
        self.aim = [17]

        self.timePlotFresh = 0.0005

        self.factorStoreMemoryLongGreedy=0.8
        self.counterNowMax=100


    def ParameterBrain(self):
        self.numEpisode=100000

        self.numAssignTE=100

        self.sizeMemory=5000
        self.sizeBatch=64

        self.factorGreedyEpsilon=0.7
        self.factorGreedyEpsilonInc=0.001
        self.factorGreedyEpsilonMax=1.

        self.factorRewardDecayGamma=0.9

        self.factorLearningRate=0.00001

        self.outputNNGraph=True


    def InitENV(self):
        self.numAction = len(self.actionSpace)

        self.stateNow = 0
        self.stateNext=0

        self.rewardNow=-1e10
        self.rewardGlobal=-1e10

        self.actionNow=self.actionSpace[0]
        
        self.counterNow = 0
        self.counterLearnPreStore=self.sizeBatch
        
        self.Input4NN()
        self.numFeature=len(self.inputNow)
        self.numMemoryPiece=self.numFeature*2+2

        self.printName = 'env'


    def InitBrain(self):
        self.memoryLong=np.zeros((self.sizeMemory,self.numMemoryPiece+1))
        self.memoryShort=np.zeros((0,self.numMemoryPiece))

        self.counterMemoryStore=0
        self.counterLearn=0

        self.histLoss=[]

    def BuildEnv(self):
        self.env = [0]*(self.numCol*self.numRow)
        for iBarrier in self.barrier:
            self.env[iBarrier] = -1
        for iAim in self.aim:
            self.env[iAim] = 1

    def BuildNet(self):
        # 整体思路：

        # 输入
        self.stateNowNet=tf.placeholder(tf.float32,[None,self.numFeature],name='stateNowNet')
        self.stateNextNet=tf.placeholder(tf.float32,[None,self.numFeature],name='stateNextNet')
        self.rewardNowNet=tf.placeholder(tf.float32,[None,],name='rewardNowNet')
        self.actionNowNet=tf.placeholder(tf.int32,[None,],name='actionNowNet')

        # 初始化
        initializeW,initializeB=tf.random_normal_initializer(0,0.3),tf.constant_initializer(0.1)


        # build evaluate net:
        with tf.variable_scope('netEval'):
            netEval_1=tf.layers.dense(self.stateNowNet,10,tf.nn.relu6,
            kernel_initializer=initializeW,bias_initializer=initializeB,name='netEval_1')

            netEval_2=tf.layers.dense(netEval_1,10,tf.nn.relu6,
            kernel_initializer=initializeW,bias_initializer=initializeB,name='netEval_2')


            self.netEval=tf.layers.dense(netEval_2,self.numAction,
                kernel_initializer=initializeW,bias_initializer=initializeB,name='netEval')

        # Build target net:
        with tf.variable_scope('netTarget'):
            netTarget_1=tf.layers.dense(self.stateNextNet,10,tf.nn.relu,
            kernel_initializer=initializeW,bias_initializer=initializeB,name='netTarget_1')

            netTarget_2=tf.layers.dense(netTarget_1,10,tf.nn.relu,
            kernel_initializer=initializeW,bias_initializer=initializeB,name='netTarget_2')


            self.netTarget=tf.layers.dense(netTarget_2,self.numAction,
            kernel_initializer=initializeW,bias_initializer=initializeB,name='netTarget')

        # qTarget：未来的
        # qTarget = r + gamma * qMaxS_
        with tf.variable_scope('qTarget'):
            qTarget=self.rewardNowNet+self.factorRewardDecayGamma*tf.reduce_max(self.netTarget,axis=1,name='qTarget')
            self.qTarget=tf.stop_gradient(qTarget)


        # qEval: 现在的
        with tf.variable_scope('qEval'):
            indexAction=tf.stack([tf.range(tf.shape(self.actionNowNet)[0],dtype=tf.int32),self.actionNowNet],axis=1)
            self.qEval=tf.gather_nd(params=self.netEval,indices=indexAction)

        # loss
        with tf.variable_scope('loss'):
            self.loss=tf.reduce_mean(tf.squared_difference(self.qTarget,self.qEval,name='TD_Error'),name='loss')

        # train:
        with tf.variable_scope('train'):
            self.train=tf.train.RMSPropOptimizer(self.factorLearningRate).minimize(self.loss)

    def InitNet(self):
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

    def ResetEnv(self):
        #print(self.rewardNow,self.rewardGlobal*0.6,self.rewardGlobal)
        #if (self.rewardNow>=self.rewardGlobal*0.6) or (np.random.uniform()>self.factorStoreMemoryLongGreedy):
        #    self.MemoryLong()
        #if self.rewardNow>self.rewardGlobal:
        #    self.rewardGlobal=self.rewardNow

        self.MemoryLong()

        self.memoryShort=np.zeros((0,self.numMemoryPiece))

        self.stateNow = 0.
        self.actionNow=self.actionSpace[0]
        self.rewardNow=0.
        self.stateNext=0.
        self.counterNow = 0
        self.Print()



    def Print(self):
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

        plt.title(self.counterNow)

        if self.timePlotFresh>0:
            plt.pause(self.timePlotFresh)
        else:
            plt.show()


    def MemoryPiece(self):
        self.Input4NN()
        action_reward=np.array([self.actionNow,self.rewardNow])
        self.memoryPiece=np.hstack((self.inputNow,action_reward,self.inputNext))

    def MemoryShort(self):
        self.MemoryPiece()
        self.memoryShort=np.vstack((self.memoryShort,self.memoryPiece))

    def MemoryLong(self):
        for iShort in self.memoryShort:
            iMemoryLong=np.hstack((iShort,np.array(self.rewardNow)))
            indexMemory=self.counterMemoryStore % self.sizeMemory
            self.memoryLong[indexMemory,:]=iMemoryLong
            self.counterMemoryStore+=1

        print(self.counterMemoryStore)



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
            self.rewardNow -= 0.001
            self.doneNow = False

        if self.counterNow == self.counterNowMax:
            self.rewardNow += 0
            self.doneNow = True


        self.Print()

        self.counterNow += 1

    def SelAction(self):
        self.Input4NN

        inputNow=self.inputNow[np.newaxis,:]

        if np.random.uniform()<self.factorGreedyEpsilon:
            qActionNow=self.sess.run(self.netEval,feed_dict={self.stateNowNet:inputNow})
            self.actionNow=np.argmax(qActionNow)
            #print('贪婪')
        else:
            self.actionNow=self.actionSpace[np.random.randint(0,self.numAction)]
            #print('随机')  

    def SelSamples(self):
        if self.counterMemoryStore>self.sizeMemory:
            indexSample=np.random.choice(self.sizeMemory,size=self.sizeBatch)
        else:
            indexSample=np.random.choice(self.counterMemoryStore,size=self.sizeBatch)

        self.memoryBatch=self.memoryLong[indexSample,:]

    def Learn(self):
        if self.counterLearn % self.numAssignTE==0:
            self.sess.run(self.assignTE)

        self.SelSamples()

        _,lossNow=self.sess.run([self.train,self.loss],feed_dict={
        self.stateNowNet:self.memoryBatch[:,:self.numFeature],
        self.actionNowNet:self.memoryBatch[:,self.numFeature],
        self.rewardNowNet:self.memoryBatch[:,self.numFeature+1],
        self.stateNextNet:self.memoryBatch[:,-(self.numFeature+1):-1],
        })

        self.histLoss.append(lossNow)

        self.factorGreedyEpsilon=self.factorGreedyEpsilon+self.factorGreedyEpsilonInc if self.factorGreedyEpsilon<self.factorGreedyEpsilonMax else self.factorGreedyEpsilonMax

        self.counterLearn+=1


if __name__ == '__main__':

    myDQN = DQN()
    while True:
        myDQN.SelAction()
        #print(myDQN.actionNow)
        myDQN.UpdateState()
        #print(myDQN.stateNow,myDQN.actionNow,myDQN.rewardNow,myDQN.stateNext,myDQN.doneNow)

        myDQN.MemoryShort()

        if (myDQN.counterNow % 5 ==0) and (myDQN.counterMemoryStore > myDQN.counterLearnPreStore):
            myDQN.Learn()


        if myDQN.doneNow:
            myDQN.ResetEnv()

        myDQN.counterLearn+=1
        if myDQN.counterLearn>myDQN.numEpisode:
            break
        






    #myDQN.ResetEnv()




    print('END')
