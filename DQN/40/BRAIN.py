# Author: Peiyong Jiang

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BRAIN:
    def __init__(self):
        self.numAction=4
        self.numFeature=2

        self.factorGreedyEpsilon=0.5
        self.factorGreedyEpsilonInc=0.001
        self.factorGreedyEpsilonMax=1.

        self.factorRewardDecayGamma=0.9

        self.factorLearningRate=0.0001

        self.sizeMemory=1000
        self.sizeBatch=64

        self.numAssignTE=100

        self.outputNNGraph=True

        self.memory=np.zeros((self.sizeMemory,self.numFeature*2+2))

        self.counterMemory=0
        self.counterLearn=0

        self.histLoss=[]

        self.BuildNet()



        # Get all variables in the netTarget and netEval
        paramsTarget=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='netTarget')
        paramsEval=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='netEval')

        with tf.variable_scope('assignTE'):
            self.assignTE=[tf.assign(t,e) for t,e in zip(paramsTarget,paramsEval)]

        self.sess=tf.Session()
        if self.outputNNGraph:
            os.system('rm -fr ./logs/*')
            tf.summary.FileWriter("logs/",self.sess.graph)

        self.sess.run(tf.global_variables_initializer())



        #os.system('tensorboard --logdir logs')




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


    def StoreMemory(self,stateNow,counterRunNow,actionNow,rewardNow,stateNext,counterRunNext):
        pieceMemory=np.hstack((stateNow,counterRunNow,actionNow,rewardNow,stateNext,counterRunNext))
        indexMemory=self.counterMemory % self.sizeMemory
        self.memory[indexMemory,:]=pieceMemory
        self.counterMemory+=1

    def SelSamples(self):
        if self.counterMemory>self.sizeMemory:
            indexSample=np.random.choice(self.sizeMemory,size=self.sizeBatch)
        else:
            indexSample=np.random.choice(self.counterMemory,size=self.sizeBatch)

        memoryBatch=self.memory[indexSample,:]
        return memoryBatch

    def SelAction(self,stateNow,counterRunNow):
        stateNow=np.array([stateNow,counterRunNow])
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
        plt.pause(0.0000001)
        #plt.show()




if __name__ == "__main__":
    print('-'*50)
    brain=BRAIN()

    print('END @ BRAIN')
