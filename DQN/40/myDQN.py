# Author: Peiyong Jiang

import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def BuildEnv(numRow,numCol,barrier,girl):
        env=[0]*(numCol*numRow)
        for iBarrier in barrier:
            env[iBarrier]=-1
        for iGirl in girl:
            env[iGirl]=1
        return env


def Print(env,stateNow,numRow,numCol,counterRun):
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

        iRow=stateNow // numCol
        iCol=stateNow % numCol
        plt.plot(iCol,iRow,'bo',markersize='30')    

        plt.axis('equal') 
        plt.axis([-1,7,-1,5])


        pltTitle=counterRun
        plt.title(pltTitle)

        plt.pause(timeFresh)

        #plt.show()    

def Reset(env,stateNow,numRow,numCol,counterRun):
    stateNow=0
    counterRun=0
    Print(env,stateNow,numRow,numCol,counterRun)

def UpdateState(env,stateNow,actionNow,numRow,numCol,counterRun,barrier,girl):

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


        Print(env,stateNext,numRow,numCol,counterRun)

        counterRun+=1

        return stateNext,rewardNow,doneNow,counterRun



def BuildNet(numFeature,numAction,factorRewardDecayGamma):
    # 整体思路：

    # 输入
    stateNow=tf.placeholder(tf.float32,[None,numFeature],name='stateNow')
    stateNext=tf.placeholder(tf.float32,[None,numFeature],name='stateNext')
    rewardNow=tf.placeholder(tf.float32,[None,],name='rewardNow')
    actionNow=tf.placeholder(tf.int32,[None,],name='actionNow')

    # 初始化
    initializeW,initializeB=tf.random_normal_initializer(0,0.3),tf.constant_initializer(0.1)


    # build evaluate net:
    with tf.variable_scope('netEval'):
        netEval_1=tf.layers.dense(stateNow,20,tf.nn.relu,
        kernel_initializer=initializeW,bias_initializer=initializeB,name='netEval_1')

        netEval=tf.layers.dense(netEval_1,numAction,
        kernel_initializer=initializeW,bias_initializer=initializeB,name='netEval')

    # Build target net:
    with tf.variable_scope('netTarget'):
        netTarget_1=tf.layers.dense(stateNext,20,tf.nn.relu,
        kernel_initializer=initializeW,bias_initializer=initializeB,name='netTarget_1')

        netTarget=tf.layers.dense(netTarget_1,numAction,
        kernel_initializer=initializeW,bias_initializer=initializeB,name='netTarget')

    # qTarget：未来的
    # qTarget = r + gamma * qMaxS_
    with tf.variable_scope('qTarget'):
        qTarget=rewardNow+factorRewardDecayGamma*tf.reduce_max(netTarget,axis=1,name='qTarget')
        qTarget=tf.stop_gradient(qTarget)


    # qEval: 现在的
    with tf.variable_scope('qEval'):
        indexAction=tf.stack([tf.range(tf.shape(actionNow)[0],dtype=tf.int32),actionNow],axis=1)
        qEval=tf.gather_nd(params=netEval,indices=indexAction)

    # loss
    with tf.variable_scope('loss'):
        loss=tf.reduce_mean(tf.squared_difference(qTarget,qEval,name='TD_Error'),name='loss')

    # train:
    with tf.variable_scope('train'):
        train=tf.train.RMSPropOptimizer(factorLearningRate).minimize(loss)
    
    return stateNow,stateNext,rewardNow,actionNow,qTarget,qEval,loss,train

def StoreMemory(stateNow,counterRunNow,actionNow,rewardNow,stateNext,counterRunNext):
    pieceMemory=np.hstack((stateNow,counterRunNow,actionNow,rewardNow,stateNext,counterRunNext))
    indexMemory=self.counterMemory % self.sizeMemory
    memory[indexMemory,:]=pieceMemory
    counterMemory+=1
    return 



actionSpace=[0,1,2,3]   # 上，下，左，右
numAction=len(actionSpace)
numFeature=2

numCol=7
numRow=5
barrier=[34]
girl=[17]
boy=0

timeFresh=0.005

numRun=100
counterRun=0

printName='env'


factorGreedyEpsilon=0.5
factorGreedyEpsilonInc=0.001
factorGreedyEpsilonMax=1.

factorRewardDecayGamma=0.9

factorLearningRate=0.0001

sizeMemory=1000
sizeBatch=64

numAssignTE=100

outputNNGraph=True

memory=np.zeros((sizeMemory,numFeature*2+2))

counterMemory=0
counterLearn=0

histLoss=[]



#env=BuildEnv(numRow,numCol,barrier,girl)
#Print(env,boy,numRow,numCol,counterRun)
#Reset(env,boy,numRow,numCol,counterRun)

stateNow,stateNext,rewardNow,actionNow,qTarget,qEval,loss,train=BuildNet(numFeature,numAction,factorRewardDecayGamma)

paramsTarget=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='netTarget')
paramsEval=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='netEval')

with tf.variable_scope('assignTE'):
    assignTE=[tf.assign(t,e) for t,e in zip(paramsTarget,paramsEval)]

sess=tf.Session()
if outputNNGraph:
    os.system('rm -fr ./logs/*')
    tf.summary.FileWriter("logs/",sess.graph)

sess.run(tf.global_variables_initializer())














print('End@myDQN')
