# Author: Peiyong Jiang

import numpy as np
import tensorflow as tf


class BRAIN:
    def __init__(self):
        self.numAction=4
        self.numFeature=2

        self.factorGreedyEpsilon=0.9
        self.factorGreedyEpsilonInc=0.001
        self.factorGreedyEpsilonMac=0.95

        self.factorRewardDecayGamma=0.9

        self.factorLearningRate=0.1

        self.sizeMemory=3000
        self.sizeBatch=32

        self.numAssignTE=500

        self.outputNNGraph=True

        self.memory=np.zeros((self.sizeMemory,self.numFeature*2+2))


        self.BuildNet()


        # Get all variables in the netTarget and netEval
        paramsTarget=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='netTarget')
        paramsEval=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='netEval')

        with tf.variable_scope('assignTF'):
            self.assignTE=[tf.assign(t,e) for t,e in zip(paramsTarget,paramsEval)]

        self.sess=tf.Session()
        if self.outputNNGraph:
            tf.summary.FileWriter("logs/",self.sess.graph)

        self.sess.run(tf.global_variables_initializer())



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


    def StoreMemory(self):
        pass

    def SelSamples(self):
        pass

    def SelAction(self):
        pass

    def Learn(self):
        pass

    def PlotCost(self):
        pass



if __name__ == "__main__":

    brain=BRAIN()

    print('END @ BRAIN')
