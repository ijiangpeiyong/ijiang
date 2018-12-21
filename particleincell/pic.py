# Author : Peiyong Jiang
# jiangpeiyong@126.com

import numpy as np
from scipy import constants as const

class Beam():
    def __init__(self):  # 初始化
        pass

    #####################################################################
    def SetAMU(self,amu=931.494):
        self.amu=amu

    #--------------------------------------------------
    def SetNumPart(self,numPart):      # 宏粒子个数
        self.numPart=np.int32(numPart)
    
    #---------------------------------------------------
    def SetFrequency(self,freq):    # 束流的基础频率，通常用于计算束团长度 MHz
        self.freq=freq*1e6

    #----------------------------------------------------
    def SetEs(self,Es=1.5):   # 同步粒子能量   MeV
        self.Es=Es
    
    def SetZs(self,zs=0.):   # 同步粒子z向位置   mm
        self.zs=zs

    def SetXs(self,xs=0.):   # 同步粒子x向位置   mm
        self.xs=xs
    
    def SetXps(self,xps=0.):   # 同步粒子xp位置   mrad
        self.xps=xps
    
    def SetYs(self,ys=0.):    # 同步粒子y位置    mm
        self.ys=ys
    
    def SetYps(self,yps=0.):   # 同步粒子yp位置   mrad
        self.yps=yps

    #--------------------------------------------------------
    def SetEmitNormX(self,emitNormX=0.22):  #设置归一化发射度： X  mm-mrad
        self.emitNormX=emitNormX

    def SetEmitNormY(self,emitNormY=0.22):   #设置归一化发射度： Y  mm-mrad
        self.emitNormY=emitNormY

    def SetEmitNormZ(self,emitNormZ=0.25):    #设置归一化发射度： Z  mm-mrad
        self.emitNormZ=emitNormZ


    def SetTwissAlphaX(self,twissAlphaX=0.):    # 设置束流twiss参数 alpha X
        self.twissAlphaX=twissAlphaX

    def SetTwissAlphaY(self,twissAlphaY=0.):    # 设置束流twiss参数 alpha Y
        self.twissAlphaY=twissAlphaY

    def SetTwissAlphaZ(self,twissAlphaZ=0.):    # 设置束流twiss参数 alpha Z
        self.twissAlphaZ=twissAlphaZ


    def SetTwissBetaX(self,twissBetaX=1.):    # 设置束流twiss参数 beta X   m
        self.twissBetaX=twissBetaX

    def SetTwissBetaY(self,twissBetaY=1.):   # 设置束流twiss参数 beta Y    m
        self.twissBetaY=twissBetaY

    def SetTwissBetaZ(self,twissBetaZ=1.):   # 设置束流twiss参数 beta Z    m
        self.twissBetaZ=twissBetaZ

    #-------------------------------------------------
    def SetBeamLength(self,beamLength=360.):    # 设置束流长度  单位是 °  
        self.beamLength=beamLength    

    def SetBeamDpp(self,beamDpp=0.01):     # 设置束流 dp/p
        self.beamDpp=beamDpp

    #-------------------------------------------------
    def SetBeamDist(self,beamDist):      # 设置束流分布类型
        self.beamDist=beamDist
        
    #####################################################################
    #--------------------------------------------
    def CalTwissGammaX(self):     # 计算束流Twiss gamma  X
        self.twissGammaX=(1.+self.twissAlphaX**2)/self.twissBetaX
    def CalTwissGammaY(self):     # 计算束流Twiss gamma  Y
        self.twissGammaY=(1.+self.twissAlphaY**2)/self.twissBetaY
    def CalTwissGammaZ(self):     # 计算束流Twiss gamma  Z
        self.twissGammaZ=(1.+self.twissAlphaZ**2)/self.twissBetaZ

    #-------------------------------------------
    def CalEmitNatureX(self):     #　　计算束流的自然发射度　X
        self.emitNatureX=self.emitNormX/(self.gammaCs*self.betaCs)
        
    def CalEmitNatureY(self):     #　　计算束流的自然发射度　Y
        self.emitNatureY=self.emitNormY/(self.gammaCs*self.betaCs)

    def CalEmitNatureZ(self):     #　　计算束流的自然发射度　Z
        self.emitNatureZ=self.emitNormZ/(self.gammaCs**3*self.betaCs)    

    #--------------------------------------------
    def CalGammaCs(self):        # 计算同步粒子gammac
        self.gammaCs=1.+self.Es/self.amu

    def CalBetaCs(self):     # 计算同步粒子betac
        self.betaCs=np.sqrt(1.-1./self.gammaCs**2)

    def CalPs(self):
        self.ps=self.gammaCs*self.betaCs

    #---------------------------------------------
    def CalWave(self):    # 计算基础频率对应的波长    mm  !
        self.wave=const.c/self.freq*1e3      

    ##################################################################
    #-------------------------------------------- 这些都是都是生成基础数据
    def G4d(self):      # 计算4d gs 单位圆   G：gs
        mean=[0,0,0,0]
        cov=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        self.x, self.xp, self.y ,self.yp = np.random.multivariate_normal(mean, cov, self.numPart).T

    def K4d(self):     # 计算4d KV 单位圆    K：kv
        self.W4d()
        r=np.sqrt(self.x**2+self.xp**2+self.y**2+self.yp**2)
        self.x/=r
        self.xp/=r
        self.y/=r
        self.yp/=r

    def W4d(self):     # 计算4d WB  单位圆    W：WB
        numPart=np.int32(self.numPart*3.57)
        dataRandom=np.random.random((numPart,4))*2.-1.
        r=dataRandom[:,0]**2+dataRandom[:,1]**2+dataRandom[:,2]**2+dataRandom[:,3]**2
        indexR=r<1.
        dataRandom=dataRandom[indexR,:]
        self.x, self.xp, self.y ,self.yp=dataRandom[0:self.numPart,0],dataRandom[0:self.numPart,1],dataRandom[0:self.numPart,2],dataRandom[0:self.numPart,3]


    def Uz(self):     #  计算z方向 均匀  u:uniform   范围：0-1
        self.z=np.random.random((self.numPart))
    
    def Gdpp(self):     #  计算dp/p  GS分布   
        self.dpp=np.random.randn((self.numPart))

    #---------------------------- 根据基础分布，获得要求分布
    def G4dUzGdpp(self):           # xy的4个方向是 GS 4d，z方向是均匀的，dpp是gs的。 RFQ的入口分布。
        self.G4d()
        self.Uz()
        self.Gdpp()

    def G6d(self):      # xyz六个方向都是gs的
        mean=[0,0,0,0,0,0]
        cov=[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        self.x, self.xp, self.y ,self.yp,self.z,self.p = np.random.multivariate_normal(mean, cov, self.numPart).T

    def K4dUzGp(self):   # xy的4个方向是 KV 4d，z方向是均匀的，dpp是gs的。
        self.K4d()
        self.Uz()
        self.Gdpp()

    def W6d(self):        # xyz六个方向都是wb的
        numPart=np.int32(self.numPart*3.57)
        dataRandom=np.random.random((numPart,6))*2.-1.
        r=dataRandom[:,0]**2+dataRandom[:,1]**2+dataRandom[:,2]**2+dataRandom[:,3]**2+dataRandom[:,4]**2+dataRandom[:,5]**2
        indexR=r<1.
        dataRandom=dataRandom[indexR,:]
        self.x, self.xp, self.y ,self.yp,self.z,self.p=dataRandom[0:self.numPart,0],dataRandom[0:self.numPart,1],dataRandom[0:self.numPart,2],dataRandom[0:self.numPart,3],dataRandom[0:self.numPart,4],dataRandom[0:self.numPart,5]


    def W4dUzGp(self):   # xy的4个方向是 WB 4d，z方向是均匀的，dpp是gs的。
        self.W4d()
        self.Uz()
        self.Gdpp()

    ####################################################################
    #---------------------------------------------

    def Twiss4D(self):
        pass


    #---------------------------------------------
    def BeamGen(self):
        if self.beamDist=="G4dUzGp":
            self.G4dUzGp()
            self.CalTwissGammaX()
            self.CalTwissGammaY()

        if self.beamDist=='G6d':
            self.G6d()
            self.CalTwissGammaX()
            self.CalTwissGammaY()
            self.CalTwissGammaZ()

        if self.beamDist=='K4dUzGp':
            self.K4dUzGp()
            self.CalTwissGammaX()
            self.CalTwissGammaY()

        if self.beamDist=='W6d':
            self.W6d()
            self.CalTwissGammaX()
            self.CalTwissGammaY()
            self.CalTwissGammaZ()

        if self.beamDist=='W4dUzGp':
            self.W4dUzGp()
            self.CalTwissGammaX()
            self.CalTwissGammaY()







if __name__=="__main__":
    beam1=Beam()
    

    pass






print('END')