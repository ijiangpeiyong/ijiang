# Author : Peiyong Jiang
# jiangpeiyong@126.com

import numpy as np
from scipy import constants as const
import matplotlib.pyplot as plt

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
        self.CalGammaCs()
        self.CalBetaCs()
        self.emitNatureX=self.emitNormX/(self.gammaCs*self.betaCs)
        
    def CalEmitNatureY(self):     #　　计算束流的自然发射度　Y
        self.CalGammaCs()
        self.CalBetaCs()
        self.emitNatureY=self.emitNormY/(self.gammaCs*self.betaCs)

    def CalEmitNatureZ(self):     #　　计算束流的自然发射度　Z
        self.CalGammaCs()
        self.CalBetaCs()
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
        self.x, self.xp, self.y ,self.yp,self.z,self.zp = np.random.multivariate_normal(mean, cov, self.numPart).T

    def K4dUzGdpp(self):   # xy的4个方向是 KV 4d，z方向是均匀的，dpp是gs的。
        self.K4d()
        self.Uz()
        self.Gdpp()

    def W6d(self):        # xyz六个方向都是wb的
        numPart=np.int32(self.numPart*3.57)
        dataRandom=np.random.random((numPart,6))*2.-1.
        r=dataRandom[:,0]**2+dataRandom[:,1]**2+dataRandom[:,2]**2+dataRandom[:,3]**2+dataRandom[:,4]**2+dataRandom[:,5]**2
        indexR=r<1.
        dataRandom=dataRandom[indexR,:]
        self.x, self.xp, self.y ,self.yp,self.z,self.zp=dataRandom[0:self.numPart,0],dataRandom[0:self.numPart,1],dataRandom[0:self.numPart,2],dataRandom[0:self.numPart,3],dataRandom[0:self.numPart,4],dataRandom[0:self.numPart,5]


    def W4dUzGdpp(self):   # xy的4个方向是 WB 4d，z方向是均匀的，dpp是gs的。
        self.W4d()
        self.Uz()
        self.Gdpp()

    ####################################################################
    #---------------------------------------------

    def CalTwiss4DMatrix(self):      # 得到用于生成束流分布的４Ｄ　ｃｏｖ矩阵
        self.twiss4DMatrix=np.array([[self.emitNatureX*self.twissBetaX,-self.emitNatureX*self.twissAlphaX,0.,0.],
        [-self.emitNatureX*self.twissAlphaX,self.emitNatureX*self.twissGammaX,0.,0.],
        [-0.,0.,self.emitNatureY*self.twissBetaY,-self.emitNatureY*self.twissAlphaY],
        [-0.,0.,-self.emitNatureY*self.twissAlphaY,self.emitNatureY*self.twissGammaY]])
 
    def CalTwiss6DMatrix(self):      # 得到用于生成束流分布的６Ｄ　ｃｏｖ矩阵
        self.twiss6DMatrix=np.array([[self.emitNatureX*self.twissBetaX,-self.emitNatureX*self.twissAlphaX,0.,0.,0.,0.],
        [-self.emitNatureX*self.twissAlphaX,self.emitNatureX*self.twissGammaX,0.,0.,0.,0.],
        [-0.,0.,self.emitNatureY*self.twissBetaY,-self.emitNatureY*self.twissAlphaY,0.,0.],
        [-0.,0.,-self.emitNatureY*self.twissAlphaY,self.emitNatureY*self.twissGammaY,0.,0.],
        [-0.,0.,0.,0.,self.emitNatureZ*self.twissBetaZ,-self.emitNatureZ*self.twissAlphaZ],
        [-0.,0.,0.,0.,-self.emitNatureZ*self.twissAlphaZ,self.emitNatureZ*self.twissGammaZ]])


    def CalTwiss4DMatrixEigen(self):       # 对用于生成束流分布的４ｄ　ｃｏｖ矩阵进行本征分析
        self.twiss4DMatrixEig,self.twiss4DMatrixVec=np.linalg.eig(self.twiss4DMatrix)
        self.twiss4DMatrixEigDiagSqrt=np.diag(np.sqrt(self.twiss4DMatrixEig))
        

    def CalTwiss6DMatrixEigen(self):       # 对用于生成束流分布的６ｄ　ｃｏｖ矩阵进行本征分析
        twiss6DMatrixEig,self.twiss6DMatrixVec=np.linalg.eig(self.twiss6DMatrix)
        self.twiss6DMatrixEigDiagSqrt=np.diag(np.sqrt(twiss6DMatrixEig))

    def CalBeamExtension4D(self):          # 生成束流时候，用于束流拉伸　　＠　４Ｄ
        self.x,self.xp,self.y,self.yp=np.dot(self.twiss6DMatrixEigDiagSqrt,[self.x,self.xp,self.y,self.yp])
    
    def CalBeamExtension6D(self):         # 生成束流时候，用于束流拉伸　　＠　６Ｄ
        self.x,self.xp,self.y,self.yp,self.z,self.zp=np.dot(self.twiss6DMatrixEigDiagSqrt,[self.x,self.xp,self.y,self.yp,self.z,self.zp])

    def CalBeamRotation4D(self):         # 生成束流时候，用于束流旋转　　＠　４Ｄ
        self.x,self.xp,self.y,self.yp=np.dot(self.twiss6DMatrixVec,[self.x,self.xp,self.y,self.yp])

    def CalBeamRotation6D(self):        # 生成束流时候，用于束流旋转　　＠　６Ｄ
        self.x,self.xp,self.y,self.yp,self.z,self.zp=np.dot(self.twiss6DMatrixVec,[self.x,self.xp,self.y,self.yp,self.z,self.zp])


    def CalBeamTranslation4D(self):
        pass

    def CalBeamTranslation6D(self):
        pass




    #---------------------------------------------
    def BeamGen(self):
        if self.beamDist=="G4dUzGdpp":
            self.G4dUzGdpp()
            self.CalTwissGammaX()
            self.CalTwissGammaY()

            self.CalEmitNatureX()
            self.CalEmitNatureY()

            self.CalTwiss4DMatrix()
            self.CalTwiss4DMatrixEigen()

            self.CalBeamExtension4D()
            self.CalBeamRotation4D()

        if self.beamDist=='G6d':
            self.G6d()
            self.CalTwissGammaX()
            self.CalTwissGammaY()
            self.CalTwissGammaZ()

            self.CalEmitNatureX()
            self.CalEmitNatureY()
            self.CalEmitNatureZ()
            
            self.CalTwiss6DMatrix()
            self.CalTwiss6DMatrixEigen()

            self.CalBeamExtension6D()
            self.CalBeamRotation6D()

        if self.beamDist=='K4dUzGdpp':
            self.K4dUzGdpp()
            self.CalTwissGammaX()
            self.CalTwissGammaY()

            self.CalEmitNatureX()
            self.CalEmitNatureY()

            self.CalTwiss4DMatrix()
            self.CalTwiss4DMatrixEigen()

            self.CalBeamExtension4D()
            self.CalBeamRotation4D()

        if self.beamDist=='W6d':
            self.W6d()
            self.CalTwissGammaX()
            self.CalTwissGammaY()
            self.CalTwissGammaZ()

            self.CalEmitNatureX()
            self.CalEmitNatureY()
            self.CalEmitNatureZ()

            self.CalTwiss6DMatrix()
            self.CalTwiss6DMatrixEigen()
    
            self.CalBeamExtension6D()
            self.CalBeamRotation6D()

        if self.beamDist=='W4dUzGdpp':
            self.W4dUzGdpp()
            self.CalTwissGammaX()
            self.CalTwissGammaY()

            self.CalEmitNatureX()
            self.CalEmitNatureY()

            self.CalTwiss4DMatrix()
            self.CalTwiss4DMatrixEigen()

            self.CalBeamExtension4D()
            self.CalBeamRotation4D()






if __name__=="__main__":
    myBeam=Beam()

    #----- 测试ＧＳ　６Ｄ
    myBeam.SetAMU(938.272)
    myBeam.SetBeamDist('G6d')
    myBeam.SetEs(0.035)
    myBeam.SetNumPart(1e5)
    myBeam.SetTwissAlphaX(-1)
    myBeam.SetTwissBetaX(1)
    myBeam.SetTwissAlphaY(1)
    myBeam.SetTwissBetaY(1)
    myBeam.SetTwissAlphaZ(0)
    myBeam.SetTwissBetaZ(1)
    myBeam.SetEmitNormX(0.22)
    myBeam.SetEmitNormY(0.22)
    myBeam.SetEmitNormZ(0.25)

    myBeam.BeamGen()

    plt.figure('gs-6d')
    plt.subplot(221)
    plt.plot(myBeam.x,myBeam.xp,'.')
    plt.axis('equal')
    plt.grid('on')
    plt.subplot(222)
    plt.plot(myBeam.y,myBeam.yp,'.')
    plt.grid('on')
    plt.axis('equal')
    plt.subplot(223)
    plt.plot(myBeam.z,myBeam.zp,'.')
    plt.grid('on')
    plt.axis('equal')
    plt.subplot(224)
    plt.plot(myBeam.x,myBeam.y,'.')
    plt.grid('on')
    plt.axis('equal')





    plt.show()

    






print('END')