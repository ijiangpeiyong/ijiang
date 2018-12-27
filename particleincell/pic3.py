# Author : Peiyong Jiang
# jiangpeiyong@126.com


import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants as const
from scipy import sparse
from scipy.fftpack import dstn, idstn, irfft, rfft, dst, idst, dct, idct, fft, ifft


class Beam():
    def __init__(self):  # 初始化
        pass

    #####################################################################
    def SetAMU(self,amu=938.272):
        self.amu=amu

    #--------------------------------------------------
    def SetNumPart(self,numPart):      # 宏粒子个数
        self.numPart=np.int32(numPart)
        #self.loss=np.full((self.numPart),np.nan)


    #---------------------------------------------------
    '''
    def SetBeamMass(self,beamMass=1.):
        self.beamMass=np.full((self.numPart),beamMass)

    def SetBeamCharge(self,beamCharge=1.):
        self.beamCharge=np.full((self.numPart),beamCharge)    
    '''

    def SetGenBeamMass(self,genBeamMass=1.):
        self.genBeamMass=genBeamMass

    def SetGenBeamCharge(self,genBeamCharge=1.):
        self.genBeamCharge=genBeamCharge    


    #---------------------------------------------------
    def SetGenFreq(self,genFreq):    # 束流的基础频率，通常用于计算束团长度 MHz
        self.genFreq=genFreq*1e6

    #----------------------------------------------------
    def SetGenEs(self,genEs=1.5):   # 同步粒子能量   MeV
        self.genEs=genEs

    def SetGenDPPs(self,genDPPs=0.):    # 同步粒子的Ｄｐｐ
        self.genDPPs=genDPPs
    
    def SetGenZs(self,genZs=0.):   # 同步粒子z向位置   mm
        self.genZs=genZs

    def SetGenZPs(self,genZPs=0.):   # 同步粒子zp位置   mrad
        self.genZPs=genZPs
    

    def SetGenXs(self,genXs=0.):   # 同步粒子x向位置   mm
        self.genXs=genXs
    
    def SetGenXPs(self,genXPs=0.):   # 同步粒子xp位置   mrad
        self.genXPs=genXPs
    
    def SetGenYs(self,genYs=0.):    # 同步粒子y位置    mm
        self.genYs=genYs
    
    def SetGenYPs(self,genYPs=0.):   # 同步粒子yp位置   mrad
        self.genYPs=genYPs

    #--------------------------------------------------------
    def SetGenEmitNormX(self,genEmitNormX=0.22):  #设置归一化发射度： X  mm-mrad
        self.genEmitNormX=genEmitNormX

    def SetGenEmitNormY(self,genEmitNormY=0.22):   #设置归一化发射度： Y  mm-mrad
        self.genEmitNormY=genEmitNormY

    def SetGenEmitNormZ(self,genEmitNormZ=0.25):    #设置归一化发射度： Z  mm-mrad
        self.genEmitNormZ=genEmitNormZ


    def SetGenTwissAlphaX(self,genTwissAlphaX=0.):    # 设置束流twiss参数 alpha X
        self.genTwissAlphaX=genTwissAlphaX

    def SetGenTwissAlphaY(self,genTwissAlphaY=0.):    # 设置束流twiss参数 alpha Y
        self.genTwissAlphaY=genTwissAlphaY

    def SetGenTwissAlphaZ(self,genTwissAlphaZ=0.):    # 设置束流twiss参数 alpha Z
        self.genTwissAlphaZ=genTwissAlphaZ


    def SetGenTwissBetaX(self,genTwissBetaX=1.):    # 设置束流twiss参数 beta X   m
        self.genTwissBetaX=genTwissBetaX

    def SetGenTwissBetaY(self,genTwissBetaY=1.):   # 设置束流twiss参数 beta Y    m
        self.genTwissBetaY=genTwissBetaY

    def SetGenTwissBetaZ(self,genTwissBetaZ=1.):   # 设置束流twiss参数 beta Z    m
        self.genTwissBetaZ=genTwissBetaZ

    #-------------------------------------------------
    def SetGenBeamLength(self,genBeamLength=360.):    # 设置束流长度  单位是 °  　　全宽
        self.genBeamLength=genBeamLength    

    def SetGenBeamDpp(self,genBeamDpp=0.01):     # 设置束流 dp/p  半高
        self.genBeamDpp=genBeamDpp

    #-------------------------------------------------
    def SetGenBeamDist(self,genBeamDist):      # 设置生成束流分布类型
        self.genBeamDist=genBeamDist

        
    #####################################################################
    #--------------------------------------------
    def GenTwissGammaX(self):     # 计算束流Twiss gamma  X
        self.genTwissGammaX=(1.+self.genTwissAlphaX**2)/self.genTwissBetaX
    def GenTwissGammaY(self):     # 计算束流Twiss gamma  Y
        self.genTwissGammaY=(1.+self.genTwissAlphaY**2)/self.genTwissBetaY
    def GenTwissGammaZ(self):     # 计算束流Twiss gamma  Z
        self.genTwissGammaZ=(1.+self.genTwissAlphaZ**2)/self.genTwissBetaZ

    #-------------------------------------------
    def GenEmitNatureX(self):     #　　计算束流的自然发射度　X
        self.GenGammaCs()
        self.GenBetaCs()
        self.genEmitNatureX=self.genEmitNormX/(self.genGammaCs*self.genBetaCs)
        
    def GenEmitNatureY(self):     #　　计算束流的自然发射度　Y
        self.GenGammaCs()
        self.GenBetaCs()
        self.genEmitNatureY=self.genEmitNormY/(self.genGammaCs*self.genBetaCs)

    def GenEmitNatureZ(self):     #　　计算束流的自然发射度　Z
        self.GenGammaCs()
        self.GenBetaCs()
        self.genEmitNatureZ=self.genEmitNormZ/(self.genGammaCs**3*self.genBetaCs)    

    #--------------------------------------------
    def GenGammaCs(self):        # 计算同步粒子gammac
        self.genGammaCs=1.+self.genEs/self.amu

    def GenBetaCs(self):     # 计算同步粒子betac
        self.genBetaCs=np.sqrt(1.-1./self.genGammaCs**2)

    def GenPs(self):
        self.genPs=self.genGammaCs*self.genBetaCs

    #---------------------------------------------
    def GenWave(self):    # 计算基础频率对应的波长    mm  !
        self.genWave=const.c/self.genFreq*1e3      

    ##################################################################
    #-------------------------------------------- 这些都是都是生成基础数据
    def GenG4d(self):      # 计算4d gs 单位圆   G：gs
        mean=[0,0,0,0]
        cov=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        self.genX, self.genXP, self.genY ,self.genYP = np.random.multivariate_normal(mean, cov, self.genNumPart).T

    def GenK4d(self):     # 计算4d KV 单位圆    K：kv
        self.GenW4d()
        r=np.sqrt(self.genX**2+self.genXP**2+self.genY**2+self.genYP**2)
        self.genX/=r
        self.genXP/=r
        self.genY/=r
        self.genYP/=r

    def GenW4d(self):     # 计算4d WB  单位圆    W：WB
        numPart=np.int32(self.genNumPart*3.57)
        dataRandom=np.random.random((numPart,4))*2.-1.
        r=dataRandom[:,0]**2+dataRandom[:,1]**2+dataRandom[:,2]**2+dataRandom[:,3]**2
        indexR=r<1.
        dataRandom=dataRandom[indexR,:]
        self.genX, self.genXP, self.genY ,self.genYP=dataRandom[0:self.genNumPart,0],dataRandom[0:self.genNumPart,1],dataRandom[0:self.genNumPart,2],dataRandom[0:self.genNumPart,3]


    def GenUz(self):     #  计算z方向 均匀  u:uniform   范围：-0.5~0.5
        self.genZ=np.random.random((self.genNumPart))-0.5
    
    def GenGdpp(self):     #  计算dp/p  GS分布   
        self.genDpp=np.random.randn((self.genNumPart))

    #---------------------------- 根据基础分布，获得要求分布
    def GenG4dUzGdpp(self):           # xy的4个方向是 GS 4d，z方向是均匀的，dpp是gs的。 RFQ的入口分布。
        self.GenG4d()
        self.GenUz()
        self.GenGdpp()
        

    def GenG6d(self):      # xyz六个方向都是gs的
        mean=[0,0,0,0,0,0]
        cov=[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        self.genX, self.genXP, self.genY ,self.genYP,self.genZ,self.genZP = np.random.multivariate_normal(mean, cov, self.genNumPart).T

    def GenK4dUzGdpp(self):   # xy的4个方向是 KV 4d，z方向是均匀的，dpp是gs的。
        self.GenK4d()
        self.GenUz()
        self.GenGdpp()

    def GenW6d(self):        # xyz六个方向都是wb的
        numPart=np.int32(self.genNumPart*14.25)
        dataRandom=np.random.random((numPart,6))*2.-1.
        r=dataRandom[:,0]**2+dataRandom[:,1]**2+dataRandom[:,2]**2+dataRandom[:,3]**2+dataRandom[:,4]**2+dataRandom[:,5]**2
        indexR=r<1.
        dataRandom=dataRandom[indexR,:]
        self.genX, self.genXP, self.genY ,self.genYP,self.genZ,self.genZP=dataRandom[0:self.genNumPart,0],dataRandom[0:self.genNumPart,1],dataRandom[0:self.genNumPart,2],dataRandom[0:self.genNumPart,3],dataRandom[0:self.genNumPart,4],dataRandom[0:self.genNumPart,5]


    def GenW4dUzGdpp(self):   # xy的4个方向是 WB 4d，z方向是均匀的，dpp是gs的。
        self.GenW4d()
        self.GenUz()
        self.GenGdpp()

    ####################################################################
    #---------------------------------------------

    def GenTwiss4Dcov(self):      # 得到用于生成束流分布的４Ｄ　ｃｏｖ矩阵
        self.genTwiss4Dcov=np.array([[self.genEmitNatureX*self.genTwissBetaX,-self.genEmitNatureX*self.genTwissAlphaX,0.,0.],
        [-self.genEmitNatureX*self.genTwissAlphaX,self.genEmitNatureX*self.genTwissGammaX,0.,0.],
        [-0.,0.,self.genEmitNatureY*self.genTwissBetaY,-self.genEmitNatureY*self.genTwissAlphaY],
        [-0.,0.,-self.genEmitNatureY*self.genTwissAlphaY,self.genEmitNatureY*self.genTwissGammaY]])
 
    def GenTwiss6Dcov(self):      # 得到用于生成束流分布的６Ｄ　ｃｏｖ矩阵
        self.genTwiss6Dcov=np.array([[self.genEmitNatureX*self.genTwissBetaX,-self.genEmitNatureX*self.genTwissAlphaX,0.,0.,0.,0.],
        [-self.genEmitNatureX*self.genTwissAlphaX,self.genEmitNatureX*self.genTwissGammaX,0.,0.,0.,0.],
        [-0.,0.,self.genEmitNatureY*self.genTwissBetaY,-self.genEmitNatureY*self.genTwissAlphaY,0.,0.],
        [-0.,0.,-self.genEmitNatureY*self.genTwissAlphaY,self.genEmitNatureY*self.genTwissGammaY,0.,0.],
        [-0.,0.,0.,0.,self.genEmitNatureZ*self.genTwissBetaZ,-self.genEmitNatureZ*self.genTwissAlphaZ],
        [-0.,0.,0.,0.,-self.genEmitNatureZ*self.genTwissAlphaZ,self.genEmitNatureZ*self.genTwissGammaZ]])


    def GenTwiss4DcovEigen(self):       # 对用于生成束流分布的４ｄ　ｃｏｖ矩阵进行本征分析
        twiss4DcovEig,self.genTwiss4DcovVec=np.linalg.eig(self.genTwiss4Dcov)
        self.genTwiss4DcovEigDiagSqrt=np.diag(np.sqrt(twiss4DcovEig))
        

    def GenTwiss6DcovEigen(self):       # 对用于生成束流分布的６ｄ　ｃｏｖ矩阵进行本征分析
        twiss6DMatrixEig,self.genTwiss6DcovVec=np.linalg.eig(self.genTwiss6Dcov)
        self.genTwiss6DcovEigDiagSqrt=np.diag(np.sqrt(twiss6DMatrixEig))

    def GenBeamExtensionZ(self):             # 生成束流时候，用于束流拉伸　　＠   z
        self.GenWave()
        self.GenGammaCs()
        self.GenBetaCs()
        lengthBetaLambda=self.genWave*self.genBetaCs
        lengthBeam=lengthBetaLambda*self.genBeamLength/360.
        self.genZ*=lengthBeam
        
    def GenBeamExtensionDpp(self):          # 生成束流时候，用于束流拉伸　　@    dpp
        self.genDpp*=self.genBeamDpp


    def GenBeamExtension4D(self):          # 生成束流时候，用于束流拉伸　　＠　４Ｄ
        self.genX,self.genXP,self.genY,self.genYP=np.dot(self.genTwiss4DcovEigDiagSqrt,[self.genX,self.genXP,self.genY,self.genYP])
    
    def GenBeamExtension6D(self):         # 生成束流时候，用于束流拉伸　　＠　６Ｄ
        self.genX,self.genXP,self.genY,self.genYP,self.genZ,self.genZP=np.dot(self.genTwiss6DcovEigDiagSqrt,[self.genX,self.genXP,self.genY,self.genYP,self.genZ,self.genZP])


    def GenBeamRotation4D(self):         # 生成束流时候，用于束流旋转　　＠　４Ｄ
        self.genX,self.genXP,self.genY,self.genYP=np.dot(self.genTwiss4DcovVec,[self.genX,self.genXP,self.genY,self.genYP])

    def GenBeamRotation6D(self):        # 生成束流时候，用于束流旋转　　＠　６Ｄ
        self.genX,self.genXP,self.genY,self.genYP,self.genZ,self.genZP=np.dot(self.genTwiss6DcovVec,[self.genX,self.genXP,self.genY,self.genYP,self.genZ,self.genZP])


    def GenBeamTranslation4D(self):           # 生成束流时候，用于束流平移　　＠　４Ｄ
        self.genX+=self.genXs 
        self.genXP+=self.genXPs
        self.genY+=self.genYs
        self.genYP+=self.genYPs
    
    def GenBeamTranslationZ(self):      # 生成束流时候，用于束流平移　　＠　Z   
        self.genZ+=self.genZs
    
    def GenBeamTranslationDpp(self):      # 生成束流时候，用于束流平移　　＠　Dpp
        self.genDpp+=self.genDPPs


    def GenBeamTranslation6D(self):           # 生成束流时候，用于束流平移　　＠　６ｄ
        self.genX+=self.genXs 
        self.genXP+=self.genXPs
        self.genY+=self.genYs
        self.genYP+=self.genYPs
        self.genZ+=self.genZs
        self.genZP+=self.genZPs


    #################################################################
    #---------------------------------------------
    def GenBeamOnDist(self):        # 根据束流分布类型，获得ｘ－ｐｘ－ｙ－ｐｙ－ｚ－ｐｚ
        if self.genBeamDist=="G4dUzGdpp":
            # G4D:
            self.GenG4dUzGdpp()

            self.GenTwissGammaX()
            self.GenTwissGammaY()

            self.GenEmitNatureX()
            self.GenEmitNatureY()

            self.GenTwiss4Dcov()
            self.GenTwiss4DcovEigen()

            self.GenBeamExtension4D()
            self.GenBeamRotation4D()

            self.GenBeamExtensionZ()
            self.GenBeamExtensionDpp()

            self.GenBeamTranslation4D()
            self.GenBeamTranslationZ()
            self.GenBeamTranslationDpp()

            self.GenGammaCs()
            self.GenBetaCs()
            self.GenPs()

            self.z=self.genZ           #  [mm]
            self.pz=self.genPs*(1.0+self.genDpp)    # [betaC * gammaC]

            self.x=self.genX          #    [mm]
            self.px=self.pz*self.genXP/1000.       # [betaC * gammaC]

            self.y=self.genY         #     [mm]
            self.py=self.pz*self.genYP/1000.       # [betaC * gammaC]

    

        if self.genBeamDist=='K4dUzGdpp':
            self.GenK4dUzGdpp()

            self.GenTwissGammaX()
            self.GenTwissGammaY()

            self.GenEmitNatureX()
            self.GenEmitNatureY()

            self.GenTwiss4Dcov()
            self.GenTwiss4DcovEigen()

            self.GenBeamExtension4D()
            self.GenBeamRotation4D()

            self.GenBeamExtensionZ()
            self.GenBeamExtensionDpp()

            self.GenBeamTranslation4D()
            self.GenBeamTranslationZ()
            self.GenBeamTranslationDpp()

            self.GenGammaCs()
            self.GenBetaCs()
            self.GenPs()

            self.z=self.genZ           #  [mm]
            self.pz=self.genPs*(1.0+self.genDpp)    # [betaC * gammaC]

            self.x=self.genX          #    [mm]
            self.px=self.pz*self.genXP/1000.       # [betaC * gammaC]

            self.y=self.genY         #     [mm]
            self.py=self.pz*self.genYP/1000.       # [betaC * gammaC]




        if self.genBeamDist=='W4dUzGdpp':
            self.GenW4dUzGdpp()

            self.GenTwissGammaX()
            self.GenTwissGammaY()

            self.GenEmitNatureX()
            self.GenEmitNatureY()

            self.GenTwiss4Dcov()
            self.GenTwiss4DcovEigen()

            self.GenBeamExtension4D()
            self.GenBeamRotation4D()

            self.GenBeamExtensionZ()
            self.GenBeamExtensionDpp()

            self.GenBeamTranslation4D()
            self.GenBeamTranslationZ()
            self.GenBeamTranslationDpp()

            self.GenGammaCs()
            self.GenBetaCs()
            self.GenPs()

            self.z=self.genZ           #  [mm]
            self.pz=self.genPs*(1.0+self.genDpp)    # [betaC * gammaC]

            self.x=self.genX          #    [mm]
            self.px=self.pz*self.genXP/1000.       # [betaC * gammaC]

            self.y=self.genY         #     [mm]
            self.py=self.pz*self.genYP/1000.       # [betaC * gammaC]


        if self.genBeamDist=='G6d':
            self.GenG6d()
            self.GenTwissGammaX()
            self.GenTwissGammaY()
            self.GenTwissGammaZ()

            self.GenEmitNatureX()
            self.GenEmitNatureY()
            self.GenEmitNatureZ()
            
            self.GenTwiss6Dcov()
            self.GenTwiss6DcovEigen()

            self.GenBeamExtension6D()
            self.GenBeamRotation6D()

            self.GenBeamTranslation6D()


            self.GenGammaCs()
            self.GenBetaCs()
            self.GenPs()

            self.z=self.genZ           #  [mm]
            self.pz=self.genPs*(1.0+self.genZP/1000.)    # [betaC * gammaC]

            self.x=self.genX          #    [mm]
            self.px=self.pz*self.genXP/1000.       # [betaC * gammaC]

            self.y=self.genY         #     [mm]
            self.py=self.pz*self.genYP/1000.       # [betaC * gammaC]




        if self.genBeamDist=='W6d':
            self.GenW6d()
            self.GenTwissGammaX()
            self.GenTwissGammaY()
            self.GenTwissGammaZ()

            self.GenEmitNatureX()
            self.GenEmitNatureY()
            self.GenEmitNatureZ()

            self.GenTwiss6Dcov()
            self.GenTwiss6DcovEigen()
    
            self.GenBeamExtension6D()
            self.GenBeamRotation6D()

            self.GenBeamTranslation6D()


            self.GenGammaCs()
            self.GenBetaCs()
            self.GenPs()

            self.z=self.genZ           #  [mm]
            self.pz=self.genPs*(1.0+self.genZP/1000.)    # [betaC * gammaC]

            self.x=self.genX          #    [mm]
            self.px=self.pz*self.genXP/1000.       # [betaC * gammaC]

            self.y=self.genY         #     [mm]
            self.py=self.pz*self.genYP/1000.       # [betaC * gammaC]        


    #########################################################################################
    def BeamGen(self,genNumPart=0):
        if genNumPart==0:
            self.genNumPart=self.numPart
        else:
            self.genNumPart=genNumPart
        self.GenBeamOnDist()

        #self.BeamTrans()


        self.m=np.ones_like(self.x)*self.genBeamMass
        self.q=np.ones_like(self.x)*self.genBeamCharge
        self.loss=np.ones_like(self.x)*np.nan

        return([self.x,self.px,self.y,self.py,self.z,self.pz,self.m,self.q,self.loss])



    ###########################################################
    # 以上是束流生成程序
    # 以下是束流统计程序
    ###########################################################


    ##############################################################################################
    def SetStatBeamX(self,x,xp):        # 设置(传来)需要统计的数据  X 两个方向
        self.statX,self.statXP=x,xp
    
    def SetStatBeamY(self,y,yp):        # 设置(传来)需要统计的数据  Y　两个方向
        self.statY,self.statYP=y,yp

    def SetStatBeamZ(self,z,zp):        # 设置(传来)需要统计的数据  Z　两个方向
        self.statZ,self.statZP=z,zp

    def SetStatBeamXY(self,x,xp,y,yp):        # 设置(传来)需要统计的数据  XY四个方向
        self.statX,self.statXP,self.statY,self.statYP=x,xp,y,yp

    def SetStatBeamXYZ(self,x,xp,y,yp,z,zp):        # 设置(传来)需要统计的数据  XYZ六个方向
        self.statX,self.statXP,self.statY,self.statYP,self.statZ,self.statZP=x,xp,y,yp,z,zp
    

    def SetStatBeamDist(self,statBeamDist):    # 设置统计束流分布类型
        self.statBeamDist=statBeamDist

    #---------------------------------------------
    def StatBeamMeanX(self):        # 获得束流偏差　Ｘ
        self.statXmean=self.statX.mean()
    def StatBeamMeanXP(self):        # 获得束流偏差　ＸＰ
        self.statXPmean=self.statXP.mean()

    def StatBeamMeanY(self):        # 获得束流偏差　Ｙ
        self.statYmean=self.statY.mean()
    def StatBeamMeanYP(self):        # 获得束流偏差　ＹＰ
        self.statYPmean=self.statYP.mean()

    def StatBeamMeanZ(self):        # 获得束流偏差　Ｚ
        self.statZmean=self.statZ.mean()
    def StatBeamMeanZP(self):        # 获得束流偏差　ＺＰ
        self.statZPmean=self.statZP.mean()

    #---------------------------------------------

    def StatBeamCovX(self):        # 获得束流方差　Ｘ－ＸＰ
        self.statXcov=np.cov([self.statX,self.statXP])

    def StatBeamCovY(self):        # 获得束流方差　Ｙ－ＹＰ
        self.statYcov=np.cov([self.statY,self.statYP])    

    def StatBeamCovZ(self):        # 获得束流方差　Ｚ－ＺＰ
        self.statZcov=np.cov([self.statZ,self.statZP])  

    def StatBeamCovXY(self):        # 获得束流方差４Ｄ：　Ｘ－ＸＰ－Ｙ－ＹＰ
        self.statXYcov=np.cov([self.statX,self.statXP,self.statY,self.statYP])

    def StatBeamCovXYZ(self):        # 获得束流方差６Ｄ：　Ｘ－ＸＰ－Ｙ－ＹＰ－Ｚ－ZP
        self.statXYZcov=np.cov([self.statX,self.statXP,self.statY,self.statYP,self.statZ,self.statZP])

    def BeamStatRMS(self):    # 统计束流的ＲＭＳ信息
        if self.statBeamDist=='x':
            self.StatBeamMeanX()
            self.StatBeamMeanXP()
            self.StatBeamCovX()
            
            x0=self.statXmean
            xp0=self.statXPmean

            emitX=np.sqrt(np.linalg.det(self.statXcov))
            betaX=self.statXcov[0,0]/emitX
            alphX=-self.statXcov[0,1]/emitX
            gammaX=self.statXcov[1,1]/emitX
            
            return [x0,xp0,emitX,alphX,betaX,gammaX]

        if self.statBeamDist=='y':
            self.StatBeamMeanY()
            self.StatBeamMeanYP()
            self.StatBeamCovY()
            
            y0=self.statYmean
            yp0=self.statYPmean

            emitY=np.sqrt(np.linalg.det(self.statYcov))
            betaY=self.statYcov[0,0]/emitY
            alphY=-self.statYcov[0,1]/emitY
            gammaY=self.statYcov[1,1]/emitY
            
            return [y0,yp0,emitY,alphY,betaY,gammaY]

        if self.statBeamDist=='z':
            self.StatBeamMeanZ()
            self.StatBeamMeanZP()
            self.StatBeamCovZ()
            
            z0=self.statZmean
            zp0=self.statZPmean

            emitZ=np.sqrt(np.linalg.det(self.statZcov))
            betaZ=self.statZcov[0,0]/emitZ
            alphZ=-self.statZcov[0,1]/emitZ
            gammaZ=self.statZcov[1,1]/emitZ
            
            return [z0,zp0,emitZ,alphZ,betaZ,gammaZ]

        if self.statBeamDist=='xy':
            self.StatBeamMeanX()
            self.StatBeamMeanXP()
            self.StatBeamMeanY()
            self.StatBeamMeanYP()

            self.StatBeamCovXY()


            x0=self.statXmean
            xp0=self.statXPmean
            y0=self.statYmean
            yp0=self.statYPmean

            emit4d=np.sqrt(np.linalg.det(self.statXYcov))

            return  [x0,xp0,y0,yp0,emit4d, self.statXYcov]

        if self.statBeamDist=='xyz':
            self.StatBeamMeanX()
            self.StatBeamMeanXP()
            self.StatBeamMeanY()
            self.StatBeamMeanYP()
            self.StatBeamMeanZ()
            self.StatBeamMeanZP()

            self.StatBeamCovXYZ()

            x0=self.statXmean
            xp0=self.statXPmean
            y0=self.statYmean
            yp0=self.statYPmean
            z0=self.statZmean
            zp0=self.statZPmean

            emit6d=np.sqrt(np.linalg.det(self.statXYZcov))

            return  [x0,xp0,y0,yp0,z0,zp0,emit6d, self.statXYZcov]

    def GetStatBeamMean(self,coordinate='x'):     # 获取各个方向的束流中心
        if coordinate=='x':
            return self.x[np.isnan(self.loss)].mean()
        if coordinate=='px':
            return self.px[np.isnan(self.loss)].mean()
        if coordinate=='xp':
            return self.xp[np.isnan(self.loss)].mean()
        if coordinate=='y':
            return self.y[np.isnan(self.loss)].mean()
        if coordinate=='py':
            return self.py[np.isnan(self.loss)].mean()
        if coordinate=='yp':
            return self.yp[np.isnan(self.loss)].mean()
        if coordinate=='z':
            return self.z[np.isnan(self.loss)].mean()
        if coordinate=='pz':
            return self.pz[np.isnan(self.loss)].mean()  
        if coordinate=='zp':
            return self.zp[np.isnan(self.loss)].mean()  


    def GetStatBeamStd(self,coordinate='x'):      # 获取各个方向的束流均方差
        if coordinate=='x':
            return self.x[np.isnan(self.loss)].std()
        if coordinate=='px':
            return self.px[np.isnan(self.loss)].std()
        if coordinate=='xp':
            return self.xp[np.isnan(self.loss)].std()            
        if coordinate=='y':
            return self.y[np.isnan(self.loss)].std()
        if coordinate=='py':
            return self.py[np.isnan(self.loss)].std()
        if coordinate=='yp':
            return self.yp[np.isnan(self.loss)].std()
        if coordinate=='z':
            return self.z[np.isnan(self.loss)].std()
        if coordinate=='pz':
            return self.pz[np.isnan(self.loss)].std()      
        if coordinate=='zp':
            return self.zp[np.isnan(self.loss)].std()           
            
    def GetStatBeamVar(self,coordinate='x'):        # 获取各个方向的束流方差
        if coordinate=='x':
            return self.x[np.isnan(self.loss)].var()
        if coordinate=='px':
            return self.px[np.isnan(self.loss)].var()
        if coordinate=='xp':
            return self.xp[np.isnan(self.loss)].var()
        if coordinate=='y':
            return self.y[np.isnan(self.loss)].var()
        if coordinate=='py':
            return self.py[np.isnan(self.loss)].var()
        if coordinate=='yp':
            return self.yp[np.isnan(self.loss)].var()
        if coordinate=='z':
            return self.z[np.isnan(self.loss)].var()
        if coordinate=='pz':
            return self.pz[np.isnan(self.loss)].var()   
        if coordinate=='zp':
            return self.zp[np.isnan(self.loss)].var()   


    def GetStatBeamCov(self,coordinates=['x','xp']):         # 获取各个平面的束流协方差
        x=[] 
        for iCoor in coordinates:
            x.append(eval('self.'+iCoor+'[np.isnan(self.loss)]'))
        return np.cov(x)

    def GetStatBeamEmitNatureRMS(self,coordinate='x'):      # 获取各个平面的自然发射度
        if coordinate=='x':
            cov=self.GetStatBeamCov(['x','xp'])
            return np.sqrt(np.linalg.det(cov))
        if coordinate=='y':
            cov=self.GetStatBeamCov(['y','yp'])
            return np.sqrt(np.linalg.det(cov))
        if coordinate=='z':
            cov=self.GetStatBeamCov(['y','yp'])
            return np.sqrt(np.linalg.det(cov))    

    def GetStatBeamEmitNormRMS(self,coordinate='x'):      # 获取各个平面的归一化发射度
        emitNature=self.GetStatBeamEmitNatureRMS(coordinate)
        pz0=self.GetStatBeamMean('pz')
        gammaC=np.sqrt(1.+pz0**2)
        if (coordinate=='x') or (coordinate=='y'):
            emitNorm= emitNature*pz0
        if coordinate=='z':
            emitNorm= emitNature*pz0*gammaC**2
        return emitNorm
    
    def GetStatBeamBeta(self,coordinate='x'):                # 获取各个平面的 beta
        if coordinate=='x':
            cov=self.GetStatBeamCov(['x','xp'])
        if coordinate=='y':
            cov=self.GetStatBeamCov(['y','yp'])
        if coordinate=='z':
            cov=self.GetStatBeamCov(['y','yp'])

        emitNature=np.sqrt(np.linalg.det(cov))
        beta=cov[0,0]/emitNature

        return beta         

    def GetStatBeamAlpha(self,coordinate='x'):                # 获取各个平面的 alpha
        if coordinate=='x':
            cov=self.GetStatBeamCov(['x','xp'])
        if coordinate=='y':
            cov=self.GetStatBeamCov(['y','yp'])
        if coordinate=='z':
            cov=self.GetStatBeamCov(['y','yp'])

        emitNature=np.sqrt(np.linalg.det(cov))
        alpha=-cov[0,1]/emitNature

        return alpha         

    def GetStatBeamGamma(self,coordinate='x'):                # 获取各个平面的 gamma
        if coordinate=='x':
            cov=self.GetStatBeamCov(['x','xp'])
        if coordinate=='y':
            cov=self.GetStatBeamCov(['y','yp'])
        if coordinate=='z':
            cov=self.GetStatBeamCov(['y','yp'])

        emitNature=np.sqrt(np.linalg.det(cov))
        gamma=cov[1,1]/emitNature

        return gamma 




  
    ############################################################
    ##   Trans 
    ############################################################
    def TransBeamPX2XP(self,x,px,y,py,z,pz):       #　束流从动量ｐｘ　到角度　ｘｐ　转化
        pz0=pz.mean()
        zp=(pz-pz0)/pz0*1000.
        xp=px/pz*1000.
        yp=py/pz*1000.
        return x,xp,y,yp,z,zp

    def BeamTrans(self):          #　束流从动量ｐｘ　到角度　ｘｐ　转化　，并返回接口
        self.x,self.xp,self.y,self.yp,self.z,self.zp=self.TransBeamPX2XP(self.x,self.px,self.y,self.py,self.z,self.pz)
        return [self.x,self.xp,self.y,self.yp,self.z,self.zp]


    #########################################################
    ## 往程序内部 set 束流分布
    #########################################################
    def BeamSet(self,part):         # 函数接口：　网程序内部　set　束流分布
        self.x,self.px,self.y,self.py,self.z,self.pz,self.m,self.q,self.loss=part[0,:],part[1,:],part[2,:],part[3,:],part[4,:],part[5,:],part[6,:],part[7,:],part[8,:]



    #########################################################
    ##    并行相关
    #########################################################

    def SetParaNumCPU(self,paraNumCPU=0):      # 设置并行 cpu 个数
        self.paraNumCPU=np.int32(paraNumCPU)

    def ParaAllocationBeamGen(self):      # 给每个 cpu　分配生成粒子的个数
        mpNumPart=self.ParaAllocationNumbersOnCPU(self.numPart,self.paraNumCPU)

        '''
        if self.numPart % self.paraNumCPU==0:
            mpNumPart=[self.numPart // self.paraNumCPU]*self.paraNumCPU
            return mpNumPart
        numPart=self.numPart//self.paraNumCPU+1
        numPartLast=self.numPart-numPart*(self.paraNumCPU-1)
        
        mpNumPart=[numPart]*(self.paraNumCPU-1)
        mpNumPart.append(numPartLast)
        '''
        return mpNumPart

    def ParaAllocationBeamWeigh(self):      # 给每个 cpu　分配粒子称重（ｗｅｉｇｈｔｉｎｇ）的个数
        mpNumPart=self.ParaAllocationBeamGen()
        mpWeighPart=[]

        for idNumPart in range(len(mpNumPart)):
            if idNumPart==0:
                numPartStart=0
                numPartEnd=mpNumPart[0]
            else:
                numPartStart=numPartEnd
                numPartEnd+=mpNumPart[idNumPart]

            mpWeighPart.append([self.x[numPartStart:numPartEnd],self.y[numPartStart:numPartEnd],self.z[numPartStart:numPartEnd],self.q[numPartStart:numPartEnd],self.loss[numPartStart:numPartEnd]])
        
        return mpWeighPart

    def ParaAllocationNumbersOnCPU(self,number,cpu):
        numberList=[number//cpu]*cpu
        if number % cpu !=0:
            for iCPU in range(number % cpu):
                numberList[iCPU]+=1

        return numberList


    def ParaAllocationBeamGridOnZ(self):      # 给每个 cpu　分配grid的数据量，用于ｄｓｔ.　该函数沿着ｚ轴划分数据，在ｘ－ｙ上处理数据
        mpNumGridList=self.ParaAllocationNumbersOnCPU(self.weighGridZ,self.paraNumCPU)

        mpWeighGrid=[]

        for iCPU in range(self.paraNumCPU):
            if iCPU==0:
                idStart=0
                idEnd=mpNumGridList[0]
            else:
                idStart=idEnd
                idEnd+=mpNumGridList[iCPU]

            mpWeighGrid.append(self.weighGrid[:,:,idStart:idEnd])  
 
        return mpWeighGrid

    def ParaAllocationBeamGridOnX(self):      # 给每个 cpu　分配grid的数据量，用于dst/dct.　该函数沿着x轴划分数据，在z上处理数据
        mpNumGridList=self.ParaAllocationNumbersOnCPU(self.weighGridX,self.paraNumCPU)

        mpWeighGrid=[]

        for iCPU in range(self.paraNumCPU):
            if iCPU==0:
                idStart=0
                idEnd=mpNumGridList[0]
            else:
                idStart=idEnd
                idEnd+=mpNumGridList[iCPU]


            mpWeighGrid.append(self.weighGrid[idStart:idEnd,:,:])  
 
        return mpWeighGrid

    def ParaAllocationFFTEigenOnX(self):
        mpNumFFTEigenList=self.ParaAllocationNumbersOnCPU(self.weighGridX,self.paraNumCPU)

        mpFFTEigen=[]

        for iCPU in range(self.paraNumCPU):
            if iCPU==0:
                idStart=0
                idEnd=mpNumFFTEigenList[0]
            else:
                idStart=idEnd
                idEnd+=mpNumFFTEigenList[iCPU]


            mpFFTEigen.append(self.fftK2[idStart:idEnd,:,:])  
 
        return mpFFTEigen

    def ParaAllocation_BeamGrid_FFTEigen_OnX(self):
        mpCPUList=self.ParaAllocationNumbersOnCPU(self.weighGridX,self.paraNumCPU)

        mp_weighGrid_fftK2=[]

        for iCPU in range(self.paraNumCPU):
            if iCPU==0:
                idStart=0
                idEnd=mpCPUList[0]
            else:
                idStart=idEnd
                idEnd+=mpCPUList[iCPU]


            mp_weighGrid_fftK2.append([self.weighGrid[idStart:idEnd,:,:],self.fftK2[idStart:idEnd,:,:]])  
 
        return mp_weighGrid_fftK2



    #############################################################
    ### Weighting Paricles  
    #############################################################
    

    def BeamGridSet(self,weighGrid):
        self.weighGrid=weighGrid


    def SetWeighGrid3D(self,weighGridX,weighGridY,weighGridZ):    # 设置束流 space-charge 求解域　格点数
        self.weighGridX=2**np.int32(weighGridX)
        self.weighGridY=2**np.int32(weighGridY)
        self.weighGridZ=2**np.int32(weighGridZ)


    def SetWeighFreq(self,weighFreq):    # 设置束流 space-charge 求解域 的求解基准频率，用于纵向周期长度设定
        self.weighFreq=weighFreq*1e6
        self.weighWave=const.c/self.weighFreq*1e3

    def SetWeighBoundaryX(self,weighXmax=20.,weighXmin=None):       # 设置束流边界　　　Ｘ
        if weighXmin==None:
            weighXmin=-weighXmax
        self.weighXmax=weighXmax
        self.weighXmin=weighXmin

        self.weighDeltaX=(self.weighXmax-self.weighXmin)/(np.float(self.weighGridX+1))

    def SetWeighBoundaryY(self,weighYmax=20.,weighYmin=None):       # 设置束流边界　　　Ｙ
        if weighYmin==None:
            weighYmin=-weighYmax
        self.weighYmax=weighYmax
        self.weighYmin=weighYmin

        self.weighDeltaY=(self.weighYmax-self.weighYmin)/(np.float(self.weighGridY+1))


    def SetWeighBoundaryZ(self,weighZmax=np.nan,weighZmin=np.nan):       # 求解束流边界　　　Ｚ
        if np.isnan(weighZmax) and np.isnan(weighZmin):
            zc=self.GetStatBeamMean('z')   # c: certer
            pzc=self.GetStatBeamMean('pz')   # c: certer

            betaC=self.FuncP2BetaC(pzc)   # C: light speed
            betaLambda=betaC*self.weighWave

            self.weighZmax=zc+betaLambda/2.
            self.weighZmin=zc-betaLambda/2.
        elif np.isnan(weighZmin):
            self.weighZmax=weighZmax
            self.weighZmin=-weighZmax
        else:
            self.weighZmax=weighZmax
            self.weighZmin=weighZmin
       
        self.weighDeltaZ=(self.weighZmax-self.weighZmin)/(np.float(self.weighGridZ+1))


    def WeighBeam(self,x,y,z,q):
        xMin=self.weighXmin
        yMin=self.weighYmin
        zMin=self.weighZmin

        dx=self.weighDeltaX
        dy=self.weighDeltaY
        dz=self.weighDeltaZ

        xRelative=x-xMin
        yRelative=y-yMin
        zRelative=z-zMin

        xI1=xRelative//dx
        xI2=xI1+1

        yI1=yRelative//dy
        yI2=yI1+1
        
        zI1=zRelative//dz
        zI2=zI1+1

        xF2=(xRelative % dx)/dx
        xF1=1.-xF2

        yF2=(yRelative % dy)/dy
        yF1=1.-yF2

        zF2=(zRelative % dz)/dz
        zF1=1.-zF2

        # print(np.min(xF2),np.min(yF2),np.min(zF2))
        # print('='*20)

        # row 111   顺序为：　row z-y-x

        row111=np.int32(xI1)
        #col111=np.int32(zI1*self.weighGridY+yI1)
        col111=np.int32(yI1*(self.weighGridZ+2)+zI1)
        data111=q*zF1*yF1*xF1

        row112=np.int32(xI2)
        #col112=np.int32(zI1*self.weighGridY+yI1)
        col112=np.int32(yI1*(self.weighGridZ+2)+zI1)
        data112=q*zF1*yF1*xF2

        row121=np.int32(xI1)
        #col121=np.int32(zI1*self.weighGridY+yI2)
        col121=np.int32(yI1*(self.weighGridZ+2)+zI2)
        data121=q*zF1*yF2*xF1

        row122=np.int32(xI2)
        #col122=np.int32(zI1*self.weighGridY+yI2)
        col122=np.int32(yI1*(self.weighGridZ+2)+zI2)
        data122=q*zF1*yF2*xF2

        row211=np.int32(xI1)
        #col211=np.int32(zI2*self.weighGridY+yI1)
        col211=np.int32(yI2*(self.weighGridZ+2)+zI1)
        data211=q*zF2*yF1*xF1

        row212=np.int32(xI2)
        #col212=np.int32(zI2*self.weighGridY+yI1)
        col212=np.int32(yI2*(self.weighGridZ+2)+zI1)
        data212=q*zF2*yF1*xF2

        row221=np.int32(xI1)
        #col221=np.int32(zI2*self.weighGridY+yI2)
        col221=np.int32(yI2*(self.weighGridZ+2)+zI2)
        data221=q*zF2*yF2*xF1

        row222=np.int32(xI2)
        #col222=np.int32(zI2*self.weighGridY+yI2)
        col222=np.int32(yI2*(self.weighGridZ+2)+zI2)
        data222=q*zF2*yF2*xF2


        row=np.hstack((row111,row112,row121,row122,row211,row212,row221,row222))
        col=np.hstack((col111,col112,col121,col122,col211,col212,col221,col222))
        data=np.hstack((data111,data112,data121,data122,data211,data212,data221,data222))

        data/=(self.weighDeltaX*self.weighDeltaY*self.weighDeltaZrelative*1e-9)

        #print(np.min(xF2),1.-np.min(xF2),np.min(np.min(xF2)),'hhh')
        #print(np.max(xF2))

        return [row,col,data]





    def BeamWeigh(self,weighPart=None):       #　束流称重　主函数，有输入输出接口
        if weighPart==None:
            weighPartX,weighPartY,weighPartZ,weighPartQ,weighPartLoss=self.x,self.y,self.z,self.q,self.loss
        else:
            weighPartX,weighPartY,weighPartZ,weighPartQ,weighPartLoss=weighPart[0],weighPart[1],weighPart[2],weighPart[3],weighPart[4]


        
        # print(weighPartQ,'aaa')

        indexUse=np.isnan(weighPartLoss)

        weighX=weighPartX[indexUse]
        weighY=weighPartY[indexUse]
        weighZ=weighPartZ[indexUse]
        weighQ=weighPartQ[indexUse]



        [mpRow,mpCol,mpData]=self.WeighBeam(weighX,weighY,weighZ,weighQ)
        mpWeighGridSparse=[mpRow,mpCol,mpData]

        return mpWeighGridSparse




    #####################################################
    ##  Check Beam Loss
    #####################################################

    def BeamLoss(self):       # 判断束流丢失情况
        self.LossBeamX()
        self.LossBeamY()
        self.LossBeamZ()

    def LossBeamX(self):    # 丢在ｘ方向的是正数，数值是丢失位置ｚ。
        indexLoss=(np.isnan(self.loss)) * ( (self.x>self.weighXmax) + (self.x<self.weighXmin) )
        self.loss[indexLoss]=self.z[indexLoss]
        
    def LossBeamY(self):    # 丢在y方向的是负数，数值是丢失位置ｚ。
        indexLoss=(np.isnan(self.loss)) * ( (self.y>self.weighYmax) + (self.y<self.weighYmin) )
        self.loss[indexLoss]=self.z[indexLoss]


    def LossBeamZ(self):    # z方向不丢失，循环截取在一个周期中
 
        betaLambda=self.weighZmax-self.weighZmin

        while True:
            indexLarger=(np.isnan(self.loss)) * (self.z>self.weighZmax)
            if ~(indexLarger==1).any():
                break
            self.z[indexLarger]-=betaLambda

            
        while True:
            indexLess=(np.isnan(self.loss)) * (self.z<self.weighZmin)
            if ~(indexLess==1).any():
                break
            self.z[indexLess]+=betaLambda






    








    ###########################################################
    ##  基本函数　　basic function
    ###########################################################
    def FuncP2GammaC(self,p):
        return np.sqrt(1.+p**2)
    def FuncP2BetaC(self,p):
        gammaC=self.FuncP2GammaC(p)
        return np.sqrt(1.-1./gammaC**2)


    



    ####################################################################
    ##   FFT
    ####################################################################
    def BeamFFTxy(self,mpWeighGridOnZ):
        mpWeighGridOnZ=dstn(mpWeighGridOnZ,axes=[0,1],type=1,overwrite_x=True)
        return mpWeighGridOnZ

    def BeamFFTxyInv(self,mpWeighGridOnZ):
        mpWeighGridOnZ=idstn(mpWeighGridOnZ,axes=[0,1],type=1,overwrite_x=True)
        mpWeighGridOnZ/=((2.*(self.weighDeltaX+1.))*(2.*(self.weighDeltaY+1.)))
        return mpWeighGridOnZ

    '''     # 毛毛多，比较扎手
    def BeamFFTz(self,mp_weighGrid_fftK2):
        mpWeighGridOnX=mp_weighGrid_fftK2[0]
        fftK2=mp_weighGrid_fftK2[1]

        mpWeighGridOnX=rfft(mpWeighGridOnX,axis=-1,overwrite_x=True)

        mpWeighGridOnX=mpWeighGridOnX/fftK2

        mpWeighGridOnX=irfft(mpWeighGridOnX,axis=-1,overwrite_x=True)

        return mpWeighGridOnX
    '''
    '''   # 可能是错的
    def BeamFFTz(self,mp_weighGrid_fftK2):
        mpWeighGridOnX=mp_weighGrid_fftK2[0]
        fftK2=mp_weighGrid_fftK2[1]

        mpWeighGridOnXs=dst(mpWeighGridOnX,axis=-1,overwrite_x=True)
        mpWeighGridOnXc=dct(mpWeighGridOnX,axis=-1,overwrite_x=True)

        mpWeighGridOnXs=mpWeighGridOnXs/fftK2
        mpWeighGridOnXc=mpWeighGridOnXc/fftK2

        mpWeighGridOnXs=idst(mpWeighGridOnXs,type=1,axis=-1,overwrite_x=True)
        mpWeighGridOnXc=idct(mpWeighGridOnXc,type=1,axis=-1,overwrite_x=True)

        mpWeighGridOnXs/=2.*(self.weighGridZ+1.)
        mpWeighGridOnXc/=2.*self.weighGridZ

        mpWeighGridOnX=mpWeighGridOnXs+mpWeighGridOnXc

        return mpWeighGridOnX
    '''
    #'''
    def BeamFFTz(self,mp_weighGrid_fftK2):
        mpWeighGridOnX=mp_weighGrid_fftK2[0]
        fftK2=mp_weighGrid_fftK2[1]

        mpWeighGridOnX=fft(mpWeighGridOnX,axis=-1,overwrite_x=True)

        mpWeighGridOnX=mpWeighGridOnX/fftK2

        mpWeighGridOnX=np.real(ifft(mpWeighGridOnX,axis=-1,overwrite_x=True))

        return mpWeighGridOnX
    #'''


    def FFTEigenInitialize(self):
   
        pzc=self.GetStatBeamMean('pz')   # c: certer
        gammaCc=self.FuncP2GammaC(pzc)

        self.weighDeltaZrelative=self.weighDeltaZ*gammaCc

        k2x=-(2.*np.sin((np.arange(self.weighGridX)+1.)/(1.+self.weighGridX)/2.*np.pi)/self.weighDeltaX)**2
        self.fftK2X=np.repeat(np.repeat(k2x[:,np.newaxis],self.weighGridY,axis=1)[:,:,np.newaxis],self.weighGridZ,axis=2)
        
        k2y=-(2.*np.sin((np.arange(self.weighGridY)+1.)/(1.+self.weighGridY)/2.*np.pi)/self.weighDeltaY)**2
        self.fftK2Y=np.repeat(np.repeat(k2y[np.newaxis,:],self.weighGridX,axis=0)[:,:,np.newaxis],self.weighGridZ,axis=2)

        k2z=-(2.*np.sin(np.arange(self.weighGridZ)/self.weighGridZ*np.pi))**2/self.weighDeltaZrelative**2
        self.fftK2Z=np.repeat(np.repeat(k2z[np.newaxis,:],self.weighGridX,axis=0)[:,np.newaxis,:],self.weighGridY,axis=1)


    def FFTEigens(self):

        pzc=self.GetStatBeamMean('pz')   # c: certer
        gammaCc=self.FuncP2GammaC(pzc)

        weighDeltaZrelative=self.weighDeltaZ*gammaCc
        self.fftK2Z*=(weighDeltaZrelative**2/self.weighDeltaZrelative**2)
        self.weighDeltaZrelative=weighDeltaZrelative

        self.fftK2=self.fftK2X+self.fftK2Y+self.fftK2Z

        #print(np.shape(self.fftK2))
   
        
    














if __name__=="__main__":
    myBeam=Beam()



    myBeam.SetNumPart(12345)
    myBeam.SetAMU(938.272)
    myBeam.SetGenBeamMass(1)
    myBeam.SetGenBeamCharge(1)
    myBeam.SetGenBeamDist('W6d')
    myBeam.SetGenEs(0.035)
    myBeam.SetGenTwissAlphaX(-1)
    myBeam.SetGenTwissBetaX(1)
    myBeam.SetGenTwissAlphaY(1)
    myBeam.SetGenTwissBetaY(1)
    myBeam.SetGenTwissAlphaZ(0)
    myBeam.SetGenTwissBetaZ(0.05)
    myBeam.SetGenEmitNormX(0.22)
    myBeam.SetGenEmitNormY(0.22)
    myBeam.SetGenEmitNormZ(0.025)

    myBeam.SetGenXs()
    myBeam.SetGenXPs()
    myBeam.SetGenYs()
    myBeam.SetGenYPs()
    myBeam.SetGenZs()
    myBeam.SetGenZPs()

    myNumCPU=10
    myBeam.SetParaNumCPU(myNumCPU)

    mpNumPart=myBeam.ParaAllocationBeamGen()
    with mp.Pool(myNumCPU) as p:
        mpPart=p.map(myBeam.BeamGen,mpNumPart)
    part=np.hstack(mpPart)
    myBeam.BeamSet(part)
    myBeam.BeamTrans()

    myBeam.SetWeighGrid3D(5,6,7)
    myBeam.SetWeighFreq(162.5)
    myBeam.SetWeighBoundaryX(50)
    myBeam.SetWeighBoundaryY(50)
    myBeam.SetWeighBoundaryZ()

    myBeam.BeamLoss()

    myBeam.FFTEigenInitialize()


    mpWeighPart=myBeam.ParaAllocationBeamWeigh()
    with mp.Pool(myNumCPU) as p:
        mpWeighGridSparse=p.map(myBeam.BeamWeigh,mpWeighPart)
    weighGridSparse=np.hstack(mpWeighGridSparse)
    weighGrid=sparse.coo_matrix((weighGridSparse[2],(weighGridSparse[0],weighGridSparse[1])),shape=(myBeam.weighGridX+2,(myBeam.weighGridY+2)*(myBeam.weighGridZ+2))).toarray().reshape((myBeam.weighGridX+2,myBeam.weighGridY+2,myBeam.weighGridZ+2))[1:myBeam.weighGridX+1,1:myBeam.weighGridY+1,1:myBeam.weighGridZ+1]
    myBeam.BeamGridSet(weighGrid)

    weighGridPart=weighGrid  #

    mpWeighGridOnZ=myBeam.ParaAllocationBeamGridOnZ()
    with mp.Pool(myNumCPU) as p:
        weighGridOnZ=p.map(myBeam.BeamFFTxy,mpWeighGridOnZ)
    weighGrid=np.dstack(weighGridOnZ)
    myBeam.BeamGridSet(weighGrid)

    weighGridXY=weighGrid  #


    myBeam.FFTEigens()

    mp_weighGrid_fftK2=myBeam.ParaAllocation_BeamGrid_FFTEigen_OnX()
    with mp.Pool(myNumCPU) as p:
        weighGridOnX=p.map(myBeam.BeamFFTz,mp_weighGrid_fftK2)

    weighGrid=np.vstack(weighGridOnX)
    myBeam.BeamGridSet(weighGrid)
    
    mpWeighGridOnZ=myBeam.ParaAllocationBeamGridOnZ()
    with mp.Pool(myNumCPU) as p:
        weighGridOnZ=p.map(myBeam.BeamFFTxyInv,mpWeighGridOnZ)
    weighGridU=np.dstack(weighGridOnZ)

    '''
    fig=plt.figure('x-y')
    ax=Axes3D(fig)
    for i in range(myBeam.weighGridZ):
        ax.clear()
        ax.set_title(i)

        X = np.linspace(myBeam.weighXmin,myBeam.weighXmax, myBeam.weighGridX)
        Y = np.linspace(myBeam.weighYmin,myBeam.weighYmax, myBeam.weighGridY)
        Y,X = np.meshgrid(Y,X)
        Z=weighGridU[:,:,i]
        
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
            linewidth=0, antialiased=False)

        plt.pause(0.01)
    '''
    
    fig=plt.figure('x-z')
    ax=Axes3D(fig)
    for i in range(myBeam.weighGridY):
        ax.clear()
        ax.set_title(i)

        X = np.linspace(myBeam.weighXmin,myBeam.weighXmax, myBeam.weighGridX)
        Z = np.linspace(myBeam.weighZmin,myBeam.weighZmax, myBeam.weighGridZ)
        Z,X = np.meshgrid(Z,X)
        Y=weighGridU[:,i,:]
        
        surf = ax.plot_surface(X, Z,Y, rstride=1, cstride=1, cmap=cm.jet,
            linewidth=0, antialiased=False)

        plt.pause(0.01)
    




































print('-'*50)
print('END')
