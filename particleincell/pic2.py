# Author : Peiyong Jiang
# jiangpeiyong@126.com

import numpy as np
from scipy import constants as const
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

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

        return([self.x,self.px,self.y,self.py,self.z,self.pz])



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

    def BeamStatRMS(self):    # 统计束流
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

    def GetStatBeamMean(self,coordinate='x'):
        if coordinate=='x':
            return self.x.mean()
        if coordinate=='px':
            return self.px.mean()
        if coordinate=='xp':
            return self.xp.mean()
        if coordinate=='y':
            return self.y.mean()
        if coordinate=='py':
            return self.py.mean()
        if coordinate=='yp':
            return self.yp.mean()
        if coordinate=='z':
            return self.z.mean()
        if coordinate=='pz':
            return self.pz.mean()  
        if coordinate=='zp':
            return self.zp.mean()  


    def GetStatBeamStd(self,coordinate='x'):        
        if coordinate=='x':
            return self.x.std()
        if coordinate=='px':
            return self.px.std()
        if coordinate=='xp':
            return self.xp.std()            
        if coordinate=='y':
            return self.y.std()
        if coordinate=='py':
            return self.py.std()
        if coordinate=='yp':
            return self.yp.std()
        if coordinate=='z':
            return self.z.std()
        if coordinate=='pz':
            return self.pz.std()      
        if coordinate=='zp':
            return self.zp.std()           
            
    def GetStatBeamVar(self,coordinate='x'):        
        if coordinate=='x':
            return self.x.var()
        if coordinate=='px':
            return self.px.var()
        if coordinate=='xp':
            return self.xp.var()
        if coordinate=='y':
            return self.y.var()
        if coordinate=='py':
            return self.py.var()
        if coordinate=='yp':
            return self.yp.var()
        if coordinate=='z':
            return self.z.var()
        if coordinate=='pz':
            return self.pz.var()   
        if coordinate=='zp':
            return self.zp.var()   


    def GetStatBeamCov(self,coordinates=['x','xp']):   
        x=[] 
        for iCoor in coordinates:
            x.append(eval('self.'+iCoor))
        return np.cov(x)

    def GetStatBeamEmitNatureRMS(self,coordinate='x'):
        if coordinate=='x':
            cov=self.GetStatBeamCov(['x','xp'])
            return np.sqrt(np.linalg.det(cov))
        if coordinate=='y':
            cov=self.GetStatBeamCov(['y','yp'])
            return np.sqrt(np.linalg.det(cov))
        if coordinate=='z':
            cov=self.GetStatBeamCov(['y','yp'])
            return np.sqrt(np.linalg.det(cov))    

    def GetStatBeamEmitNormRMS(self,coordinate='x'):
        emitNature=self.GetStatBeamEmitNatureRMS(coordinate)
        pz0=self.GetStatBeamMean('pz')
        gammaC=np.sqrt(1.+pz0**2)
        if (coordinate=='x') or (coordinate=='y'):
            emitNorm= emitNature*pz0
        if coordinate=='z':
            emitNorm= emitNature*pz0*gammaC**2
        return emitNorm
    
    def GetStatBeamBeta(self,coordinate='x'):
        if coordinate=='x':
            cov=self.GetStatBeamCov(['x','xp'])
        if coordinate=='y':
            cov=self.GetStatBeamCov(['y','yp'])
        if coordinate=='z':
            cov=self.GetStatBeamCov(['y','yp'])

        emitNature=np.sqrt(np.linalg.det(cov))
        beta=cov[0,0]/emitNature

        return beta         

    def GetStatBeamAlpha(self,coordinate='x'):
        if coordinate=='x':
            cov=self.GetStatBeamCov(['x','xp'])
        if coordinate=='y':
            cov=self.GetStatBeamCov(['y','yp'])
        if coordinate=='z':
            cov=self.GetStatBeamCov(['y','yp'])

        emitNature=np.sqrt(np.linalg.det(cov))
        alpha=-cov[0,1]/emitNature

        return alpha         

    def GetStatBeamGamma(self,coordinate='x'):
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
    def TransBeamPX2XP(self,x,px,y,py,z,pz):
        pz0=pz.mean()
        zp=(pz-pz0)/pz0*1000.
        xp=px/pz*1000.
        yp=py/pz*1000.
        return x,xp,y,yp,z,zp

    def BeamTrans(self):
        self.x,self.xp,self.y,self.yp,self.z,self.zp=self.TransBeamPX2XP(self.x,self.px,self.y,self.py,self.z,self.pz)
        return [self.x,self.xp,self.y,self.yp,self.z,self.zp]



    #########################################################
    ##    并行相关
    #########################################################

    def SetParaNumCPU(self,paraNumCPU=0):
        self.paraNumCPU=np.int32(paraNumCPU)

    def ParaAllocationBeamGen(self):
        if self.numPart % self.paraNumCPU==0:
            mpNumPart=[self.numPart // self.paraNumCPU]*self.paraNumCPU
            return mpNumPart
        numPart=self.numPart//self.paraNumCPU+1
        numPartLast=self.numPart-numPart*(self.paraNumCPU-1)
        
        mpNumPart=[numPart]*(self.paraNumCPU-1)
        mpNumPart.append(numPartLast)
        return mpNumPart



if __name__=="__main__":
    myBeam=Beam()

    ###########################################################################
    '''
    #----- 测试ＧＳ　６Ｄ  束流　生成
    myBeam.SetAMU(938.272)
    myBeam.SetGenBeamDist('G6d')
    myBeam.SetGenEs(0.035)
    myBeam.SetNumPart(1e5)
    myBeam.SetGenTwissAlphaX(-1)
    myBeam.SetGenTwissBetaX(1)
    myBeam.SetGenTwissAlphaY(1)
    myBeam.SetGenTwissBetaY(1)
    myBeam.SetGenTwissAlphaZ(0)
    myBeam.SetGenTwissBetaZ(1)
    myBeam.SetGenEmitNormX(0.22)
    myBeam.SetGenEmitNormY(0.22)
    myBeam.SetGenEmitNormZ(0.25)

    myBeam.SetGenXs()
    myBeam.SetGenXPs()
    myBeam.SetGenYs()
    myBeam.SetGenYPs()
    myBeam.SetGenZs()
    myBeam.SetGenZPs()

    myBeam.BeamGen()
    myBeam.BeamTrans()

    plt.figure('gs-6d')
    plt.subplot(221)
    plt.plot(myBeam.genX,myBeam.genXP,'.')
    plt.axis('equal')
    plt.grid('on')
    plt.subplot(222)
    plt.plot(myBeam.genY,myBeam.genYP,'.')
    plt.grid('on')
    plt.axis('equal')
    plt.subplot(223)
    plt.plot(myBeam.genZ,myBeam.genZP,'.')
    plt.grid('on')
    plt.axis('equal')
    plt.subplot(224)
    plt.plot(myBeam.genX,myBeam.genY,'.')
    plt.grid('on')
    plt.axis('equal')

    plt.figure('gs　－　inner')
    plt.subplot(221)
    plt.plot(myBeam.x,myBeam.px,'.')
    #plt.axis('equal')
    plt.grid('on')
    plt.subplot(222)
    plt.plot(myBeam.y,myBeam.py,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(223)
    plt.plot(myBeam.z,myBeam.pz,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(224)
    plt.plot(myBeam.x,myBeam.y,'.')
    plt.grid('on')
    plt.axis('equal')

    #plt.show()

    #----- 测试ＧＳ　  束流　统计
    myBeam.SetStatBeamDist('x')
    myBeam.SetStatBeamX(myBeam.genX,myBeam.genXP)
    print(myBeam.BeamStatRMS())
    
    myBeam.SetStatBeamDist('y')
    myBeam.SetStatBeamY(myBeam.genY,myBeam.genYP)
    print(myBeam.BeamStatRMS())

    myBeam.SetStatBeamDist('z')
    myBeam.SetStatBeamZ(myBeam.genZ,myBeam.genZP)
    print(myBeam.BeamStatRMS())

    print('-'*20)
    myBeam.SetStatBeamDist('xy')
    myBeam.SetStatBeamXY(myBeam.genX,myBeam.genXP,myBeam.genY,myBeam.genYP)
    print(myBeam.BeamStatRMS())

    print('-'*20)
    myBeam.SetStatBeamDist('xyz')
    myBeam.SetStatBeamXYZ(myBeam.genX,myBeam.genXP,myBeam.genY,myBeam.genYP,myBeam.genZ,myBeam.genZP)
    print(myBeam.BeamStatRMS())   
    
    print('#'*50)

    print(myBeam.GetStatBeamMean('xp'))
    print(myBeam.GetStatBeamStd('xp'))
    print(myBeam.GetStatBeamVar('xp'))
    print(myBeam.GetStatBeamCov(['x','xp']))

    print(myBeam.GetStatBeamEmitNatureRMS('x'))
    print(myBeam.GetStatBeamEmitNormRMS('x'))
    print(myBeam.GetStatBeamBeta('x'))
    print(myBeam.GetStatBeamAlpha('x'))
    print(myBeam.GetStatBeamGamma('x'))



    '''

    #################################################################

    '''
    #----- 测试WB　６Ｄ  束流　生成
    myBeam.SetAMU(938.272)
    myBeam.SetGenBeamDist('W6d')
    myBeam.SetGenEs(0.035)
    myBeam.SetNumPart(1e5)
    myBeam.SetGenTwissAlphaX(-1)
    myBeam.SetGenTwissBetaX(1)
    myBeam.SetGenTwissAlphaY(1)
    myBeam.SetGenTwissBetaY(1)
    myBeam.SetGenTwissAlphaZ(0)
    myBeam.SetGenTwissBetaZ(1)
    myBeam.SetGenEmitNormX(0.22)
    myBeam.SetGenEmitNormY(0.22)
    myBeam.SetGenEmitNormZ(0.25)


    myBeam.SetGenXs()
    myBeam.SetGenXPs()
    myBeam.SetGenYs()
    myBeam.SetGenYPs()
    myBeam.SetGenZs()
    myBeam.SetGenZPs()

    myBeam.BeamGen()
    myBeam.BeamTrans()

    plt.figure('wb-6d')
    plt.subplot(221)
    plt.plot(myBeam.genX,myBeam.genXP,'.')
    plt.axis('equal')
    plt.grid('on')
    plt.subplot(222)
    plt.plot(myBeam.genY,myBeam.genYP,'.')
    plt.grid('on')
    plt.axis('equal')
    plt.subplot(223)
    plt.plot(myBeam.genZ,myBeam.genZP,'.')
    plt.grid('on')
    plt.axis('equal')
    plt.subplot(224)
    plt.plot(myBeam.genX,myBeam.genY,'.')
    plt.grid('on')
    plt.axis('equal')


    plt.figure('wb　－　inner')
    plt.subplot(221)
    plt.plot(myBeam.x,myBeam.px,'.')
    #plt.axis('equal')
    plt.grid('on')
    plt.subplot(222)
    plt.plot(myBeam.y,myBeam.py,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(223)
    plt.plot(myBeam.z,myBeam.pz,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(224)
    plt.plot(myBeam.x,myBeam.y,'.')
    plt.grid('on')
    plt.axis('equal')


    plt.show()
    '''
    ##########################################################################
    
    '''
    #----- 测试　G4dUzGdpp  束流　生成
    myBeam.SetAMU(938.272)
    myBeam.SetGenBeamDist('G4dUzGdpp')
    myBeam.SetGenEs(0.035)
    myBeam.SetNumPart(1e5)
    myBeam.SetGenTwissAlphaX(-1)
    myBeam.SetGenTwissBetaX(1)
    myBeam.SetGenTwissAlphaY(1)
    myBeam.SetGenTwissBetaY(1)
    myBeam.SetGenBeamLength(360)
    myBeam.SetGenBeamDpp(0.01)
    myBeam.SetGenEmitNormX(0.22)
    myBeam.SetGenEmitNormY(0.22)

    myBeam.SetGenFreq(162.5)

    myBeam.SetGenXs()
    myBeam.SetGenXPs()
    myBeam.SetGenYs()
    myBeam.SetGenYPs()
    myBeam.SetGenZs()
    myBeam.SetGenDPPs()
    myBeam.SetGenFreq(162.5)

    myBeam.SetGenXs()
    myBeam.SetGenXPs()
    myBeam.SetGenYs()
    myBeam.SetGenYPs()
    myBeam.SetGenZs()
    myBeam.SetGenDPPs()


    myBeam.BeamGen()
    myBeam.BeamTrans()

    plt.figure('G4dUzGdpp　－　original')
    plt.subplot(221)
    plt.plot(myBeam.genX,myBeam.genXP,'.')
    plt.axis('equal')
    plt.grid('on')
    plt.subplot(222)
    plt.plot(myBeam.genY,myBeam.genYP,'.')
    plt.grid('on')
    plt.axis('equal')
    plt.subplot(223)
    plt.plot(myBeam.genZ,myBeam.genDpp,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(224)
    plt.plot(myBeam.genX,myBeam.genY,'.')
    plt.grid('on')
    plt.axis('equal')

    plt.figure('G4dUzGdpp　－　inner')
    plt.subplot(221)
    plt.plot(myBeam.x,myBeam.px,'.')
    #plt.axis('equal')
    plt.grid('on')
    plt.subplot(222)
    plt.plot(myBeam.y,myBeam.py,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(223)
    plt.plot(myBeam.z,myBeam.pz,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(224)
    plt.plot(myBeam.x,myBeam.y,'.')
    plt.grid('on')
    plt.axis('equal')



    plt.show()
    '''

    ######################################################################
    '''
    #----- 测试　K4dUzGdpp  束流　生成
    myBeam.SetAMU(938.272)
    myBeam.SetGenBeamDist('K4dUzGdpp')
    myBeam.SetGenEs(0.035)
    myBeam.SetNumPart(1e5)
    myBeam.SetGenTwissAlphaX(-1)
    myBeam.SetGenTwissBetaX(1)
    myBeam.SetGenTwissAlphaY(1)
    myBeam.SetGenTwissBetaY(1)
    myBeam.SetGenBeamLength(360)
    myBeam.SetGenBeamDpp(0.01)
    myBeam.SetGenEmitNormX(0.22)
    myBeam.SetGenEmitNormY(0.22)

    myBeam.SetGenFreq(162.5)

    myBeam.SetGenXs()
    myBeam.SetGenXPs()
    myBeam.SetGenYs()
    myBeam.SetGenYPs()
    myBeam.SetGenZs()
    myBeam.SetGenDPPs()


    myBeam.BeamGen()
    myBeam.BeamTrans()

    plt.figure('K4dUzGdpp')
    plt.subplot(221)
    plt.plot(myBeam.genX,myBeam.genXP,'.')
    plt.axis('equal')
    plt.grid('on')
    plt.subplot(222)
    plt.plot(myBeam.genY,myBeam.genYP,'.')
    plt.grid('on')
    plt.axis('equal')
    plt.subplot(223)
    plt.plot(myBeam.genZ,myBeam.genDpp,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(224)
    plt.plot(myBeam.genX,myBeam.genY,'.')
    plt.grid('on')
    plt.axis('equal')

    plt.figure('K4dUzGdpp　－　inner')
    plt.subplot(221)
    plt.plot(myBeam.x,myBeam.px,'.')
    #plt.axis('equal')
    plt.grid('on')
    plt.subplot(222)
    plt.plot(myBeam.y,myBeam.py,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(223)
    plt.plot(myBeam.z,myBeam.pz,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(224)
    plt.plot(myBeam.x,myBeam.y,'.')
    plt.grid('on')
    plt.axis('equal')




    plt.show()
    '''
    #################################################

    '''
    #----- 测试　W4dUzGdpp  束流　生成
    myBeam.SetAMU(938.272)
    myBeam.SetGenBeamDist('W4dUzGdpp')
    myBeam.SetGenEs(0.035)
    myBeam.SetNumPart(1e5)
    myBeam.SetGenTwissAlphaX(-1)
    myBeam.SetGenTwissBetaX(1)
    myBeam.SetGenTwissAlphaY(1)
    myBeam.SetGenTwissBetaY(1)
    myBeam.SetGenBeamLength(360)
    myBeam.SetGenBeamDpp(0.01)
    myBeam.SetGenEmitNormX(0.22)
    myBeam.SetGenEmitNormY(0.22)

    myBeam.SetGenFreq(162.5)

    myBeam.SetGenXs()
    myBeam.SetGenXPs()
    myBeam.SetGenYs()
    myBeam.SetGenYPs()
    myBeam.SetGenZs()
    myBeam.SetGenDPPs(0)


    myBeam.BeamGen()
    myBeam.BeamTrans()

    plt.figure('W4dUzGdpp')
    plt.subplot(221)
    plt.plot(myBeam.genX,myBeam.genXP,'.')
    plt.axis('equal')
    plt.grid('on')
    plt.subplot(222)
    plt.plot(myBeam.genY,myBeam.genYP,'.')
    plt.grid('on')
    plt.axis('equal')
    plt.subplot(223)
    plt.plot(myBeam.genZ,myBeam.genDpp,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(224)
    plt.plot(myBeam.genX,myBeam.genY,'.')
    plt.grid('on')
    plt.axis('equal')


    plt.figure('W4dUzGdpp　－　inner')
    plt.subplot(221)
    plt.plot(myBeam.x,myBeam.px,'.')
    #plt.axis('equal')
    plt.grid('on')
    plt.subplot(222)
    plt.plot(myBeam.y,myBeam.py,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(223)
    plt.plot(myBeam.z,myBeam.pz,'.')
    plt.grid('on')
    #plt.axis('equal')
    plt.subplot(224)
    plt.plot(myBeam.x,myBeam.y,'.')
    plt.grid('on')
    plt.axis('equal')


    plt.show()
    '''

    ############################################
    ##  测试并行生成粒子　　　mp
    ############################################

    ###---------测试并行用时
    '''  
    myBeam.SetAMU(938.272)
    myBeam.SetGenBeamDist('G6d')
    myBeam.SetGenEs(0.035)
    myBeam.SetNumPart(1e7)
    myBeam.SetGenTwissAlphaX(-1)
    myBeam.SetGenTwissBetaX(1)
    myBeam.SetGenTwissAlphaY(1)
    myBeam.SetGenTwissBetaY(1)
    myBeam.SetGenTwissAlphaZ(0)
    myBeam.SetGenTwissBetaZ(1)
    myBeam.SetGenEmitNormX(0.22)
    myBeam.SetGenEmitNormY(0.22)
    myBeam.SetGenEmitNormZ(0.25)

    myBeam.SetGenXs()
    myBeam.SetGenXPs()
    myBeam.SetGenYs()
    myBeam.SetGenYPs()
    myBeam.SetGenZs()
    myBeam.SetGenZPs()

    tocList=[]
    for myNumCPU in range(1,25):
        tic=time.time()
        #myNumCPU=1
        myBeam.SetParaNumCPU(myNumCPU)
        mpNumPart=myBeam.ParaAllocationBeamGen()
        #print(mpNumPart)
        #print(np.sum(mpNumPart))
        with mp.Pool(myNumCPU) as p:
            mpPart=p.map(myBeam.BeamGen,mpNumPart)
            print(np.shape(mpPart))
        part=np.hstack(mpPart)

        toc=time.time()-tic
        tocList.append(toc)
    
    plt.figure('toc 4 gen particles')
    plt.plot(tocList)
    plt.show()
    '''

    ##--------------------------------------------

    myBeam.SetAMU(938.272)
    myBeam.SetGenBeamDist('G6d')
    myBeam.SetGenEs(0.035)
    myBeam.SetNumPart(1e7)
    myBeam.SetGenTwissAlphaX(-1)
    myBeam.SetGenTwissBetaX(1)
    myBeam.SetGenTwissAlphaY(1)
    myBeam.SetGenTwissBetaY(1)
    myBeam.SetGenTwissAlphaZ(0)
    myBeam.SetGenTwissBetaZ(1)
    myBeam.SetGenEmitNormX(0.22)
    myBeam.SetGenEmitNormY(0.22)
    myBeam.SetGenEmitNormZ(0.25)

    myBeam.SetGenXs()
    myBeam.SetGenXPs()
    myBeam.SetGenYs()
    myBeam.SetGenYPs()
    myBeam.SetGenZs()
    myBeam.SetGenZPs()

    myNumCPU=1
    myBeam.SetParaNumCPU(myNumCPU)
    mpNumPart=myBeam.ParaAllocationBeamGen()

    with mp.Pool(myNumCPU) as p:
        mpPart=p.map(myBeam.BeamGen,mpNumPart)
        print(np.shape(mpPart))
    part=np.hstack(mpPart)






























































print('END')



