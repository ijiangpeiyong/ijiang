# Author : Peiyong Jiang
# jiangpeiyong@126.com

import numpy as np

class Beam():
    def __init__(self):
        pass

    def SetNumPart(self,numPart):
        self.numPart=np.int32(numPart)
    
    def SetEnergy(self,energyMeV=1.5):
        self.energy=energyMeV

    def SetFrequency(self,freqMHz):
        self.freqMHz=freqMHz

    def SetEmitNormX(self,emitNormX=0.22):
        self.emitNormX=emitNormX
    def SetEmitNormY(self,emitNormY=0.22):
        self.emitNormY=emitNormY
    def SetEmitNormZ(self,emitNormZ=0.25):
        self.emitNormZ=emitNormZ

    def SetTwissAlphaX(self,twissAlphaX=0.):
        self.twissAlphaX=twissAlphaX
    def SetTwissAlphaY(self,twissAlphaY=0.):
        self.twissAlphaY=twissAlphaY
    def SetTwissAlphaZ(self,twissAlphaZ=0.):
        self.twissAlphaZ=twissAlphaZ

    def SetTwissBetaX(self,twissBetaX=1.):
        self.twissBetaX=twissBetaX
    def SetTwissBetaY(self,twissBetaY=1.):
        self.twissBetaY=twissBetaY
    def SetTwissBetaZ(self,twissBetaZ=1.):
        self.twissBetaZ=twissBetaZ

    def SetBeamLength(self,beamLength=1.):
        self.beamLength=beamLength    
    def SetBeamDpp(self,beamDpp=0.01):
        self.beamDpp=beamDpp
    
    def CalTwissGammaX(self):
        self.twissGammaX=(1.+self.twissAlphaX**2)/self.twissBetaX
    def CalTwissGammaY(self):
        self.twissGammaY=(1.+self.twissAlphaY**2)/self.twissBetaY
    def CalTwissGammaZ(self):
        self.twissGammaZ=(1.+self.twissAlphaZ**2)/self.twissBetaZ

    def SetBeamDist(self,beamDist):
        self.beamDist=beamDist

    #--------------------------------------------
    def G4d(self):
        mean=[0,0,0,0]
        cov=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        self.x, self.xp, self.y ,self.yp = np.random.multivariate_normal(mean, cov, self.numPart).T

    def K4d(self):
        self.W4d()
        r=np.sqrt(self.x**2+self.xp**2+self.y**2+self.yp**2)
        self.x/=r
        self.xp/=r
        self.y/=r
        self.yp/=r

    def W4d(self):
        numPart=np.int32(self.numPart*3.57)
        dataRandom=np.random.random((numPart,4))*2.-1.
        r=dataRandom[:,0]**2+dataRandom[:,1]**2+dataRandom[:,2]**2+dataRandom[:,3]**2
        indexR=r<1.
        dataRandom=dataRandom[indexR,:]
        self.x, self.xp, self.y ,self.yp=dataRandom[0:self.numPart,0],dataRandom[0:self.numPart,1],dataRandom[0:self.numPart,2],dataRandom[0:self.numPart,3]


    def Uz(self):
        self.z=np.random.random((self.numPart))
    
    def Gp(self):
        self.p=np.random.randn((self.numPart))

    #--------------------------------------------
    def G4dUzGp(self):
        self.G4d()
        self.Uz()
        self.Gp()

    def G6d(self):
        mean=[0,0,0,0,0,0]
        cov=[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
        self.x, self.xp, self.y ,self.yp,self.z,self.p = np.random.multivariate_normal(mean, cov, self.numPart).T

    def K4dUzGp(self):
        self.K4d()
        self.Uz()
        self.Gp()

    def W6d(self):
        numPart=np.int32(self.numPart*3.57)
        dataRandom=np.random.random((numPart,6))*2.-1.
        r=dataRandom[:,0]**2+dataRandom[:,1]**2+dataRandom[:,2]**2+dataRandom[:,3]**2+dataRandom[:,4]**2+dataRandom[:,5]**2
        indexR=r<1.
        dataRandom=dataRandom[indexR,:]
        self.x, self.xp, self.y ,self.yp,self.z,self.p=dataRandom[0:self.numPart,0],dataRandom[0:self.numPart,1],dataRandom[0:self.numPart,2],dataRandom[0:self.numPart,3],dataRandom[0:self.numPart,4],dataRandom[0:self.numPart,5]


    def W4dUzGp(self):
        self.W4d()
        self.Uz()
        self.Gp()

    def 


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