import numpy as np



import matplotlib.pyplot as plt
from scipy.special import erfi
from cmath import sqrt

class Halo():


    def __init__(self,k0=1,v=1,aRmsMM=5.,nHalo=2,nRmsP=1.):
        self.k0=k0
        self.k20=k0**2
        self.v=v
        self.k=k0*v
        self.v2=v**2
        self.k2sc=(1-self.v2)*self.k20

        self.a=aRmsMM*1e-3
        self.c2f=self.k2sc*3*self.a**3

        self.rMax=self.a*nHalo

        self.nRmsP=nRmsP
        self.ap=self.a*self.k*self.nRmsP

        self.M=self.rMax*self.ap

        self.E=self.Ueff(self.rMax)*1.0001

    def Ul(self,r):
        ul=1./2.*self.k20*r**2
        return ul

    def Usc(self,r):
        arg=np.sqrt(r**2/(2.*self.a**2))
        cArg=arg*1.j
        usc=self.c2f*(1./self.a)*(np.sqrt(np.pi)/2.)*erfi(cArg)/cArg
        usc=np.real(usc)
        return usc

    def Um(self,r):
        um=self.M**2/(2.*r**2)
        return um

    def Ur(self,r):
        ul=self.Ul(r)
        usc=self.Usc(r)

        ur=ul+usc

        return ur

    def Ueff(self,r):
        ur=self.Ur(r)
        um=self.Um(r)

        ueff=ur+um

        return ueff

    def DiffE(self,r):
        dE=self.E-self.Ueff(r)
        return dE

    def DiffE2(self,r):
        dE=self.DiffE(r)
        if dE<0:
            dE2=np.nan
            return dE2
        dE2=np.sqrt(dE)
        return dE2


    def PlotUr(self):
        rMax=self.rMax*1.2
        r=np.linspace(0.,rMax,1000)
        ur=self.Ur(r)
        plt.figure('ur')
        plt.plot(r,ur)


    def PlotUeff(self):
        rMax=self.rMax*2.5
        r=np.linspace(2,rMax,1000)
        ueff=self.Ueff(r)
        plt.figure('ueff')
        plt.plot(r,ueff)
        plt.grid('on')
        plt.hold('on')
        plt.plot(r,np.ones_like(r)*self.E,'r')
        # print(Ueff)

    def SetInitial(self,r0=0.,phi0=0):
        if r0==0. :
            self.r0=self.rMax*0.99999
        else:
            self.r0=r0
        self.phi0=phi0


    def PhiDot(self,r):
        phiDot=self.M/r**2
        return phiDot


    def Solver(self,tEnd=10,tStep=0.1,TCut=0):
        h=tStep
        tStart=0
        r=self.r0
        phi=0

        tList=[]
        rList=[]
        phiList=[]

        tList.append(tStart)
        rList.append(r)
        phiList.append(phi)

        nCut=0
        while True:

            k1=self.DiffE2(r)
            k2=self.DiffE2(r+h/2*k1)
            k3=self.DiffE2(r+h/2*k2)
            k4=self.DiffE2(r+h*k3)

            rRec=r
            r+=h/6.*(k1+2.*k2+2.*k3+k4)

            if np.isnan(r):
                h*=-1.
                r=rRec
                nCut+=1
                if TCut>0 and nCut>=TCut:
                    break
                continue

            p=self.PhiDot(r)
            phi+=np.abs(h)*p

            tList.append(tStart)
            rList.append(r)
            phiList.append(phi)



            tStart+=np.abs(h)



            if tStart>=tEnd:
                if TCut>0:
                    pass
                else:
                    break
        return tList,rList,phiList


if __name__=='__main__':
    recHalo=np.linspace(0.2,2,3)
    recPhiEnd=[]
    recTEnd=[]

    for iHalo in recHalo:
        print(iHalo)
        halo=Halo(k0=0.5, v=1.,nHalo=2,nRmsP=iHalo)

        halo.SetInitial()
        tList,rList,phiList=halo.Solver(tEnd=10,tStep=1e-4,TCut=5)

        recPhiEnd.append(phiList[-1])
        recTEnd.append(tList[-1])


        plt.figure('phi-r')
        plt.subplot(111,polar=True)
        plt.plot(phiList,rList)
        plt.pause(0.005)



    recPhiEnd=np.array(recPhiEnd)
    recTEnd=np.array(recTEnd)

    plt.figure('nHalo-PhiEnd')
    plt.plot(recHalo,recPhiEnd,'ro')
    plt.hold('on')
    plt.plot(recHalo,recPhiEnd,'b-')

    plt.figure('nHalo-tEnd')
    plt.plot(recHalo,recTEnd,'ro')
    plt.hold('on')
    plt.plot(recHalo,recTEnd,'b-')

    plt.figure('tEnd-phiEnd')
    plt.plot(recTEnd,recPhiEnd,'ro')
    plt.hold('on')
    plt.plot(recTEnd,recPhiEnd,'b-')

    plt.figure('nHalo-phiEnd_tEnd')
    plt.plot(recHalo,recTEnd/recPhiEnd,'ro')
    plt.hold('on')
    plt.plot(recHalo,recTEnd/recPhiEnd,'b-')



    plt.show()

    print('END')


#
