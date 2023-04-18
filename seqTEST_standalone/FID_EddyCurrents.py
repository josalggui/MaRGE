"""
Created on Tue Nov  9 10:37:29 2021

@author: Teresa
"""
import os
import sys
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************
import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt

def FID_EddyCurrents(
    init_gpa= False,                 
    larmorFreq=3.059, #MHz
    rfExAmp=0.3, 
    rfReAmp=None, 
    rfExPhase = 0,
    rfExTime=41, #us
    rfReTime=None,
    nReadout = 400,
    tAdq = 4, #ms
    tEcho = 20, #ms
    echo = 2, #0 FID, 1 Echo, 2 Both
    tRepetition = 1000,  #ms
    shimming=[135, -40, -5],
    # shimming=[0,0,0],
    nDummyPulses = 0,

    gAxis =1,
    gNsteps =16,
    gRiseTime = 0.5,    #ms
    gAmp = 0.3, # 1 a.u. ---> 10 V
    gFlatTop = 1.0, #ms
    tDelay = 18.0, #ms
    plotSeq = 0):

    tAdq = tAdq * 1e3
    tEcho = tEcho * 1e3
    tRepetition = tRepetition * 1e3
    gRiseTime = gRiseTime * 1e3
    gFlatTop = gFlatTop * 1e3
    tDelay = tDelay * 1e3
    gDuration = gFlatTop + 2*gRiseTime

    #CONSTANTES
    tStart = 20
    txGatePre = 15
    txGatePost = 1
    blkTime = 300  #Tiempo que dejo entre el rfPulse y la lectura. 
    gAmpMax = 1
    nGOptions = 3 
    nReadout = nReadout + 5
    shimming=np.array(shimming)*1e-4
    tDelayShimming = 15*1e3
    
    #CONDICIONES PARAMETROS INICIALES.
    if gAmp > gAmpMax:
        gAmp = gAmpMax 
    if rfReAmp is None:
        rfReAmp = rfExAmp
    if rfReTime is None:
        rfReTime = 2*rfExTime
    
    rfExPhase = rfExPhase*np.pi/180
    rfExAmp = rfExAmp*np.exp(1j*rfExPhase)
    rfRePhase = np.pi/2
    rfReAmp = rfReAmp *np.exp(1j*rfRePhase)
    
    #DEFINICIÓN DE LOS PULSOS
    def rfPulse(tIni, rfAmp, rfDuration):
        txTime = np.array([tIni,tIni+rfDuration])
        txAmp = np.array([rfAmp,0])
        
        txGateTime = np.array([tIni-txGatePre,tIni+rfDuration+txGatePost])
        txGateAmp = np.array([1,0])
        
        expt.add_flodict({
                        'tx0': (txTime, txAmp),
                        'tx_gate': (txGateTime, txGateAmp)})
        
        tFin = tIni+rfDuration
        return tFin
    
    def readoutGate(tIni,tRd):
        rxTime = np.array([tIni, tIni+tRd])
        rxAmp = np.array([1,0])
        
        rxGateTime = rxTime
        rxGateAmp = rxAmp
        expt.add_flodict({
                        'rx0_en': (rxTime, rxAmp),
                        'rx_gate': (rxGateTime, rxGateAmp),
                        })
        
        tFin = tIni+tRd
        return tFin
    
    def gradPulse(tIni, gDuration, gAmp, gAxis):
        if gAxis == 0:
            shim=shimming[0]
        elif gAxis == 1:
            shim=shimming[1]
        elif gAxis == 2:
            shim=shimming[2]
        elif gAxis == 3:
            shim = 0
    
        tRise = np.linspace(tIni, tIni+gRiseTime, gNsteps, endpoint=True)
        aRise = np.linspace(shim, gAmp, gNsteps+1, endpoint=True)
        aRise = np.delete(aRise,0)
        tDown = np.linspace(tIni+gDuration-gRiseTime,tIni+gDuration,gNsteps,endpoint=True)
        aDown = np.linspace(gAmp,shim,gNsteps+1,endpoint=True)
        aDown = np.delete(aDown,0)
        gTime = np.concatenate((tRise,tDown),  axis=0)
        gAmp = np.concatenate((aRise,aDown),  axis=0)
        if gAxis == 0:
                expt.add_flodict({'grad_vx': (gTime,gAmp)})
        elif gAxis == 1:
                expt.add_flodict({'grad_vy': (gTime,gAmp)})
        elif gAxis == 2:
                expt.add_flodict({'grad_vz': (gTime,gAmp)})
        tFin = tIni+gDuration        
        return tFin
    
    def iniSequence(tIni):
            expt.add_flodict({
                    'grad_vx': (np.array([tIni]),np.array([shimming[0]])), 
                    'grad_vy': (np.array([tIni]),np.array([shimming[1]])),  
                    'grad_vz': (np.array([tIni]),np.array([shimming[2]])),
                 })

    def endSequence(tEnd):
        expt.add_flodict({
                'grad_vx': (np.array([tEnd]),np.array([0]) ), 
                'grad_vy': (np.array([tEnd]),np.array([0]) ), 
                'grad_vz': (np.array([tEnd]),np.array([0]) ),
             })
    
    #SECUENCIA
    BW = nReadout/tAdq
    rx_period = 1/BW
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    
    gAmp = np.linspace(-gAmp,gAmp,3,endpoint=True)
    if gAxis == 0:
            gAmp = np.array(gAmp)+shimming[0]
    elif gAxis == 1:
            gAmp = np.array(gAmp)+shimming[1]
    elif gAxis == 2:
            gAmp = np.array(gAmp)+shimming[2]
    iniSequence(10)

    timeSeq = tStart+tDelayShimming #For shimming stabilization
    tDelaySeq = tDelay + gDuration
    for m in range(nDummyPulses):
        timeSeq = rfPulse(timeSeq+tDelaySeq, rfExAmp, rfExTime)
        timeSeq = rfPulse(timeSeq + tEcho, rfReAmp, rfReTime)
        timeSeq = tRepetition * (m + 1) + tStart +tDelayShimming

    for n in range(nGOptions):
        gradPulse(timeSeq, gDuration, gAmp[n], gAxis)
        timeSeq = rfPulse(timeSeq+tDelaySeq,rfExAmp, rfExTime)
        if echo == 0:
            readoutGate(timeSeq+blkTime,tAdq)
        elif echo == 1:
            timeSeq = rfPulse(timeSeq+tEcho,rfReAmp, rfReTime) 
            timeSeq = readoutGate(timeSeq+tEcho-tAdq/2,tAdq)
        elif echo == 2:
            readoutGate(timeSeq+blkTime,tAdq)
            timeSeq = rfPulse(timeSeq+tEcho,rfReAmp, rfReTime) 
            timeSeq = readoutGate(timeSeq+tEcho-tAdq/2,tAdq)
        timeSeq=tRepetition*(nDummyPulses+n+1)+tStart+tDelayShimming
    endSequence(timeSeq-tStart)
    
    #RUN
    if plotSeq==1:                
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        print('Running...')
        rxd, msgs = expt.run()
        expt.__del__()
        print('End')
        data = rxd['rx0']*13.788
        if echo == 2: 
            nGOptions = nGOptions*2
        dataIndiv = np.reshape(data,(nGOptions, nReadout))
        dataIndiv = np.delete(dataIndiv,  (0, 1,2, 3, 4), axis=1)
#        Plot
        if echo == 0 or echo == 1:
            plt.figure(1)
            plt.plot(np.abs(dataIndiv[0]), 'r', label="-g")
            plt.plot(np.abs(dataIndiv[1]), 'b', label=" 0")
            plt.plot(np.abs(dataIndiv[2]), 'g', label="+g")
            plt.legend()
            
            plt.figure(2)

            fVector = np.linspace(-BW * 1e3 / 2, BW * 1e3 / 2, nReadout-5)
            spectrum0 = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataIndiv[0]))))
            spectrum0 = np.reshape(spectrum0, -1)
            spectrum1 = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataIndiv[1]))))
            spectrum1 = np.reshape(spectrum1, -1)
            spectrum2 = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataIndiv[2]))))
            spectrum2 = np.reshape(spectrum2, -1)
            # dataIndivFft = np.fft.fft(dataIndiv)
            # dataOr1, dataOr2 = np.split(dataIndivFft, 2, axis=1)
            # dataOr = np.concatenate((dataOr2, dataOr1), axis=1)
            plt.plot(fVector,np.abs(spectrum0), 'r', label="-g")
            plt.plot(fVector,np.abs(spectrum1), 'b', label=" 0")
            plt.plot(fVector,np.abs(spectrum2), 'g', label="+g")
            plt.legend()
        elif echo == 2:
            plt.rcParams["figure.figsize"] = (16,6)
            BWaux=(nReadout/(tAdq*1e-6))*1e-3 #kHz
            x1 = np.linspace(blkTime, tAdq, nReadout-5, endpoint=True)*1e-3
            x2 = np.linspace(3*tEcho/2-tAdq/2, tAdq,  nReadout-5, endpoint=True)*1e-3
            x3=  np.linspace(-BWaux/2, BWaux/2, nReadout-5, endpoint=True)
            fig = plt.subplot(121)
            plt.plot(x1, np.abs(dataIndiv[0]), 'r', label="-g")
            plt.plot(x1, np.abs(dataIndiv[2]), 'b', label=" 0")
            plt.plot(x1,  np.abs(dataIndiv[4]), 'g', label="+g")
            plt.xlabel('t(ms)')
            plt.ylabel('A(mV)')
#            plt.ylim(0, 20)
            plt.title('FID')
            plt.legend()
            fig = plt.subplot(122)
            plt.plot(x2, np.abs(dataIndiv[1]), 'r', label="-g")
            plt.plot(x2, np.abs(dataIndiv[3]), 'b', label="0")
            plt.plot(x2, np.abs(dataIndiv[5]), 'g', label="+g")
            plt.xlabel('t(ms)')
            plt.ylabel('A(mV)')
#            plt.ylim(0, 20)
            plt.title('Echo')
            plt.legend()
#            fig = plt.subplot(133)
#            dataIndivFft = np.fft.fft(dataIndiv)
#            dataOr1, dataOr2 = np.split(dataIndivFft, 2, axis=1)
#            dataOr = np.concatenate((dataOr2, dataOr1), axis=1)
#            plt.plot(x3, np.abs(dataOr[0]), 'r', label="-g")
#            plt.plot(x3, np.abs(dataOr[2]), 'b', label=" 0")
#            plt.plot(x3, np.abs(dataOr[4]), 'g', label="+g")
#            plt.xlabel('f(kHz)')
#            plt.ylabel('A(a.u.)')
#            plt.title('FFT')
#            plt.legend()
        plt.show()


if __name__ == "__main__":
    FID_EddyCurrents()
