"""
Created on Tue Nov  9 10:37:29 2021

@author: Teresa
"""

import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt

def FID_EddyCurrents(
    init_gpa= False,                 
    larmorFreq=3.0765, 
    rfExAmp=0.3, 
    rfReAmp=None, 
    rfExPhase = 0,
    rfExTime=35, 
    rfReTime=None,
    nReadout = 1000,
    tAdq =10*1e3,
    tEcho = 20*1e3,
    echo = 0, #0 FID, 1 Echo.
    tRepetition = 2000*1e3, 
#    shimming=[-100, -90, 50],
    shimming=[0, 0, 0],
#    shimming=[-70, -90, 10],
    gAxis = 3,
    gNsteps = 1, #50
    gRiseTime = 0,
    gDuration = 0,
    gAmp = 0.2, # Max.1=50A 0.2=2V=10A; 0.2V=1A
    tDelay = 0, 
    plotSeq =1):
    
    #CONSTANTES
    tStart = 20
    txGatePre = 15
    txGatePost = 1
    blkTime = 300  #Tiempo que dejo entre el rfPulse y la lectura. 
    gAmpMax = 1
    nGOptions = 3
    nReadout = nReadout +5
    shimming=np.array(shimming)*1e-4
    
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
    timeSeq = tStart
    for n in range(nGOptions):
        timeSeq = gradPulse(timeSeq, gDuration, gAmp[n], gAxis)
        timeSeq = rfPulse(timeSeq+tDelay,rfExAmp, rfExTime) 
#        timeSeq = readoutGate(timeSeq+blkTime,tAdq)
        if echo == 0:
            timeSeq=readoutGate(timeSeq+blkTime,tAdq)
        elif echo == 1: 
            timeSeq = rfPulse(timeSeq+tEcho,rfReAmp, rfReTime) 
            timeSeq = readoutGate(timeSeq+tEcho-tAdq/2,tAdq)
        timeSeq=tRepetition*(n+1)+tStart
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
        dataIndiv = np.reshape(data,(nGOptions, nReadout))
        dataIndiv = np.delete(dataIndiv,  (0, 1,2, 3, 4), axis=1)
        
        #Plot
        plt.figure(1)
        plt.plot(np.abs(dataIndiv[0]), 'r', label="-g")
        plt.plot(np.abs(dataIndiv[1]), 'b', label=" 0")
        plt.plot(np.abs(dataIndiv[2]), 'g', label="+g")
        plt.legend()
#        
#        plt.figure(2)
#        
#        plt.plot(np.abs(np.fft.fft(dataIndiv[0])), 'r', label="-g")
#        plt.plot(np.abs(dataIndiv[1]), 'b', label=" 0")
#        plt.plot(np.abs(dataIndiv[2]), 'g', label="+g")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    FID_EddyCurrents()
