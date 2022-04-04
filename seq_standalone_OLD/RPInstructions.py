"""
Created on Tue Nov  9 10:37:29 2021

@author: Teresa
"""

import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import time 

def FID_EddyCurrents(
    init_gpa= False,                 
    larmorFreq=3.07535, 
    rfExAmp=0.3, 
    rfReAmp=None, 
    rfExPhase = 0,
    rfExTime=38, 
    rfReTime=None,
    nReadout = 100,
    tAdq =18*1e3,
    tEcho = 20*1e3,
    echo = 2, #0 FID, 1 Echo, 2 Both
    tRepetition = 2000*1e3,  
#    shimming=[-100, -70, 80],
    shimming=[0, 0, 0],
#    shimming=[-70, -90, 10],
#    gAxis =3,
#    gNsteps =20, #50
#    gRiseTime = 150,
#    gDuration = 400,
#    gAmp = 0.2, # Max.1=50A 0.2=2V=10A; 0.2V=1A
    tDelay =0, 
    plotSeq =1):
    
    #CONSTANTES
    tStart = 20
    txGatePre = 15
    txGatePost = 1
    blkTime = 300  #Tiempo que dejo entre el rfPulse y la lectura. 
    gAmpMax = 1
    nRdOptions = 2 
#    nReadout = nReadout +5
    shimming=np.array(shimming)*1e-4
#    tDelayShimming = 15*1e3
    oversamplingFactor=6
    tAdqIni = tAdq
    nA = 200
    nB = 500
    nC=1200
    
    #CONDICIONES PARAMETROS INICIALES.
#    if gAmp > gAmpMax:
#        gAmp = gAmpMax 
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
    
    for i in range(3):
        if i == 0:
            nReadout = nA
        elif i ==1: 
            nReadout = nB
        elif i ==2: 
            nReadout = nC
        #SECUENCIA
        # BW
        tAdq=tAdqIni
        BW = nReadout/tAdq
        BWov = BW*oversamplingFactor
        samplingPeriod = 1/BWov
        
    #    BW = nReadout/tAdq
    #    rx_period=1/(BW*oversampling_factor)
    #    rx_period = 1/BW
        expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = expt.get_rx_ts()[0]
        BW = 1/samplingPeriod/oversamplingFactor
        tAdq = nReadout/BW  
        tAdqTest =nReadout/BWov
    
        iniSequence(10)
        timeSeq = tStart
        timeSeq = rfPulse(timeSeq,rfExAmp, rfExTime) 
        readoutGate(timeSeq+blkTime,tAdq)
        t1=timeSeq+blkTime
        t2=t1+tAdq
        timeSeq = rfPulse(timeSeq+tEcho,rfReAmp, rfReTime) 
        t3=timeSeq+tEcho-tAdq/2
        timeSeq = readoutGate(timeSeq+tEcho-tAdq/2,tAdq)
        t4=timeSeq
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
            data = sig.decimate(rxd['rx0']*13.788, oversamplingFactor, ftype='fir', zero_phase=True)
            if i == 0:
                dataIndiv1 = np.reshape(data,(nRdOptions, nReadout))
#                dataIndiv1=dataIndiv1[1]
                tAdq1 = tAdq
                samplingPeriod1=samplingPeriod
                dataIndivFft = np.fft.fft(dataIndiv1)
                dataOr1, dataOr2 = np.split(dataIndivFft, 2, axis=1)
                dataIndiv1 = np.concatenate((dataOr2, dataOr1), axis=1)
                dataIndiv1 = np.abs(dataIndiv1[0])
            elif i == 1:
                dataIndiv2 = np.reshape(data,(nRdOptions, nReadout))
#                dataIndiv2=dataIndiv2[1]
                tAdq2 = tAdq
                samplingPeriod2=samplingPeriod
                dataIndivFft = np.fft.fft(dataIndiv2)
                dataOr1, dataOr2 = np.split(dataIndivFft, 2, axis=1)
                dataIndiv2 = np.concatenate((dataOr2, dataOr1), axis=1)
                dataIndiv2 = np.abs(dataIndiv2[0])
            elif i == 2:
                dataIndiv3 = np.reshape(data,(nRdOptions, nReadout))
#                dataIndiv3=dataIndiv3[1]
                tAdq3 = tAdq
                samplingPeriod3=samplingPeriod
                dataIndivFft = np.fft.fft(dataIndiv3)
                dataOr1, dataOr2 = np.split(dataIndivFft, 2, axis=1)
                dataIndiv3 = np.concatenate((dataOr2, dataOr1), axis=1)
                dataIndiv3 = np.abs(dataIndiv3[0])
            time.sleep(1)
    #        dataIndiv = np.delete(dataIndiv,  (0, 1,2, 3, 4), axis=1)
    
#        Plot
#            plt.rcParams["figure.figsize"] = (16,6)
#        BWaux=(nReadout/(tAdq*1e-6))*1e-3 #kHz
#        x1 = np.linspace(t1, t2, nReadout, endpoint=True)*1e-3
#        x2 = np.linspace(t3, t4,   nReadout, endpoint=True)*1e-3
    xA = np.linspace(-tAdq1/2, tAdq1/2,   nA, endpoint=True)*1e-3
    xB = np.linspace(-tAdq2/2, tAdq2/2,   nB, endpoint=True)*1e-3
    xC = np.linspace(-tAdq3/2, tAdq3/2,   nC, endpoint=True)*1e-3
    
    xA = np.linspace(- samplingPeriod1/2,  samplingPeriod1/2,   nA, endpoint=True)
    xB = np.linspace(- samplingPeriod2/2, samplingPeriod2/2,   nB, endpoint=True)
    xC = np.linspace(- samplingPeriod3/2,  samplingPeriod3/2,   nC, endpoint=True)
#        y1 = np.abs(dataIndiv[0])
    yA = np.abs(dataIndiv1)
    yB = np.abs(dataIndiv2)
    yC = np.abs(dataIndiv3)
#        tPlot = np.concatenate((x1, x2),  axis=0)
#        aPlot = np.concatenate((np.abs(dataIndiv[0]), np.abs(dataIndiv[1])),  axis=0)
#        plt.plot(x1[0:4], y1[0:4], 'ro')
#        plt.plot(x1[4:], y1[4:], 'b')
    plt.plot(xA[0:4], yA[0:4], 'go')
    plt.plot(xA[4:], yA[4:], 'g',  label = '100')
    
    plt.plot(xB[0:4], yB[0:4], 'bo')
    plt.plot(xB[4:], yB[4:], 'b', label = '500')
    
    plt.plot(xC[0:4], yC[0:4], 'yo')
    plt.plot(xC[4:], yC[4:], 'y-.', label='1000')
    
    
#        tFilterEcho=(2*tEcho+tStart+rfExTime/2)*1e-3
#        tRealEcho=tFilterEcho+5*tAdq/nReadout*1e-3
    tFilterEcho=0
    tRealEcho1=tFilterEcho+5*tAdq1/nA*1e-3
    tRealEcho2=tFilterEcho+5*tAdq2/nB*1e-3
    tRealEcho3=tFilterEcho+5*tAdq3/nC*1e-3
    print(tRealEcho1)
    print(tRealEcho2)
    print(tRealEcho3)

    tRealEcho1=tFilterEcho+5*(samplingPeriod1*oversamplingFactor/tAdq1)*1e-3
    tRealEcho2=tFilterEcho+5*(samplingPeriod2*oversamplingFactor/tAdq2)*1e-3
    tRealEcho3=tFilterEcho+5*(samplingPeriod3*oversamplingFactor/tAdq3)*1e-3
    
    print(tRealEcho1)
    print(tRealEcho2)
    print(tRealEcho3)
    
    print(xA[np.argmax(yA)])
    print(xB[np.argmax(yB)])
    print(xC[np.argmax(yC)])
    
#    plt.axvline(tFilterEcho, 0, 160, color="black", linestyle="-")
#    plt.axvline(tRealEcho1, 0, 160, color="green", linestyle="--")
#    plt.axvline(tRealEcho2, 0, 160, color="blue", linestyle="--")
#    plt.axvline(tRealEcho3, 0, 160, color="yellow", linestyle="--")
    plt.xlabel('t(ms)')
    plt.ylabel('A(mV)')
    
    plt.legend()
    
#        x3=  np.linspace(-BWaux/2, BWaux/2, nReadout, endpoint=True)
#        fig = plt.subplot(121)
#        plt.plot(x1, np.abs(dataIndiv[0]), 'r')
#        plt.xlabel('t(ms)')
#        plt.ylabel('A(mV)')
#        plt.ylim(0, 200)
#        plt.title('FID')
#        plt.legend()
#        fig = plt.subplot(122)
#        plt.plot(x2, np.abs(dataIndiv[1]), 'r')
#        plt.xlabel('t(ms)')
#        plt.ylabel('A(mV)')
#        plt.ylim(0, 200)
#        plt.title('Echo')
#        plt.legend()
#        fig = plt.subplot(133)
#        dataIndivFft = np.fft.fft(dataIndiv)
#        dataOr1, dataOr2 = np.split(dataIndivFft, 2, axis=1)
#        dataOr = np.concatenate((dataOr2, dataOr1), axis=1)
#        plt.plot(x3, np.abs(dataOr[0]), 'r')
#        plt.xlabel('f(kHz)')
#        plt.ylabel('A(a.u.)')
#        plt.title('FFT')
#        plt.legend()
    plt.show()


if __name__ == "__main__":
    FID_EddyCurrents()
