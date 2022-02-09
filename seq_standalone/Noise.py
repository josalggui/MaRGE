"""
Created on Tue Nov  9 10:37:29 2021

@author: Teresa
"""

import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as sig
import time 

def rabiflops_standalone(
    init_gpa= False,                 
    larmorFreq=3.062, 
    rfExAmp=0.0, 
    rfReAmp=None, 
    rfExPhase = 0,
    rfExTimeIni=0, 
    rfExTimeEnd = 0, 
    nExTime =1, 
    nReadout =2501,
    tAdq = 50*1e3,
    tEcho =300*1e3,
    tRepetition = 500*1e3, 
    plotSeq = 0, 
    shimming=[0, 0, 0]):


#BW 500 tAcq 5
#BW 50   tAcq 50

#  INITALISATION OF VARIABLES  ################################################################################
    #CONTANTS
    tStart = 20
    txGatePre = 15
    txGatePost = 1
    oversamplingFactor=6
    shimming=np.array(shimming)*1e-4
    
    #ARRAY INITIALIZATIONS 
    txTime=[]
    txAmp=[]
    txGateTime=[]
    txGateAmp=[]
    rxTime = []
    rxAmp = []
    dataAll  =[]
    
    #RF PULSES
    if rfReAmp is None:
        rfReAmp = rfExAmp
    rfExPhase = rfExPhase*np.pi/180
    rfExAmp = rfExAmp*np.exp(1j*rfExPhase)
    rfRePhase = 0
    rfReAmp = rfReAmp *np.exp(1j*rfRePhase)
    #Excitation times
    rfExTime= np.linspace(rfExTimeIni, rfExTimeEnd, nExTime,  endpoint=True)
    
#  DEFINITION OF PULSES   ####################################################################################
    def rfPulse(tRef, rfAmp, rfDuration, txTimePrevious,txAmpPrevious,  txGateTimePrevious, txGateAmpPrevious):
        txTime = np.array([tRef-rfDuration/2,tRef+rfDuration/2])
        txAmp = np.array([rfAmp,0.])
        txGateTime = np.array([txTime[0]-txGatePre,txTime[1]+txGatePost])
        txGateAmp = np.array([1,0])
        txTime = np.concatenate((txTimePrevious,txTime),  axis=0)
        txAmp = np.concatenate((txAmpPrevious,txAmp ),  axis=0)
        txGateTime = np.concatenate((txGateTimePrevious,txGateTime),  axis=0)
        txGateAmp = np.concatenate((txGateAmpPrevious,txGateAmp),  axis=0)
        return txTime,  txAmp,  txGateTime,  txGateAmp
    
    def readoutGate(tRef,tRd,rxTimePrevious,  rxAmpPrevious):
        rxTime = np.array([tRef-tRd/2, tRef+tRd/2])
        rxAmp = np.array([1,0])
        rxTime=np.concatenate((rxTimePrevious, rxTime),  axis=0)
        rxAmp=np.concatenate((rxAmpPrevious, rxAmp),  axis=0)
        return rxTime,  rxAmp


#  SPECIFIC FUNCTIONS   ####################################################################################
    def  plotData(data, rfExTime, tAdqReal, nRd):
       plt.figure(1)
       for indexExTime in range(nExTime):
            tPlot = np.linspace(0, tAdqReal, nReadout,  endpoint ='True')*1e-3+indexExTime*tAdqReal*1e-3
            plt.plot(tPlot[5:], np.abs(data[indexExTime, 5:]))
            plt.plot(tPlot[5:], np.real(data[indexExTime, 5:]))
            plt.plot(tPlot[5:], np.imag(data[indexExTime, 5:]))
       plt.xlabel('t(ms)')
       plt.ylabel('A(mV)')
       vRMS=np.std(np.abs(data[0, 5:]))
       titleRF= 'BW = '+ str(np.round(nRd/(tAdqReal)*1e3))+'kHz; Vrms ='+str(vRMS)
       plt.title(titleRF)
       
    def plotDataK(data, BW, nReadout):
            plt.figure(2)
            fAdq =  np.linspace(-BW/2, BW/2, nReadout, endpoint=True)*1e3
            dataFft = np.fft.fft(data[0, 5:])
            dataOr1, dataOr2 = np.split(dataFft, 2, axis=0)
            dataFft= np.concatenate((dataOr2, dataOr1), axis=0)
            plt.plot(fAdq[5:], np.abs(dataFft), 'r-')
            plt.xlabel('f(kHz)')
            plt.ylabel('A(a.u.)')
            plt.title('FFT')
#            plt.xlim(-0.05,  0.05)
            plt.legend()


#  SEQUENCE  ############################################################################################

    for indexExTime in range(nExTime):
        
        rfReTime = 2*rfExTime[indexExTime]
    
        txTime=[]
        txAmp=[]
        txGateTime=[]
        txGateAmp=[]
        rxTime = []
        rxAmp = []
        
        # INIT EXPERIMENT
        BW = nReadout/tAdq
        BWov = BW*oversamplingFactor
        samplingPeriod = 1/BWov
        expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = expt.get_rx_ts()[0]
        BWReal = 1/samplingPeriod/oversamplingFactor
        tAdqReal = nReadout/BWReal  
        # TR    
        tRef = tStart+rfExTime[indexExTime]/2
        txTime, txAmp,txGateTime,txGateAmp = rfPulse(tRef,rfExAmp, rfExTime[indexExTime], txTime, txAmp, txGateTime, txGateAmp)
        tRef = tRef+tEcho/2
        txTime, txAmp, txGateTime, txGateAmp = rfPulse(tRef,rfReAmp, rfReTime, txTime, txAmp, txGateTime, txGateAmp)
        tRef = tRef+tEcho/2
        rxTime, rxAmp = readoutGate(tRef, tAdqReal, rxTime, rxAmp)
        
        expt.add_flodict({
                            'tx0': (txTime, txAmp),
                            'tx_gate': (txGateTime, txGateAmp), 
                            'rx0_en': (rxTime, rxAmp),
                            'rx_gate': (rxTime, rxAmp),
                            })
        if plotSeq == 0:
            print(indexExTime,  '.- Running...')
            rxd, msgs = expt.run()
            expt.__del__()
            print('   End')
            data = sig.decimate(rxd['rx0']*13.788, oversamplingFactor, ftype='fir', zero_phase=True)
            dataAll = np.concatenate((dataAll, data), axis=0)
        elif plotSeq == 1:
            expt.plot_sequence()
            plt.show()
            expt.__del__()

   
    if plotSeq == 1:
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq == 0:
        data = np.reshape(dataAll,  (nExTime,  nReadout))
        plotData(data, rfExTime, tAdqReal, nReadout)
        plotDataK(data, BWReal, nReadout)
        plt.show()

#  MAIN  ######################################################################################################
if __name__ == "__main__":
    rabiflops_standalone()
