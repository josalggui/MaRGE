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
from scipy.optimize import curve_fit

def EchoDelay(
    init_gpa= False,                 
    larmorFreq=3.077, 
    rfExAmp=0.3, 
    rfReAmp=None, 
    rfExPhase = 0,
    rfExTime=38, 
    rfReTime=None,
    nReadout = 1001,
    tAdq =2*1e3,
    tEcho = 20*1e3,
    tRepetition = 2000*1e3, 
    tDelay =0, 
    plotSeq =0):

#  INITALISATION OF VARIABLES  ################################################################################
#Constants
    tStart = 20
    txGatePre = 15
    txGatePost = 1
    oversamplingFactor=6
  
    if rfReAmp is None:
        rfReAmp = rfExAmp
    if rfReTime is None:
        rfReTime = 2*rfExTime
    
    rfExPhase = rfExPhase*np.pi/180
    rfExAmp = rfExAmp*np.exp(1j*rfExPhase)
    rfRePhase = np.pi/2
    rfReAmp = rfReAmp *np.exp(1j*rfRePhase)
#Arrays   
    txTime=[]
    txAmp=[]
    txGateTime=[]
    txGateAmp=[]
    rxTime = []
    rxAmp = []
    dataAll  =[]

# About loop of tAdq
    nAdq = 5
    tAdqMin = 2*1e3
    tAdqMax = 18*1e3
    tAdqAll = np.linspace(tAdqMin, tAdqMax, nAdq,  endpoint=True)
    
#  DEFINITION OF PULSES   ####################################################################################
    def rfPulse(tRef, rfAmp, rfDuration, txTimePrevious,txAmpPrevious,  txGateTimePrevious, txGateAmpPrevious):
        txTime = np.array([tRef-rfDuration/2,tRef+rfDuration/2])
        txAmp = np.array([rfAmp,0])
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
    
#  PLOT FUNCTION  ########################################################################################
    def plotDataT(data, tAdqAll, BWAll):
        plt.figure(1)
        colors = cm.rainbow(np.linspace(0, 0.8, len(tAdqAll)))
        for tAdqIndex in range(nAdq):
            tPlot = np.linspace(-tAdqAll[tAdqIndex]/2, tAdqAll[tAdqIndex]/2, nReadout,  endpoint ='True')*1e-3
            leg = 'tAdq = '+ str(np.round(tAdqAll[tAdqIndex]*1e-3))+ 'ms'
            plt.plot(tPlot[0:4], np.abs(data[tAdqIndex, 0:4]), 'r-')
            plt.plot(tPlot[5:], np.abs(data[tAdqIndex, 5:]), label = leg, color=colors[tAdqIndex])
#            plt.plot(tPlot, np.abs(data[tAdqIndex]), label = leg, color=colors[tAdqIndex])
#            tEchoExpected = (tAdqAll[tAdqIndex]/nReadout*6)*1e-3+1/(2* BWAll[tAdqIndex]*1e3)
#            tEchoExpected = (1/BWAll[tAdqIndex]*6)*1e-3+1/(2* BWAll[tAdqIndex]*1e3)
            tEchoExpected = np.mean(np.abs(data[tAdqIndex]))
            plt.axvline(tEchoExpected, 0, 160, color=colors[tAdqIndex], linestyle="--", linewidth= 0.5)
        plt.axvline(0, 0, 160, color="black", linestyle="--")
        plt.xlabel('t(ms)')
        plt.ylabel('A(mV)')
        plt.legend()
    
    def lorFunc(x, xo, g):
        return 1/np.pi/(g*((x-xo)**2+g**2))

    def plotDataK(data, BWAll):
            plt.figure(2)
            colors = cm.rainbow(np.linspace(0, 0.8, len(tAdqAll)))
            for tAdqIndex in range(nAdq):
                fAdq =  np.linspace(-BWAll[tAdqIndex]/2, BWAll[tAdqIndex]/2, nReadout, endpoint=True)*1e3
                leg = 'BW = '+ str(np.round(BWAll[tAdqIndex]*1e3))+ 'kHz'
                dataFft = np.fft.fft(data[tAdqIndex, 5:])
                dataOr1, dataOr2 = np.split(dataFft, 2, axis=0)
                dataFft= np.concatenate((dataOr2, dataOr1), axis=0)
                plt.plot(fAdq[5:], np.abs(dataFft), '-', label = leg, color=colors[tAdqIndex], markersize = 2)
                #AJUSTE
#                pars, cov = curve_fit(f=lorFunc, xdata=fAdq[5:], ydata=np.abs(dataFft), p0=[1,1], bounds=(-np.inf, np.inf))
#                fitData = lorFunc(fAdq[5:], pars[0], pars[1])
#                plt.plot(fAdq[5:], fitData, '-', label = leg, color=colors[tAdqIndex])
#        
#                print(pars[0])
#                tEchoExpected = (BWAll[tAdqIndex]*1e3/nReadout*4)
                fMaxExpected = -BWAll[tAdqIndex]/2*1e3-fAdq[5]
#                plt.plot(fAdq[0:5], [0, 0, 0, 0, 0], 'ro')
                plt.axvline(np.abs(fMaxExpected), 0,7000, color=colors[tAdqIndex], linestyle="--", linewidth= 0.5)
            plt.axvline(0, 0, 7000, color="black", linestyle="--")
            plt.xlabel('f(kHz)')
            plt.ylabel('A(a.u.)')
            plt.title('FFT')
#            plt.xlim(-0.05,  0.05)
            plt.legend()
            
    
   

#  SEQUENCE  ############################################################################################

    for tAdqIndex in range(nAdq):
        
        tAdq = tAdqAll[tAdqIndex]
        txTime=[]
        txAmp=[]
        txGateTime=[]
        txGateAmp=[]
        rxTime = []
        rxAmp = []
        
    # Init experiment
        BW = nReadout/tAdq
        BWov = BW*oversamplingFactor
        samplingPeriod = 1/BWov
        expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = expt.get_rx_ts()[0]
        BW = 1/samplingPeriod/oversamplingFactor
        if tAdqIndex == 0:
            BWAll = BW
        else:
            BWAll=np.append(BWAll,BW)
        tAdq = nReadout/BW  
        
    # TR    
        tRef = tStart+rfExTime/2 
        txTime, txAmp,txGateTime,txGateAmp = rfPulse(tRef,rfExAmp, rfExTime, txTime, txAmp, txGateTime, txGateAmp)
        tRef = tRef+tEcho/2
        txTime, txAmp, txGateTime, txGateAmp = rfPulse(tRef,rfReAmp, rfReTime, txTime, txAmp, txGateTime, txGateAmp)
        tRef = tRef+tEcho/2
        rxTime, rxAmp = readoutGate(tRef, tAdq, rxTime, rxAmp)
        
        expt.add_flodict({
                            'tx0': (txTime, txAmp),
                            'tx_gate': (txGateTime, txGateAmp), 
                            'rx0_en': (rxTime, rxAmp),
                            'rx_gate': (rxTime, rxAmp),
                            })
        time.sleep(1)
        
        if plotSeq == 0:
            print('Running...')
            rxd, msgs = expt.run()
            expt.__del__()
            print('End')
            data = sig.decimate(rxd['rx0']*13.788, oversamplingFactor, ftype='fir', zero_phase=True)
            dataAll = np.concatenate((dataAll, data), axis=0)

   
    if plotSeq == 1:
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq == 0:
        data = np.reshape(dataAll,  (nAdq,  nReadout))
        plotDataT(data, tAdqAll, BWAll)
#        plotDataK(data, BWAll)
        plt.show()

#  MAIN  ######################################################################################################
if __name__ == "__main__":
    EchoDelay()
