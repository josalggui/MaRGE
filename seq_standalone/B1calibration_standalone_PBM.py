"""
Created on Tue Nov  9 10:37:29 2021

@author: Teresa
"""

import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import scipy.signal as sig
import time 
import math

def B1calibration_standalone(
        init_gpa= False,                 
        larmorFreq=3.06, 
        rfExAmpIni=0.8, 
        rfExAmpEnd=0.8, 
        nExAmp=1, 
        rfExPhase = 0,
        rfExTimeIni=1, 
        rfExTimeEnd =100, 
        nExTime =25, 
        nReadout = 1000,
        tAdq =4*1e3,
        tRepetition = 500*1e3, 
        tRingDown=400, 
        gammabar=42.46*1e6, 
        plotSeq =0):

#  INITALISATION OF VARIABLES  ################################################################################
    #CONTANTS
    tStart = 20
    txGatePre = 5
    txGatePost = 1
    oversamplingFactor=6
    
    #ARRAY INITIALIZATIONS 
    txTime=[]
    txAmp=[]
    txGateTime=[]
    txGateAmp=[]
    rxTime = []
    rxAmp = []
    dataAll  =[]
    matrix = np.zeros((nExTime,nExAmp, nReadout))*np.exp(1j)
    t90=np.empty(nExTime,  dtype = int)
    B1=np.empty(nExTime, dtype = int)
    
    #RF PULSES
    
    rfExPhase = rfExPhase*np.pi/180
    rfAmp=np.linspace(rfExAmpIni, rfExAmpEnd, nExAmp,  endpoint=True)
    rfExAmp = rfAmp*np.exp(1j*rfExPhase)
    rfExTime= np.linspace(rfExTimeIni, rfExTimeEnd, nExTime,  endpoint=True)
    
#  DEFINITION OF PULSES   ####################################################################################
    def rfPulse(tRef, rfAmp, rfDuration, txTimePrevious,txAmpPrevious,  txGateTimePrevious, txGateAmpPrevious):
        txTime = np.array([tRef,tRef+rfDuration])
        txAmp = np.array([rfAmp,0])
        txGateTime = np.array([txTime[0]-txGatePre,txTime[1]+txGatePost])
        txGateAmp = np.array([1,0])
        txTime = np.concatenate((txTimePrevious,txTime),  axis=0)
        txAmp = np.concatenate((txAmpPrevious,txAmp ),  axis=0)
        txGateTime = np.concatenate((txGateTimePrevious,txGateTime),  axis=0)
        txGateAmp = np.concatenate((txGateAmpPrevious,txGateAmp),  axis=0)
        return txTime,  txAmp,  txGateTime,  txGateAmp
    
    def readoutGate(tRef,tRd,rxTimePrevious,  rxAmpPrevious):
        rxTime = np.array([tRef, tRef+tRd])
        rxAmp = np.array([1,0])
        rxTime=np.concatenate((rxTimePrevious, rxTime),  axis=0)
        rxAmp=np.concatenate((rxAmpPrevious, rxAmp),  axis=0)
        return rxTime,  rxAmp



#  SEQUENCE  ############################################################################################
    for indexAmp in range(nExAmp):
        for indexExTime in range(nExTime):
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
            tRef = tStart
            txTime, txAmp,txGateTime,txGateAmp = rfPulse(tRef,rfExAmp[indexAmp], rfExTime[indexExTime], txTime, txAmp, txGateTime, txGateAmp)
            tRef = tRingDown+rfExTime[indexExTime]+tRef
            rxTime, rxAmp = readoutGate(tRef, tAdqReal, rxTime, rxAmp)
            
            expt.add_flodict({
                                'tx0': (txTime, txAmp),
                                'tx_gate': (txGateTime, txGateAmp), 
                                'rx0_en': (rxTime, rxAmp),
                                'rx_gate': (rxTime, rxAmp),
                                })
            tPause = (tRepetition-rxTime[1])*1e-6
            time.sleep(tPause)
            
            print(indexExTime,  '.- Running RFamp=' + str(rfExAmp[indexAmp]) + " au, PulseTime=" + str(rfExTime[indexExTime]) + " us")
            rxd, msgs = expt.run()
            expt.__del__()
            print('   End')
            data = sig.decimate(rxd['rx0']*13.788, oversamplingFactor, ftype='fir', zero_phase=True)
            dataAll = np.concatenate((dataAll, data), axis=0)
            matrix[ indexExTime,indexAmp, : ]=data
        
        # Fit to a interpolant spline to get accurate t90
        spl = UnivariateSpline(rfExTime, abs(matrix[:,indexAmp, 1]))
        spl.set_smoothing_factor(0.5)
        interpolatedFIDtime=np.linspace(rfExTimeIni, rfExTimeEnd, 5*nExTime,  endpoint=True)
        fitteddata=spl(interpolatedFIDtime)
   
        # Get the indices of maximum element in fittedata
        data=matrix[ :,indexAmp, 1]
        indexmax = np.argmax(fitteddata)
        t90[indexAmp]=interpolatedFIDtime[indexmax]
        B1[indexAmp]=((math.pi/2)/(2*math.pi*gammabar*t90[indexAmp]*1e-6))*1e6
        
        plt.figure(1)
        plt.subplot(3, 3, indexAmp+1)
        plt.plot(interpolatedFIDtime, fitteddata,  '-k', rfExTime, abs(matrix[:,indexAmp:, 1]), 'g--',rfExTime, (matrix[:,indexAmp:, 1]).real, 'r--', rfExTime, (matrix[:,indexAmp:, 1]).imag, 'b--')
        plt.xlabel('t(us)', fontsize=6)
        plt.ylabel('A(mV)', fontsize=6)       
        plt.tick_params(labelsize=4);
        titleRF= 'RF amp=' + str(float("{:.2f}".format(rfAmp[indexAmp]))) + ', t90=' + str(float("{:.3f}".format(t90[indexAmp]))) +  ' us, B1=' + str(float("{:.3f}".format(B1[indexAmp]))) +' uT'
        plt.title(titleRF,  fontsize=10)
    plt.show()
    
    plt.figure()
    expt.plot_sequence()
    plt.show()
    expt.__del__()
        
#  MAIN  ######################################################################################################
if __name__ == "__main__":
    B1calibration_standalone()

