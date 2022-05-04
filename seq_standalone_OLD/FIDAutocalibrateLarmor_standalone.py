#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 13:49:34 2022

@author: Teresa
"""

import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from configs import hw_config as conf    #Ver si se importa bien así*********************************************
from configs import globalFunctions as globalF   #Ver si se importa bien así*********************************************

def FIDAutocalibrateLarmor_standalone(
    init_gpa=False,                         # Starts the gpa
    larmorFreq = 3.075,                     # Larmor frequency (MHz)
    rfExAmp = 0.3,                          # RF excitation pulse amplitude (a.u.)
    rfReAmp = 0,                          # RF refocusing pulse amplitude (a.u.)
    rfExPhase = 0,                          # Phase of the excitation pulse (degrees)
    rfExTime = 35,                          # RF excitation pulse time (us)
    rfReTime = 0,                          # RF refocusing pulse time (us)
    repetitionTime = 500,                   # Repetition time
    nReadout = 160,                         # Acquisition points
    acqTime = 4,                            # Acquisition time (ms)
    shimming = np.array([0, 0, 0]),   # Shimming along the X,Y and Z axes (a.u. *1e4)
    dummyPulses = 0,                        # Dummy pulses for T1 stabilization
    plotSeq =0):

    # Changing units
    repetitionTime = repetitionTime*1e3
    acqTime=acqTime*1e3
    shimming = shimming*1e-4
    
    # Global variables              
    addRdPoints = 10                        # Extra points adquired to prevent bad first adquired points by RP
    
    # Conditions about RF
    if rfReAmp==0:
        rfReAmp = 2*rfExAmp
    if rfReTime==0:
        rfReTime = rfExTime
    rfExPhase = rfExPhase*np.pi/180
    rfExAmp = rfExAmp*np.exp(1j*rfExPhase)
    rfRePhase = np.pi/2
    rfReAmp = rfReAmp *np.exp(1j*rfRePhase)
    
    # Matrix size
    nRD = nReadout+2*addRdPoints
    
    # SEQUENCE ################################################################
    # FID and calibration
    acqTimeReal = FIDsequence(larmorFreq,nRD,acqTime,
                              rfExTime,rfExAmp,repetitionTime,
                              shimming,dummyPulses)
   
    #RP and data analysis
    if plotSeq == 1:
       ex.plot_sequence()
       plt.show()
       ex.__del__()
    elif plotSeq == 0:
       print('Running...')
       rxd, msgs = ex.run()
       ex.__del__()
       print('End')
       data = sig.decimate(rxd['rx0']*13.788, conf.oversamplingFactor, ftype='fir', zero_phase=True)
       # data = data[addRdPoints:nReadout+addRdPoints]
       dataPlot(data,acqTimeReal,nRD,addRdPoints,1)
       larmorFreqCal = dataAnalysis(larmorFreq,data,acqTimeReal,nRD,addRdPoints,3)
       plt.show()
    # FID for Larmor frecuency
    acqTimeReal = FIDsequence(larmorFreqCal,nRD,acqTime,
                              rfExTime,rfExAmp,repetitionTime,
                              shimming,dummyPulses)
    #RP and plot echo
    if plotSeq == 1:
       ex.plot_sequence()
       plt.show()
       ex.__del__()
    elif plotSeq == 0:
       print('Running...')
       rxd, msgs = ex.run()
       ex.__del__()
       print('End')
       data = sig.decimate(rxd['rx0']*13.788, conf.oversamplingFactor, ftype='fir', zero_phase=True)
       # data = data[addRdPoints:nReadout+addRdPoints]
       dataPlot(data,acqTimeReal,nRD,addRdPoints,4)
       plt.show()
       
#  SPECIFIC FUNCTIONS   ####################################################################################
def FIDsequence(larmorFreq,nRD,acqTime,rfExTime,rfExAmp,repetitionTime,shimming,dummyPulses):
    # INIT EXPERIMENT
    init_gpa=False
    BW = nRD/acqTime
    BWov = BW*conf.oversamplingFactor
    samplingPeriod = 1/BWov
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BWReal = 1/samplingPeriod/conf.oversamplingFactor
    acqTimeReal = nRD/BWReal  
    
    tIni=20  #us initial time
    # Shimming
    expt.add_flodict({
            'grad_vx': (np.array([tIni]),np.array([shimming[0]])), 
            'grad_vy': (np.array([tIni]),np.array([shimming[1]])),  
            'grad_vz': (np.array([tIni]),np.array([shimming[2]])),
         })
    # Secuence instructions
    t0=20+tIni   #us Initial sequence time
    nRepetitionTime = dummyPulses+1
    for nRep in range(nRepetitionTime):
        txTime=[]
        txAmp=[]
        txGateTime=[]
        txGateAmp=[]
        rxTime = []
        rxAmp = []
        globalF.rfPulse(t0,rfExAmp,rfExTime,txTime,txAmp,txGateTime,txGateAmp)
        if nRep == nRepetitionTime-1:  #Ver este-1*********************************************
            globalF.readoutGate(acqTimeReal/2+conf.blkTime,acqTimeReal,rxTime,rxAmp)
        t0=t0+repetitionTime
    
    expt.add_flodict({
                        'tx0': (txTime, txAmp),
                        'tx_gate': (txGateTime, txGateAmp), 
                        'rx0_en': (rxTime, rxAmp),
                        'rx_gate': (rxTime, rxAmp),
                        })
    # End sequence
    tEnd = t0+nRepetitionTime*repetitionTime #Ver este tiempo *********************************************
    expt.add_flodict({
            'grad_vx': (np.array([tEnd]),np.array([0])), 
            'grad_vy': (np.array([tEnd]),np.array([0])), 
            'grad_vz': (np.array([tEnd]),np.array([0])),
         })
    return acqTimeReal

def dataAnalysis(larmorFreq,data,acqTime,nRD,addRdPoints,nFig): 
    plt.figure(nFig)
    tPlot = np.linspace(-acqTime/2, acqTime/2, nRD,  endpoint ='True')*1e-3
    angle = np.unwrap(np.angle(data[addRdPoints:nRD-addRdPoints]))
    plt.plot(tPlot[addRdPoints:nRD-addRdPoints], angle)
    plt.xlabel('Phase(Rad)')
    plt.ylabel('A(mV)')
    dPhi = angle[-1]-angle[0]
    df = dPhi/(2*np.pi*acqTime)
    larmorFreqCal = larmorFreq + df
    return larmorFreqCal
def dataPlot(data,acqTime,nRD,addRdPoints,nFig): 
    # Echo
    plt.figure(nFig)
    tPlot = np.linspace(-acqTime/2, acqTime/2, nRD,  endpoint ='True')*1e-3
    plt.plot(tPlot[addRdPoints:nRD-addRdPoints], np.abs(data[addRdPoints:nRD-addRdPoints]))
    plt.xlabel('t(ms)')
    plt.ylabel('A(mV)')
    # title= 'RF Amp = '+ str(np.real(rfExAmp))
    # plt.title(title)
    plt.legend()
    # K space
    plt.figure(nFig+1)
    kMax = nRD/acqTime
    fPlot = np.linspace(-kMax , kMax , nRD,  endpoint ='True')*1e-3
    fPlotReal=fPlot[addRdPoints:nRD-addRdPoints]
    dataReal=data[addRdPoints:nRD-addRdPoints]
    dataFft = np.fft.fft(dataReal)
    dataOr1, dataOr2 = np.split(dataFft, 2, axis=1)
    dataOr = np.concatenate((dataOr2, dataOr1), axis=1)
    plt.plot(fPlotReal,np.abs(dataOr))
    plt.xlabel('t(ms)')
    plt.ylabel('A(mV)')
    # title= 'RF Amp = '+ str(np.real(rfExAmp))
    # plt.title(title)
    plt.legend()
    
    
#  MAIN  ######################################################################################################
if __name__ == "__main__":
    FIDAutocalibrateLarmor_standalone()
