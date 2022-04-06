# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 11:10:48 2022

@author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
@email: josalggui@i3m.upv.es
"""
import numpy as np
from datetime import date,  datetime
import os
from scipy.io import savemat
import scipy.signal as sig
import experiment as ex
import configs.hw_config as hw


##############################################################
##############################################################
##############################################################


def getIndex(etl=1, nPH=1, sweepMode=1):
    """"
    @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
    @email: josalggui@i3m.upv.es
    Create 'ind' array that give you the order to sweep the k-space phase lines along an echo train length.
    sweepMode = 0: -kMax to kMax
    sweepMode = 1: 0 to kMax
    sweepMode = 2: kMax to 0
    sweepMode = 3: Niquist modulated method
    """
    n2ETL=int(nPH/2/etl)
    ind = []
    if nPH==1:
         ind = np.array([0])
    else: 
        if sweepMode==0:   # Sequential for T2 contrast
            for ii in range(int(nPH/etl)):
                ind = np.concatenate((ind, np.linspace(ii, nPH+ii, num=etl, endpoint=False)), axis=0)
            ind = ind[::-1]
        elif sweepMode==1: # Center-out for T1 contrast
            if etl==nPH:
                ind = np.zeros(nPH)
                ind[0::2] = np.linspace(int(nPH/2), nPH, num=int(nPH/2), endpoint=False)
                ind[1::2] = np.linspace(int(nPH/2)-1, -1, num=int(nPH/2),  endpoint=False)
            else:
                for ii in range(n2ETL):
                    ind = np.concatenate((ind, np.linspace(int(nPH/2)+ii, nPH+ii, num=etl, endpoint=False)), axis=0)
                    ind = np.concatenate((ind, np.linspace(int(nPH/2)-ii-1, -ii-1, num=etl, endpoint=False)), axis=0)
        elif sweepMode==2: # Out-to-center for T2 contrast
            if etl==nPH:
                ind = np.zeros(nPH)
                ind[0::2] = np.linspace(int(nPH/2), nPH, num=int(nPH/2), endpoint=False)
                ind[1::2] = np.linspace(int(nPH/2)-1, -1, num=int(nPH/2),  endpoint=False)
            else:
                for ii in range(n2ETL):
                    ind = np.concatenate((ind, np.linspace(int(nPH/2)+ii, nPH+ii, num=etl, endpoint=False)), axis=0)
                    ind = np.concatenate((ind, np.linspace(int(nPH/2)-ii-1, -ii-1, num=etl, endpoint=False)), axis=0)
            ind = ind[::-1]
        elif sweepMode==3:  # Niquist modulated to reduce ghosting artifact
            if etl==nPH:
                ind = np.arange(0, nPH, 1)
            else:
                for ii in range(int(n2ETL)):
                    ind = np.concatenate((ind, np.arange(0, nPH, 2*n2ETL)+2*ii), axis=0)
                    ind = np.concatenate((ind, np.arange(nPH-1, 0, -2*n2ETL)-2*ii), axis=0)

    return np.int32(ind)


##############################################################
##############################################################
##############################################################


def fixEchoPosition(echoes, data0):
    """"
    @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
    @email: josalggui@i3m.upv.es
    Oversampled data obtained with a given echo train length and readout gradient only is used here to determine the true position of k=0.
    After getting the position of k = 0 for each gradient-spin-echo, it shift the sampled data to place k = 0 at the center of each acquisition window.
    """
    
    etl = np.size(echoes, axis=0)
    n = np.size(echoes, axis=1)
    idx = np.argmax(np.abs(echoes), axis=1)
    idx = idx-int(n/2)
    data1 = data0*0
    for ii in range(etl):
        if idx[ii]>0:
            idx[ii] = 0
        echoes[ii, -idx[ii]::] = echoes[ii, 0:n+idx[ii]]
        data1[:, ii, -idx[ii]::] = data0[:, ii, 0:n+idx[ii]]
    return(data1)
    

##############################################################
##############################################################
##############################################################


def rfSincPulse(expt, tStart, rfTime, rfAmplitude, rfPhase=0, nLobes=7, rewrite=True):
    """"
    @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
    @email: josalggui@i3m.upv.es
    Rf pulse with sinc pulse shape. I use a Hanning window to reduce the banding of the frequency profile.
    """
    txTime = np.linspace(tStart, tStart+rfTime, num=100, endpoint=True)+hw.blkTime
    nZeros = (nLobes+1)
    tx = np.linspace(-nZeros/2, nZeros/2, num = 100, endpoint=True)
    hanning = 0.5*(1+np.cos(2*np.pi*tx/nZeros))
    txAmp = rfAmplitude*np.exp(1j*rfPhase)*hanning*np.abs(np.sinc(tx))
    txGateTime = np.array([tStart,tStart+hw.blkTime+rfTime])
    txGateAmp = np.array([1,0])
    expt.add_flodict({
        'tx0': (txTime, txAmp),
        'tx_gate': (txGateTime, txGateAmp)
        }, rewrite)


##############################################################
##############################################################
##############################################################


def rfRecPulse(expt, tStart, rfTime, rfAmplitude, rfPhase=0, rewrite=True):
    """"
    @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
    @email: josalggui@i3m.upv.es
    Rf pulse with square pulse shape
    """
    txTime = np.array([tStart+hw.blkTime,tStart+hw.blkTime+rfTime])
    txAmp = np.array([rfAmplitude*np.exp(1j*rfPhase),0.])
    txGateTime = np.array([tStart,tStart+hw.blkTime+rfTime])
    txGateAmp = np.array([1,0])
    expt.add_flodict({
        'tx0': (txTime, txAmp),
        'tx_gate': (txGateTime, txGateAmp)
        }, rewrite)


##############################################################
##############################################################
##############################################################


def rxGate(expt, tStart, gateTime, rewrite=True):
    """"
    @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
    @email: josalggui@i3m.upv.es
    """
    rxGateTime = np.array([tStart,tStart+gateTime])
    rxGateAmp = np.array([1,0])
    expt.add_flodict({
        'rx0_en':(rxGateTime, rxGateAmp), 
        'rx_gate': (rxGateTime, rxGateAmp), 
        })


##############################################################
##############################################################
##############################################################


# def gradTrap(expt, tStart, gRiseTime, gFlattopTime, gAmp, gSteps, gAxis, shimming, rewrite=True):
#     """"
#     @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
#     @email: josalggui@i3m.upv.es
#     gradient pulse with trapezoidal shape. Use 1 step to generate a square pulse.
#     """
#     tUp = np.linspace(tStart, tStart+gRiseTime, num=gSteps, endpoint=False)
#     tDown = tUp+gRiseTime+gFlattopTime
#     t = np.concatenate((tUp, tDown), axis=0)
#     dAmp = gAmp/gSteps
#     aUp = np.linspace(dAmp, gAmp, num=gSteps)
#     aDown = np.linspace(gAmp-dAmp, 0, num=gSteps)
#     a = np.concatenate((aUp, aDown), axis=0)
#     if gAxis==0:
#         expt.add_flodict({'grad_vx': (t, a+shimming[0])}, rewrite)
#     elif gAxis==1:
#         expt.add_flodict({'grad_vy': (t, a+shimming[1])}, rewrite)
#     elif gAxis==2:
#         expt.add_flodict({'grad_vz': (t, a+shimming[2])}, rewrite)
def gradTrap(expt, tStart, gRiseTime, gFlattopTime, gAmp, gSteps, gAxis, shimming, rewrite=True):
    """"
    @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
    @email: josalggui@i3m.upv.es
    gradient pulse with trapezoidal shape. Use 1 step to generate a square pulse.
    Time inputs in us
    Amplitude inputs in T/m
    """
    tUp = np.linspace(tStart, tStart+gRiseTime, num=gSteps, endpoint=False)
    tDown = tUp+gRiseTime+gFlattopTime
    t = np.concatenate((tUp, tDown), axis=0)
    dAmp = gAmp/gSteps
    aUp = np.linspace(dAmp, gAmp, num=gSteps)
    aDown = np.linspace(gAmp-dAmp, 0, num=gSteps)
    a = np.concatenate((aUp, aDown), axis=0)/hw.gFactor[gAxis] 
    if gAxis==0:
        expt.add_flodict({'grad_vx': (t, a+shimming[0])}, rewrite)
    elif gAxis==1:
        expt.add_flodict({'grad_vy': (t, a+shimming[1])}, rewrite)
    elif gAxis==2:
        expt.add_flodict({'grad_vz': (t, a+shimming[2])}, rewrite)


##############################################################
##############################################################
##############################################################


# def gradMomentTrap(expt, tStart, gFlattopTime, gMoment, gAxis, shimming, gSteps, rewrite=True):
#     """"
#     @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
#     @email: josalggui@i3m.upv.es
#     gradient pulse with trapezoidal shape. Use 1 step to generate a square pulse.
#     """
#     tUp = np.linspace(tStart, tStart+gRiseTime, num=gSteps, endpoint=False)
#     tDown = tUp+gRiseTime+gFlattopTime
#     t = np.concatenate((tUp, tDown), axis=0)
#     dAmp = gAmp/gSteps
#     aUp = np.linspace(dAmp, gAmp, num=gSteps)
#     aDown = np.linspace(gAmp-dAmp, 0, num=gSteps)
#     a = np.concatenate((aUp, aDown), axis=0)
#     if gAxis==0:
#         expt.add_flodict({'grad_vx': (t, a+shimming[0])}, rewrite)
#     elif gAxis==1:
#         expt.add_flodict({'grad_vy': (t, a+shimming[1])}, rewrite)
#     elif gAxis==2:
#         expt.add_flodict({'grad_vz': (t, a+shimming[2])}, rewrite)
        

##############################################################
##############################################################
##############################################################


def endSequence(expt, tEnd):
    expt.add_flodict({
            'grad_vx': (np.array([tEnd]),np.array([0])), 
            'grad_vy': (np.array([tEnd]),np.array([0])), 
            'grad_vz': (np.array([tEnd]),np.array([0])),
            'rx0_en':(np.array([tEnd]),np.array([0])), 
            'rx_gate': (np.array([tEnd]),np.array([0])),
            'tx0': (np.array([tEnd]),np.array([0])),
            'tx_gate': (np.array([tEnd]),np.array([0]))
         })


##############################################################
##############################################################
##############################################################


def iniSequence(expt, t0, shimming, rewrite=True):
    expt.add_flodict({
            'grad_vx': (np.array([t0]),np.array([shimming[0]]) ), 
            'grad_vy': (np.array([t0]),np.array([shimming[1]]) ), 
            'grad_vz': (np.array([t0]),np.array([shimming[2]]) ),
            'rx0_en':(np.array([t0]),np.array([0])), 
            'rx_gate': (np.array([t0]),np.array([0])),
            'tx0': (np.array([t0]),np.array([0])),
            'tx_gate': (np.array([t0]),np.array([0]))
         }, rewrite)


##############################################################
##############################################################
##############################################################


def setGradient(expt, t0, gAmp, gAxis, rewrite=True):
    """"
    @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
    @email: josalggui@i3m.upv.es
    Set the one gradient to a given value
    Time inputs in us
    Amplitude inputs in Ocra1 units
    """
    if gAxis==0:
        expt.add_flodict({'grad_vx':(np.array([t0]), np.array([gAmp]))}, rewrite)
    elif gAxis==1:
        expt.add_flodict({'grad_vy':(np.array([t0]), np.array([gAmp]))}, rewrite)
    elif gAxis==2:
        expt.add_flodict({'grad_vz':(np.array([t0]), np.array([gAmp]))}, rewrite)


##############################################################
##############################################################
##############################################################


def saveRawData(rawData):
    """"
    @author: T. Guallart-Naval, MRILab, i3M, CSIC, Valencia, Spain
    @email: teresa.guallart@tesoroimaging.com
    Save the rawData
    """
    # Save data
    dt = datetime.now()
    dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
    dt2 = date.today()
    dt2_string = dt2.strftime("%Y.%m.%d")
    if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
        os.makedirs('experiments/acquisitions/%s' % (dt2_string))
    if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
        os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
    rawData['fileName'] = "%s.%s.mat" % (rawData['seqName'],dt_string)
    savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, rawData['seqName'],dt_string),  rawData)
    return(rawData)
    

##############################################################
##############################################################
##############################################################


def freqCalibration(rawData):
    # Create inputs from rawData
    larmorFreq = rawData['larmorFreq']*1e-6
    samplingPeriod = rawData['samplingPeriod']
    nPoints = rawData['nPoints']
    addRdPoints = rawData['addRdPoints']
    
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=False, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BW = 1/samplingPeriod/hw.oversamplingFactor
    acqTime = nPoints[0]/BW        # us
    rawData['bw'] = BW
    rawData['acqTime'] = acqTime*1e-6
    createFreqCalSequence(expt, rawData)
    rxd, msgs = expt.run()
    dataFreqCal = sig.decimate(rxd['rx0']*13.788, hw.oversamplingFactor, ftype='fir', zero_phase=True)
    dataFreqCal = dataFreqCal[addRdPoints:nPoints[0]+addRdPoints]
    # Plot fid
#    plt.figure(1)
#    tVector = np.linspace(-acqTime/2, acqTime/2, num=nPoints[0],endpoint=True)*1e-3
#    plt.subplot(1, 2, 1)
#    plt.plot(tVector, np.abs(dataFreqCal))
#    plt.title("Signal amplitude")
#    plt.xlabel("Time (ms)")
#    plt.ylabel("Amplitude (mV)")
#    plt.subplot(1, 2, 2)
    angle = np.unwrap(np.angle(dataFreqCal))
#    plt.title("Signal phase")
#    plt.xlabel("Time (ms)")
#    plt.ylabel("Phase (rad)")
#    plt.plot(tVector, angle)
    # Get larmor frequency
    dPhi = angle[-1]-angle[0]
    df = dPhi/(2*np.pi*acqTime)
    larmorFreq += df
    rawData['larmorFreq'] = larmorFreq*1e6
    print("f0 = %s MHz" % (round(larmorFreq, 5)))
    # Plot sequence:
#    expt.plot_sequence()
#    plt.show()
    # Delete experiment:
    expt.__del__()
    return(rawData)


##############################################################
##############################################################
##############################################################


def createFreqCalSequence(expt, rawData):
    # Def variables
    shimming = rawData['shimming']
    rfExTime = rawData['rfExTime']*1e6
    rfExAmp = rawData['rfExAmp']
    rfReTime = rawData['rfReTime']*1e6
    rfReAmp = rawData['rfReAmp']
    echoSpacing = rawData['echoSpacing']*1e6
    acqTime = rawData['acqTime']*1e6
    addRdPoints = rawData['addRdPoints']
    BW = rawData['bw']
    repetitionTime = rawData['repetitionTime']*1e6
    
    
    t0 = 20
    
    # Shimming
    iniSequence(expt, t0, shimming)
        
    # Excitation pulse
    t0 +=20e3
    rfRecPulse(expt, t0,rfExTime,rfExAmp)
    
    # Refocusing pulse
    t0 += rfExTime/2+echoSpacing/2-rfReTime/2
    rfRecPulse(expt, t0, rfReTime, rfReAmp, np.pi/2)
    
    # Rx
    t0 += hw.blkTime+rfReTime/2+echoSpacing/2-acqTime/2-addRdPoints/BW
    rxGate(expt, t0, acqTime+2*addRdPoints/BW)
    
    # Finalize sequence
    endSequence(expt, repetitionTime)
