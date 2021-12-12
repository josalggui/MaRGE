# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 12:40:05 2021

@author: José Miguel Algarín Guisado
MRILAB @ I3M
"""

import sys
# marcos_client path for linux
sys.path.append('../marcos_client')
# marcos_client and PhysioMRI_GUI for Windows
sys.path.append('D:\CSIC\REPOSITORIOS\marcos_client')
sys.path.append('D:\CSIC\REPOSITORIOS\PhysioMRI_GUI')
import numpy as np
import experiment as ex
import matplotlib.pyplot as plt
import scipy.signal as sig
import os
from scipy.io import savemat
from datetime import date,  datetime 
import pdb
from configs.hw_config import Gx_factor
from configs.hw_config import Gy_factor
from configs.hw_config import Gz_factor
st = pdb.set_trace



#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def gtse_standalone(
    init_gpa=False,              # Starts the gpa
    larmorFreq = 3.07491e6,      # Larmor frequency
    rfExAmp = 0.3,             # rf excitation pulse amplitude
    rfReAmp = 0.3,             # rf refocusing pulse amplitude
    rfExTime = 35e-6,          # rf excitation pulse time
    rfReTime = 70e-6,            # rf refocusing pulse time
    gradAmp = 3e-3,              # Gradient amplitude in mT/m
    gradTime = 3.0e-3,                # Gradient flattop time
    echoSpacing = 50e-3,        # time between echoes limits
    gradDelays = np.linspace(2e-3, 17e-3, num=16, endpoint=True),            # gradient delay to sweep
    repetitionTime = 200e-3,     # TR
    nPoints = 1000,                 # Number of acquired points
    etl = 2,                    # Echo train length
    acqTime = 1e-3,             # Acquisition time
    axes = np.array([0, 1, 2]),       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    rdPreemphasis = 2,
    shimming = np.array([-70, -90, 10])*0,       # Shimming along the X,Y and Z axes (a.u. *1e4)
    ):
    
    # rawData fields
    rawData = {}
    inputs = {}
    outputs = {}
    auxiliar = {}
    
    # Miscellaneous
    tau = 0
    blkTime = 10             # Deblanking time (us)
    larmorFreq = larmorFreq*1e-6
    gradRiseTime = 1e-3       # Estimated gradient rise time
    gSteps = int(gradRiseTime*1e6/5)    # Gradient ramp steps
    gradDelay = 9            # Gradient amplifier delay
    rfReAmp = rfExAmp
    rfReTime = 2*rfExTime
    oversamplingFactor = 6
    shimming = shimming*1e-4
    nDelays = len(gradDelays)
    auxiliar['gradDelay'] = gradDelay*1e-6
    auxiliar['gradRiseTime'] = gradRiseTime
    auxiliar['oversamplingFactor'] = oversamplingFactor
    
    # Inputs for rawData
    inputs['larmorFreq'] = larmorFreq      # Larmor frequency
    inputs['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    inputs['rfReAmp'] = rfReAmp             # rf refocusing pulse amplitude
    inputs['rfExTime'] = rfExTime          # rf excitation pulse time
    inputs['rfReTime'] = rfReTime            # rf refocusing pulse time
    inputs['gradAmp'] = gradAmp         
    inputs['gradTime'] = gradTime
    inputs['gradDelays'] = gradDelays
    inputs['echoSpacing'] = echoSpacing        # time between echoes
    inputs['repetitionTime'] = repetitionTime     # TR
    inputs['nPoints'] = nPoints                 # Number of points along readout, phase and slice
    inputs['etl'] = etl                    # Echo train length
    inputs['acqTime'] = acqTime             # Acquisition time
    inputs['axes'] = axes       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    inputs['rdPreemphasis'] = rdPreemphasis
    inputs['shimming'] = shimming
    
    # BW
    BW = nPoints/acqTime*1e-6
    BWov = BW*oversamplingFactor
    samplingPeriod = 1/BWov
    
    # Change gradient values to OCRA units
    gFactor = reorganizeGfactor(axes)
    gradAmp = gradAmp/gFactor[0]*1000/10
    
    # Create functions
    def rfPulse(tStart,rfTime,rfAmplitude,rfPhase):
        txTime = np.array([tStart+blkTime,tStart+blkTime+rfTime])
        txAmp = np.array([rfAmplitude*np.exp(1j*rfPhase),0.])
        txGateTime = np.array([tStart,tStart+blkTime+rfTime])
        txGateAmp = np.array([1,0])
        expt.add_flodict({
            'tx0': (txTime, txAmp),
            'tx_gate': (txGateTime, txGateAmp)
            })

    def rxGate(tStart,gateTime):
        rxGateTime = np.array([tStart,tStart+gateTime])
        rxGateAmp = np.array([1,0])
        expt.add_flodict({
            'rx0_en':(rxGateTime, rxGateAmp), 
            'rx_gate': (rxGateTime, rxGateAmp), 
            })

    def gradPreemphasis(tStart, gTime, gAmp,  gAxis):
        tUp = np.linspace(tStart, tStart+gradRiseTime, num=gSteps+1, endpoint=True)
        tDown = tUp+gradRiseTime+gTime
        t = np.concatenate((tUp, tDown), axis=0)
        dAmp = gAmp/gSteps
        alpha = dAmp/5
        aUp = tau*alpha+np.linspace(dAmp, gAmp, num=gSteps)
        aDown = -tau*alpha+np.linspace(gAmp-dAmp, 0, num=gSteps)
        a = np.concatenate((aUp, np.array([gAmp]),  aDown, np.array([0])), axis=0)
        if gAxis==0:
            expt.add_flodict({'grad_vx': (t, a+shimming[0])})
        elif gAxis==1:
            expt.add_flodict({'grad_vy': (t, a+shimming[1])})
        elif gAxis==2:
            expt.add_flodict({'grad_vz': (t, a+shimming[2])})
    
    def gradTriangle(tStart, gTime, gAmp, gAxis):
        tUp = np.linspace(tStart, tStart+gTime/2, num=int(gTime/10), endpoint=False)
        tDown = tUp+gTime/2
        t = np.concatenate((tUp, tDown), axis=0)
        dAmp = gAmp/gSteps
        aUp = np.linspace(dAmp, gAmp, num=int(gTime/10))
        aDown = np.linspace(gAmp-dAmp, 0, num=int(gTime/10))
        a = np.concatenate((aUp, aDown), axis=0)
        if gAxis==0:
            expt.add_flodict({'grad_vx': (t, a+shimming[0])})
        elif gAxis==1:
            expt.add_flodict({'grad_vy': (t, a+shimming[1])})
        elif gAxis==2:
            expt.add_flodict({'grad_vz': (t, a+shimming[2])})
    
    def gradTrap(tStart, gTime, gAmp, gAxis):
        tUp = np.linspace(tStart, tStart+gradRiseTime, num=gSteps, endpoint=False)
        tDown = tUp+gradRiseTime+gTime
        t = np.concatenate((tUp, tDown), axis=0)
        dAmp = gAmp/gSteps
        aUp = np.linspace(dAmp, gAmp, num=gSteps)
        aDown = np.linspace(gAmp-dAmp, 0, num=gSteps)
        a = np.concatenate((aUp, aDown), axis=0)
        if gAxis==0:
            expt.add_flodict({'grad_vx': (t, a+shimming[0])})
        elif gAxis==1:
            expt.add_flodict({'grad_vy': (t, a+shimming[1])})
        elif gAxis==2:
            expt.add_flodict({'grad_vz': (t, a+shimming[2])})
    
    def gradPulse(tStart, gTime, gAmp,  gAxes):
        t = np.array([tStart, tStart+gradRiseTime+gTime])
        for gIndex in range(np.size(gAxes)):
            a = np.array([gAmp[gIndex], 0])
            if gAxes[gIndex]==0:
                expt.add_flodict({'grad_vx': (t, a+shimming[0])})
            elif gAxes[gIndex]==1:
                expt.add_flodict({'grad_vy': (t, a+shimming[1])})
            elif gAxes[gIndex]==2:
                expt.add_flodict({'grad_vz': (t, a+shimming[2])})
    
    def endSequence(tEnd):
        expt.add_flodict({
                'grad_vx': (np.array([tEnd]),np.array([0]) ), 
                'grad_vy': (np.array([tEnd]),np.array([0]) ), 
                'grad_vz': (np.array([tEnd]),np.array([0]) ),
             })
             
    def iniSequence(tEnd, shimming):
        expt.add_flodict({
                'grad_vx': (np.array([tEnd]),np.array([shimming[0]]) ), 
                'grad_vy': (np.array([tEnd]),np.array([shimming[1]]) ), 
                'grad_vz': (np.array([tEnd]),np.array([shimming[2]]) ),
             })

    def createSequence():
        scanTime = repetitionTime*nDelays
        # Set shimming
        iniSequence(20, shimming)
        for repeIndex in range(nDelays):
            # Initialize time
            tRep = 20e3+repeIndex*repetitionTime+blkTime+rfExTime/2
            
            # Excitation pulse
            t0 = tRep-blkTime-rfExTime/2
            rfPulse(t0,rfExTime,rfExAmp,0)
        
            # Dephasing readout
            t0 = tRep+echoSpacing/2-gradDelays[repeIndex]-gradTime-2*gradRiseTime-gradDelay
            gradPreemphasis(t0, gradTime, gradAmp/2, axes[0])

            # Echo train
            for echoIndex in range(etl):
                # Refocusing pulse
                tEcho = tRep+echoSpacing*(echoIndex+1)
                t0 = tEcho-echoSpacing/2-rfReTime/2-blkTime
                rfPulse(t0, rfReTime, rfReAmp, np.pi/2)
    
                # Readout gradient
                t0 = tEcho-gradTime/2-gradRiseTime-gradDelay
                gradPreemphasis(t0, gradTime, gradAmp*0,  axes[0])
    
                # Rx gate
                t0 = tEcho-acqTime/2
                rxGate(t0, acqTime)
        
        # Set gradients to zero
        endSequence(scanTime)
    
    # Changing time parameters to us
    rfExTime = rfExTime*1e6
    rfReTime = rfReTime*1e6
    echoSpacing = echoSpacing*1e6
    gradTime = gradTime*1e6
    gradDelays = gradDelays*1e6
    repetitionTime = repetitionTime*1e6
    gradRiseTime = gradRiseTime*1e6
        
    # Create full sequence
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BW = 1/samplingPeriod/oversamplingFactor
    acqTime = nPoints/BW        # us
    createSequence()
    
    # Plot sequence:
    expt.plot_sequence()
#    plt.show()
    
    # Run the experiment
    rxd, msgs = expt.run()
    rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
    # Get data
    data = sig.decimate(rxd['rx0'], oversamplingFactor, ftype='fir', zero_phase=True)
    expt.__del__()
    
    # Get position of each echo
    timeVector = np.linspace(-acqTime/2, acqTime/2, num=nPoints, endpoint=True)
    data = np.reshape(data, (nDelays, nPoints*etl))
    data1 = data[:, 0:nPoints]
    data2 = data[:, nPoints::]
    data = np.reshape(data, (nDelays*etl, nPoints))
    echoTime1 = np.zeros(nDelays)
    echoTime2 = np.zeros(nDelays)
    for ii in range(nDelays):
        echoTime1[ii] = timeVector[np.argmax(np.abs(data1[ii, :]))];
        echoTime2[ii] = timeVector[np.argmax(np.abs(data2[ii, :]))];
    
    outputs['EchoTime1'] = echoTime1
    outputs['EchoTime2'] = echoTime2
    
    # Save data
    dt = datetime.now()
    dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
    dt2 = date.today()
    dt2_string = dt2.strftime("%Y.%m.%d")
    if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
        os.makedirs('experiments/acquisitions/%s' % (dt2_string))
            
    if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
        os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
    inputs['name'] = "%s.%s.mat" % ("TSE",dt_string)
    rawData['inputs'] = inputs
    rawData['auxiliar'] = auxiliar
    rawData['outputs'] = outputs
    rawdata = {}
    rawdata['rawData'] = rawData
    savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, "RARE",dt_string),  rawdata)
    
    ## Plots
    fig = plt.figure(2)
    fig.set_size_inches(15, 5)
    # Plot echo map
    ax1 = fig.add_subplot(121)
    ax1.imshow(np.abs(data), cmap='gray', extent=(-acqTime/2*1e-3, acqTime/2*1e-3, 15, 0))
    ax1.set_aspect('auto')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Delay (ms)')
    # Plot echo delay
    ax2 = fig.add_subplot(122)
    ax2.plot(gradDelays*1e-3, echoTime1, gradDelays*1e-3, echoTime2)
    ax2.set_xlabel('Gradient delay (ms)')
    ax2.set_ylabel('Echo time shift (us)')
    ax2.set_title('Echo time shift VS gradient delay')
    
    plt.show()


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def reorganizeGfactor(axes):
    gFactor = np.array([0., 0., 0.])
    
    # Set the normalization factor for readout, phase and slice gradient
    for ii in range(3):
        if axes[ii]==0:
            gFactor[ii] = Gx_factor
        elif axes[ii]==1:
            gFactor[ii] = Gy_factor
        elif axes[ii]==2:
            gFactor[ii] = Gz_factor
    
    return(gFactor)

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


if __name__ == "__main__":

    gtse_standalone()
