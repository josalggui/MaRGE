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


def rare2_standalone(
    init_gpa=False,              # Starts the gpa
    nScans = 6,                 # NEX
    larmorFreq = 3.073e6,      # Larmor frequency
    rfExAmp = 0.3,             # rf excitation pulse amplitude
    rfReAmp = None,             # rf refocusing pulse amplitude
    rfExTime = 40e-6,          # rf excitation pulse time
    rfReTime = None,            # rf refocusing pulse time
    echoSpacing = 20e-3,        # time between echoes
    repetitionTime = 500e-3,     # TR
    fov = np.array([12e-2,12e-2,80e-2]),           # FOV along readout, phase and slice
    dfov = np.array([0e-2, 0e-2, -20e-2]),            # Displacement of fov center
    nPoints = np.array([60, 60, 4]),                 # Number of points along readout, phase and slice
    etl = 15,                   # Echo train length
    acqTime = 4e-3,             # Acquisition time
    axes = np.array([0, 2, 1]),       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    axesEnable = np.array([1, 1, 1]), # 1-> Enable, 0-> Disable
    sweepMode = 1,               # 0->k2k,  1->02k,  2->k20, 3->Niquist modulated
    phaseGradTime = 1000e-6,       # Phase and slice dephasing time
    rdPreemphasis = 1.008,
    drfPhase = 0, 
    dummyPulses = 1,                     # Dummy pulses for T1 stabilization
    shimming = np.array([-70, -90, 10])
    ):
    
    # rawData fields
    rawData = {}
    inputs = {}
    outputs = {}
    auxiliar = {}
    
    # Miscellaneous
    blkTime = 10             # Deblanking time (us)
    larmorFreq = larmorFreq*1e-6
    gradRiseTime = 100e-6       # Estimated gradient rise time
    gradDelay = 9            # Gradient amplifier delay
    addRdPoints = 10             # Initial rd points to avoid artifact at the begining of rd
    gammaB = 42.56e6            # Gyromagnetic ratio in Hz/T
    rfReAmp = rfExAmp
    rfReTime = 2*rfExTime
    deadTime = 200
    oversamplingFactor = 6
    addRdGradTime = 1000     # Additional readout gradient time to avoid turn on/off effects on the Rx channel
    shimming = np.array(shimming)*1e-4
    
    # Matrix size
    nRD = nPoints[0]+2*addRdPoints
    nPH = nPoints[1]*axesEnable[1]+(1-axesEnable[1])
    nSL = nPoints[2]*axesEnable[2]+(1-axesEnable[2])
    nPoints[1] = nPH
    nPoints[2] = nSL
    
    # Inputs for rawData
    inputs['nScans'] = nScans
    inputs['larmorFreq'] = larmorFreq      # Larmor frequency
    inputs['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    inputs['rfReAmp'] = rfReAmp             # rf refocusing pulse amplitude
    inputs['rfExTime'] = rfExTime          # rf excitation pulse time
    inputs['rfReTime'] = rfReTime            # rf refocusing pulse time
    inputs['echoSpacing'] = echoSpacing        # time between echoes
    inputs['repetitionTime'] = repetitionTime     # TR
    inputs['fov'] = np.array(fov)*1e-3           # FOV along readout, phase and slice
    inputs['dfov'] = dfov            # Displacement of fov center
    inputs['nPoints'] = nPoints                 # Number of points along readout, phase and slice
    inputs['etl'] = etl                    # Echo train length
    inputs['acqTime'] = acqTime             # Acquisition time
    inputs['axes'] = axes       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    inputs['axesEnable'] = axesEnable # 1-> Enable, 0-> Disable
    inputs['sweepMode'] = sweepMode               # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    inputs['phaseGradTime'] = phaseGradTime       # Phase and slice dephasing time
    inputs['rdPreemphasis'] = rdPreemphasis
    inputs['drfPhase'] = drfPhase 
    inputs['dummyPulses'] = dummyPulses                    # Dummy pulses for T1 stabilization
    inputs['shimming'] = shimming
    
    # ETL if nPH = 1
    if etl>nPH:
        etl = nPH
    
    # BW
    BW = nPoints[0]/acqTime*1e-6
    BWov = BW*oversamplingFactor
    samplingPeriod = 1/BWov
    
    # Readout dephasing time
    rdDephTime = (acqTime-gradRiseTime)/2
    
    # Phase and slice de- and re-phasing time
    if phaseGradTime==0:
        phaseGradTime = echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime
    elif phaseGradTime>echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime:
        phaseGradTime = echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime
        
    # Max gradient amplitude
    rdGradAmplitude = nPoints[0]/(gammaB*fov[0]*acqTime)*axesEnable[0]
    phGradAmplitude = nPH/(2*gammaB*fov[1]*(phaseGradTime+gradRiseTime))*axesEnable[1]
    slGradAmplitude = nSL/(2*gammaB*fov[2]*(phaseGradTime+gradRiseTime))*axesEnable[2]
    auxiliar['rdGradAmplitude'] = rdGradAmplitude
    auxiliar['phGradAmplitude'] = phGradAmplitude
    auxiliar['slGradAmplitude'] = slGradAmplitude
    
    # Change gradient values to OCRA units
    gFactor = reorganizeGfactor(axes)
    auxiliar['gFactor'] = gFactor
    rdGradAmplitude = rdGradAmplitude/gFactor[0]*1000/5
    phGradAmplitude = phGradAmplitude/gFactor[1]*1000/5
    slGradAmplitude = slGradAmplitude/gFactor[2]*1000/5
    
    # Phase and slice gradient vector
    phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
    slGradients = np.linspace(-slGradAmplitude,slGradAmplitude,num=nSL,endpoint=False)
    
    # Initialize the experiment
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BW = 1/samplingPeriod/oversamplingFactor
    acqTime = nPoints[0]/BW        # us
    auxiliar['bandwidth'] = BW*1e6
    
    # Create an rf pulse function
    def rfPulse(tStart,rfTime,rfAmplitude,rfPhase):
        txTime = np.array([tStart+blkTime,tStart+blkTime+rfTime])
        txAmp = np.array([rfAmplitude*np.exp(1j*rfPhase),0.])
        txGateTime = np.array([tStart,tStart+blkTime+rfTime])
        txGateAmp = np.array([1,0])
        expt.add_flodict({
            'tx0': (txTime, txAmp),
            'tx_gate': (txGateTime, txGateAmp)
            })

    # Readout function
    def rxGate(tStart,gateTime):
        rxGateTime = np.array([tStart,tStart+gateTime])
        rxGateAmp = np.array([1,0])
        expt.add_flodict({
            'rx0_en':(rxGateTime, rxGateAmp), 
            'rx_gate': (rxGateTime, rxGateAmp), 
            })

    # Gradients
    def gradPulse(tStart, gTime, gAmp, gAxes):
        t = np.array([tStart, tStart+gradRiseTime+gTime])
        for gIndex in range(np.size(gAxes)):
            a = np.array([gAmp[gIndex], 0])
            if gAxes[gIndex]==0:
                expt.add_flodict({'grad_vx': (t, a+shimming[0])})
            elif gAxes[gIndex]==1:
                expt.add_flodict({'grad_vy': (t, a+shimming[1])})
            elif gAxes[gIndex]==2:
                expt.add_flodict({'grad_vz': (t, a+shimming[2])})
    
    # End gradients
    def endSequence(sequenceTime):
        expt.add_flodict({
            'grad_vx': (np.array([sequenceTime]),np.array([0])), 
            'grad_vy': (np.array([sequenceTime]),np.array([0])), 
            'grad_vz': (np.array([sequenceTime]),np.array([0])),
        })
    
    def iniSequence(tEnd):
        expt.add_flodict({
            'grad_vx': (np.array([tEnd]),np.array([shimming[0]]) ), 
            'grad_vy': (np.array([tEnd]),np.array([shimming[1]]) ), 
            'grad_vz': (np.array([tEnd]),np.array([shimming[2]]) ),
        })

    # Changing time parameters to us
    rfExTime = rfExTime*1e6
    rfReTime = rfReTime*1e6
    echoSpacing = echoSpacing*1e6
    repetitionTime = repetitionTime*1e6
    gradRiseTime = gradRiseTime*1e6
    phaseGradTime = phaseGradTime*1e6
    rdDephTime = rdDephTime*1e6
    
    # Create sequence instructions
    phIndex = 0
    slIndex = 0
    nRepetitions = int(nPH*nSL)+dummyPulses
    scanTime = (nPH*nSL+dummyPulses)*repetitionTime
    iniSequence(20)
    for repeIndex in range(nRepetitions):
        # Initialize time
        t0 = 20+repetitionTime*repeIndex
        # Excitation pulse
        t0 += rfReTime/2-rfExTime/2
        rfPulse(t0,rfExTime,rfExAmp,drfPhase*np.pi/180)
    
        # Dephasing readout
        t0 += blkTime+rfExTime-gradDelay
        if repeIndex>=dummyPulses:         # This is to account for dummy pulses
            gradPulse(t0, acqTime+2*addRdGradTime, [rdGradAmplitude/2*rdPreemphasis], [axes[0]])
    
#        # First readout to avoid RP initial readout effect
#        if repeIndex>=dummyPulses:         # This is to account for dummy pulses
#            rxGate(t0+gradDelay+deadTime, acqTime+2*addRdPoints/BW)
        
        # Echo train
        for echoIndex in range(etl):
            # Refocusing pulse
            if echoIndex == 0:
                t0 += (-rfExTime+echoSpacing-rfReTime)/2-blkTime
            else:
                t0 += gradDelay-acqTime/2+echoSpacing/2-rfReTime/2-blkTime-addRdGradTime
            rfPulse(t0, rfReTime, rfReAmp, np.pi/2)

            # Dephasing phase and slice gradients
            t0 += blkTime+rfReTime
            if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                gradPulse(t0, phaseGradTime, [phGradients[phIndex]], [axes[1]])
                gradPulse(t0, phaseGradTime, [slGradients[slIndex]], [axes[2]])
            
            # Readout gradient
            t0 += -rfReTime/2+echoSpacing/2-acqTime/2-gradRiseTime-gradDelay-addRdGradTime
            if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                gradPulse(t0, acqTime+2*addRdGradTime, [rdGradAmplitude], [axes[0]])

            # Rx gate
            t0 += gradDelay+gradRiseTime+addRdGradTime-addRdPoints/BW
            if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                rxGate(t0, acqTime+2*addRdPoints/BW)

            # Rephasing phase and slice gradients
            t0 += addRdPoints/BW+acqTime-gradDelay+addRdGradTime
            if (echoIndex<etl-1 and repeIndex>=dummyPulses):
                gradPulse(t0, phaseGradTime, [-phGradients[phIndex]], [axes[1]])
                gradPulse(t0, phaseGradTime, [-slGradients[slIndex]], [axes[2]])

        # Update the phase and slice gradient
        if repeIndex>=dummyPulses:
            if phIndex == nPH-1:
                phIndex = 0
                slIndex += 1
            else:
                phIndex += 1
        
        if repeIndex==nRepetitions-1:
            endSequence(scanTime)
            
    # Plot sequence:
#    expt.plot_sequence()
#    plt.show()
    
    # Run the experiment
    dataFull = []
    for repeIndex in range(nScans):
        rxd, msgs = expt.run()
        rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
        # Get data
        currentData = sig.decimate(rxd['rx0'], oversamplingFactor, ftype='fir', zero_phase=True)
        dataFull = np.concatenate((dataFull,currentData),axis=0)
    expt.__del__()
    
    #Delete the addRdPoints
    dataFull = np.reshape(dataFull, (nPH*nSL*etl*nScans, nRD))
    dataFull = dataFull[:, addRdPoints:addRdPoints+nPoints[0]]
    dataFull = np.reshape(dataFull, (1, nPoints[0]*nPH*nSL*etl*nScans))
    
    # Average data
    data = np.reshape(dataFull, (nScans, nPoints[0]*nPoints[1]*nPoints[2]*etl))
    data = np.average(data, axis=0)
    
    # Split into different images
    data = np.reshape(data, (nPoints[2], nPoints[1], nPoints[0]*etl))
    
    # Fix the position of the sample according t dfov
    kMax = np.array(nPoints)/(2*np.array(fov))*np.array(axesEnable)
    kRD = np.linspace(-kMax[0],kMax[0],num=nPoints[0],endpoint=False)
    kPH = np.linspace(-kMax[1],kMax[1],num=nPoints[1],endpoint=False)
    kSL = np.linspace(-kMax[2],kMax[2],num=nPoints[2],endpoint=False)
    kPH = kPH[::-1]
    kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
    kRD = np.reshape(kRD, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    kPH = np.reshape(kPH, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    kSL = np.reshape(kSL, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    dPhase = np.exp(-2*np.pi*1j*(dfov[0]*kRD+dfov[1]*kPH+dfov[2]*kSL))
    dPhase = np.reshape(dPhase, (nPoints[2], nPoints[1], nPoints[0]))
    for ii in range(etl):
        data[:, :, nPoints[0]*ii:nPoints[0]*(ii+1)] = data[:, :, nPoints[0]*ii:nPoints[0]*(ii+1)]*dPhase

    # Get images with FFT
    img = data*0
    for ii in range(etl):
        img[:,:,nPoints[0]*ii:nPoints[0]*(ii+1)] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data[:,:,nPoints[0]*ii:nPoints[0]*(ii+1)])))
    
    # Plot k-space
    plt.figure(2)
    dataPlot = data[round(nSL/2), :, :]
    plt.subplot(211)
    plt.imshow(np.log(np.abs(dataPlot)),cmap='gray')
    plt.axis('off')
    
    # Plot image
    if sweepMode==3:
        imgPlot = img[round(nSL/2), round(nPH/4):round(3*nPH/4), :]
    else:
        imgPlot = img[round(nSL/2)+1, :, :]
    plt.subplot(212)
    plt.imshow(np.abs(imgPlot), cmap='gray')
    plt.axis('off')
    
    # Create sampled data
#    kRD = np.reshape(kRD, (nPoints[0]*nPoints[1]*nPoints[2], 1))
#    kPH = np.reshape(kPH, (nPoints[0]*nPoints[1]*nPoints[2], 1))
#    kSL = np.reshape(kSL, (nPoints[0]*nPoints[1]*nPoints[2], 1))
#    data = np.reshape(data, (nPoints[0]*nPoints[1]*nPoints[2], etl))
#    auxiliar['kMax'] = kMax
    outputs['sampled'] = data
    outputs['imagen'] = img
    
    # Save data
    dt = datetime.now()
    dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
    dt2 = date.today()
    dt2_string = dt2.strftime("%Y.%m.%d")
    if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
        os.makedirs('experiments/acquisitions/%s' % (dt2_string))
            
    if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
        os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
    auxiliar['fileName'] = "%s.%s.mat" % ("RARE",dt_string)
    rawData['inputs'] = inputs
    rawData['auxiliar'] = auxiliar
    rawData['kSpace'] = outputs
    rawdata = {}
    rawdata['rawData'] = rawData
    savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, "Old_RARE",dt_string),  rawdata) 
        
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

    rare2_standalone()
