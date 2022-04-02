# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:05:53 2022
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
from mrilabMethods.mrilabMethods import *   # This import all methods inside the mrilabMethods module
st = pdb.set_trace



#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def haste_standalone(
    init_gpa=False, # Starts the gpa
    nScans = 1, # NEX
    larmorFreq = 3.07449, # MHz, Larmor frequency
    rfExAmp = 0.05, # a.u., rf excitation pulse amplitude
    rfReAmp = 0.1, # a.u., rf refocusing pulse amplitude
    rfExTime = 190, # us, rf excitation pulse time
    rfReTime = 190, # us, rf refocusing pulse time
    echoSpacing = 10., # ms, time between echoes
    inversionTime = 0., # ms, Inversion recovery time
    repetitionTime = 1000., # ms, TR
    fov = np.array([120., 120.]), # mm, FOV along readout, phase and slice
    dfov = np.array([0., 0.]), # mm, displacement of fov center
    nPoints = np.array([60, 20]), # Number of points along readout, phase and slice
    slThickness = 20, # mm, slice thickness
    acqTime = 4, # ms, acquisition time
    axes = np.array([2, 0, 1]), # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    axesEnable = np.array([1, 1]), # 1-> Enable, 0-> Disable
    sweepMode = 1, # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    rdGradTime = 4.5,  # ms, readout gradient time
    rdDephTime = 1,  # ms, readout dephasing time
    phGradTime = 1, # ms, phase and slice dephasing time
    rdPreemphasis = 1.005, # readout dephasing gradient is multiplied by this factor
    ssPreemphasis = 0.4, # ssGradAmplitue is multiplied by this number for rephasing
    crusherDelay = 170,  # us, delay of the crusher gradient
    drfPhase = 0, # degrees, phase of the excitation pulse
    dummyPulses = 0, # number of dummy pulses for T1 stabilization
    shimming = np.array([-70., -90., 10.]), # a.u.*1e4, shimming along the X,Y and Z axes
    parFourierFraction = 0.65 # fraction of acquired k-space along phase direction
    ):
    
    freqCal = 0
    
    # rawData fields
    rawData = {}
    
    # Conversion of variables to non-multiplied units
    larmorFreq = larmorFreq*1e6
    rfExTime = rfExTime*1e-6
    rfReTime = rfReTime*1e-6
    fov = np.array(fov)*1e-3
    dfov = np.array(dfov)*1e-3
    echoSpacing = echoSpacing*1e-3
    acqTime = acqTime*1e-3
    shimming = shimming*1e-4
    repetitionTime= repetitionTime*1e-3
    inversionTime = inversionTime*1e-3
    rdGradTime = rdGradTime*1e-3
    rdDephTime = rdDephTime*1e-3
    phGradTime = phGradTime*1e-3
    slThickness = slThickness*1e-3
    crusherDelay = crusherDelay*1e-6
    
    # Inputs for rawData
    rawData['SeqName'] = 'HASTE'
    rawData['nScans'] = nScans
    rawData['larmorFreq'] = larmorFreq      # Larmor frequency
    rawData['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    rawData['rfReAmp'] = rfReAmp             # rf refocusing pulse amplitude
    rawData['rfExTime'] = rfExTime          # rf excitation pulse time
    rawData['rfReTime'] = rfReTime            # rf refocusing pulse time
    rawData['echoSpacing'] = echoSpacing        # time between echoes
    rawData['inversionTime'] = inversionTime       # Inversion recovery time
    rawData['repetitionTime'] = repetitionTime     # TR
    rawData['fov'] = fov           # FOV along readout, phase and slice
    rawData['dfov'] = dfov            # Displacement of fov center
    rawData['nPoints'] = nPoints                 # Number of points along readout, phase and slice
    rawData['acqTime'] = acqTime             # Acquisition time
    rawData['axesOrientation'] = axes       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    rawData['axesEnable'] = axesEnable # 1-> Enable, 0-> Disable
    rawData['sweepMode'] = sweepMode               # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    rawData['rdPreemphasis'] = rdPreemphasis
    rawData['ssPreemphasis'] = ssPreemphasis
    rawData['drfPhase'] = drfPhase 
    rawData['dummyPulses'] = dummyPulses                    # Dummy pulses for T1 stabilization
    rawData['partialFourierFraction'] = parFourierFraction
    rawData['rdDephTime'] = rdDephTime
    rawData['sliceThickness'] = slThickness
    rawData['crusherDelay'] = crusherDelay
    
    # Miscellaneous
    blkTime = 10             # Deblanking time (us)
    larmorFreq = larmorFreq*1e-6
    gradRiseTime = 400e-6       # Estimated gradient rise time
    gSteps = int(gradRiseTime*1e6/5)*0+1
    gradDelay = 9            # Gradient amplifier delay
    addRdPoints = 10             # Initial rd points to avoid artifact at the begining of rd
    gammaB = 42.56e6            # Gyromagnetic ratio in Hz/T
    oversamplingFactor = 6
    if rfReAmp==0:
        rfReAmp = 2*rfExAmp
    if rfReTime==0:
        rfReTime = rfExTime
    resolution = fov/nPoints
    rawData['resolution'] = resolution
    rawData['gradDelay'] = gradDelay*1e-6
    rawData['gradRiseTime'] = gradRiseTime
    rawData['oversamplingFactor'] = oversamplingFactor
    rawData['addRdPoints'] = addRdPoints
    
    # Matrix size
    nRD = nPoints[0]+2*addRdPoints
    nPH = nPoints[1]*axesEnable[1]+(1-axesEnable[1])
    
    # parAcqLines
    nPHreal = int(nPoints[1]*parFourierFraction)
    parAcqLines = int(nPHreal-nPoints[1]/2)
    rawData['parAcqLines'] = parAcqLines
    print(parAcqLines)
    del nPHreal
    
    # BW
    BW = nPoints[0]/acqTime*1e-6
    BWov = BW*oversamplingFactor
    samplingPeriod = 1/BWov
    
    # Readout gradient time
    if rdGradTime<acqTime:
        rdGradTime = acqTime
    rawData['rdGradTime'] = rdGradTime
    
    # Phase and slice de- and re-phasing time
    if phGradTime==0 or phGradTime>echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime:
        phGradTime = echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime
    rawData['phGradTime'] = phGradTime
    
    # Max redaout and phase gradient amplitude
    rdGradAmplitude = nPoints[0]/(gammaB*fov[0]*acqTime)*axesEnable[0]
    phGradAmplitude = nPH/(2*gammaB*fov[1]*(phGradTime+gradRiseTime))*axesEnable[1]
    rawData['rdGradAmplitude'] = rdGradAmplitude
    rawData['phGradAmplitude'] = phGradAmplitude
    
    # Slice selection gradient
    if slThickness!=0:
        ssGradAmplitude = 1/(gammaB*slThickness*rfExTime)
    else:
        ssGradAmplitude = 0
    rawData['ssGradAmplitude'] = ssGradAmplitude
    
    # Readout dephasing amplitude
    rdDephAmplitude = 0.5*rdGradAmplitude*(gradRiseTime+rdGradTime)/(gradRiseTime+rdDephTime)
    rawData['rdDephAmplitude'] = rdDephAmplitude

    # Get factors to OCRA1 units
    gFactor = reorganizeGfactor(axes)
    rawData['gFactor'] = gFactor
    
    # Phase and slice gradient vector
    phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
    
    # Get phase indexes for the given sweep mode
    ind = getIndex(2*parAcqLines, 2*parAcqLines, sweepMode)
    ind = ind-parAcqLines+int(nPH/2)
    ind = np.int32(np.concatenate((ind, np.linspace(int(nPH/2)-parAcqLines-1, -1, num=int(nPH/2)-parAcqLines, endpoint=False)), axis=0))
    rawData['sweepOrder'] = ind
    
    # Now fix the number of phases to partailly acquired k-space
    nPH = (int(nPoints[1]/2)+parAcqLines)*axesEnable[1]+(1-axesEnable[1])
    phGradients = phGradients[0:nPH]
    phGradients = phGradients[ind]
    rawData['phGradients'] = phGradients
    
    # Change units to OCRA1 board
    rdGradAmplitude = rdGradAmplitude/gFactor[0]*1000/5
    rdDephAmplitude = rdDephAmplitude/gFactor[0]*1000/5
    phGradients = phGradients/gFactor[1]*1000/5
    ssGradAmplitude = ssGradAmplitude/gFactor[2]*1000/5

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
             
    def iniSequence(tEnd):
        expt.add_flodict({
                'grad_vx': (np.array([tEnd]),np.array([shimming[0]]) ), 
                'grad_vy': (np.array([tEnd]),np.array([shimming[1]]) ), 
                'grad_vz': (np.array([tEnd]),np.array([shimming[2]]) ),
             })

    def createSequence():
        nRepetitions = int(1+dummyPulses)
        scanTime = 20e3+nRepetitions*repetitionTime
        rawData['scanTime'] = scanTime*1e-6
        print('Scan Time = ', scanTime*1e-6, ' s')
        if rdGradTime==0:   # Check if readout gradient is dc or pulsed
            dc = True
        else:
            dc = False
        
        # Set shimming
        iniSequence(20)
        for repeIndex in range(nRepetitions):
            # Initialize time
            tEx = 20e3+repetitionTime*repeIndex+inversionTime
            
            # Inversion pulse
            if repeIndex>=dummyPulses and inversionTime!=0:
                t0 = tEx-inversionTime-rfReTime/2-blkTime
                rfPulse(t0,rfReTime,rfReAmp/180*180,0)
                gradTrap(t0+blkTime+rfReTime, inversionTime*0.5, 0.2, axes[0])
                gradTrap(t0+blkTime+rfReTime, inversionTime*0.5, 0.2, axes[1])
                gradTrap(t0+blkTime+rfReTime, inversionTime*0.5, 0.2, axes[2])
            
            # DC radout gradient if desired
            if (repeIndex==0 or repeIndex>=dummyPulses) and dc==True:
                t0 = tEx-rfExTime/2-gradRiseTime-gradDelay
                gradTrap(t0, echoSpacing*(nPH+1), rdGradAmplitude, axes[0])
            
            # Slice selection gradient dephasing
            if (slThickness!=0 and repeIndex>=dummyPulses):
                t0 = tEx-rfExTime/2-gradRiseTime-gradDelay
                gradTrap(t0, rfExTime, ssGradAmplitude, axes[2])
            
            # Excitation pulse
            t0 = tEx-blkTime-rfExTime/2
            rfPulse(t0,rfExTime,rfExAmp,drfPhase*np.pi/180)
        
            # Slice selection gradient rephasing
            if (slThickness!=0 and repeIndex>=dummyPulses):
                t0 = tEx+rfExTime/2+gradRiseTime-gradDelay
                gradTrap(t0, 0., -ssGradAmplitude*ssPreemphasis, axes[2])
        
            # Dephasing readout
            t0 = tEx+rfExTime/2-gradDelay
            if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False:
                gradTrap(t0, rdDephTime, rdDephAmplitude*rdPreemphasis, axes[0])
            
            # Echo train
            for echoIndex in range(nPH):
                tEcho = tEx+echoSpacing*(echoIndex+1)
                
                # Crusher gradient
                if repeIndex>=dummyPulses:
                    t0 = tEcho-echoSpacing/2-rfReTime/2-gradRiseTime-gradDelay+crusherDelay
                    gradTrap(t0, rfReTime, ssGradAmplitude, axes[2])
                
                # Refocusing pulse
                t0 = tEcho-echoSpacing/2-rfReTime/2-blkTime
                rfPulse(t0, rfReTime, rfReAmp, np.pi/2+drfPhase*np.pi/180)
                
                # Dephasing phase gradient
                t0 = tEcho-echoSpacing/2+rfReTime/2-gradDelay
                if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                    gradTrap(t0, phGradTime, phGradients[echoIndex], axes[1])
                    
                # Readout gradient
                t0 = tEcho-rdGradTime/2-gradRiseTime-gradDelay
                if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False:         # This is to account for dummy pulses
                    gradTrap(t0, rdGradTime, rdGradAmplitude, axes[0])
    
                # Rx gate
                if (repeIndex==0 or repeIndex>=dummyPulses):
                    t0 = tEcho-acqTime/2-addRdPoints/BW
                    rxGate(t0, acqTime+2*addRdPoints/BW)
    
                # Rephasing phase and slice gradients
                t0 = tEcho+acqTime/2+addRdPoints/BW-gradDelay
                if (echoIndex<nPH-1 and repeIndex>=dummyPulses):
                    gradTrap(t0, phGradTime, -phGradients[echoIndex], axes[1])
                elif(echoIndex==nPH-1 and repeIndex>=dummyPulses):
                    gradTrap(t0, phGradTime, +phGradients[echoIndex], axes[1])

            if repeIndex==nRepetitions-1:
                endSequence(scanTime)
    
    
    def createFreqCalSequence():
        t0 = 20
        
        # Shimming
        iniSequence(t0)
            
        # Excitation pulse
        rfPulse(t0,rfExTime,rfExAmp,drfPhase*np.pi/180)
        
        # Refocusing pulse
        t0 += rfExTime/2+echoSpacing/2-rfReTime/2
        rfPulse(t0, rfReTime, rfReAmp, np.pi/2)
        
        # Rx
        t0 += blkTime+rfReTime/2+echoSpacing/2-acqTime/2-addRdPoints/BW
        rxGate(t0, acqTime+2*addRdPoints/BW)
        
        # Finalize sequence
        endSequence(repetitionTime)
        
    
    # Changing time parameters to us
    rfExTime = rfExTime*1e6
    rfReTime = rfReTime*1e6
    echoSpacing = echoSpacing*1e6
    repetitionTime = repetitionTime*1e6
    gradRiseTime = gradRiseTime*1e6
    phGradTime = phGradTime*1e6
    rdGradTime = rdGradTime*1e6
    rdDephTime = rdDephTime*1e6
    inversionTime = inversionTime*1e6
    crusherDelay = crusherDelay*1e6
    
    # Calibrate frequency
    if freqCal==1:
        expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = expt.get_rx_ts()[0]
        BW = 1/samplingPeriod/oversamplingFactor
        acqTime = nPoints[0]/BW        # us
        rawData['bw'] = BW*1e6
        createFreqCalSequence()
        rxd, msgs = expt.run()
        dataFreqCal = sig.decimate(rxd['rx0']*13.788, oversamplingFactor, ftype='fir', zero_phase=True)
        dataFreqCal = dataFreqCal[addRdPoints:nPoints[0]+addRdPoints]
        # Plot fid
    #    plt.figure(1)
        tVector = np.linspace(-acqTime/2, acqTime/2, num=nPoints[0],endpoint=True)*1e-3
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
    
    # Create full sequence
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BW = 1/samplingPeriod/oversamplingFactor
    acqTime = nPoints[0]/BW        # us
    createSequence()

    # Plot sequence:
#    expt.plot_sequence()
        
    # Run the experiment
    dataFull = []
    dummyData = []
    overData = []
    for ii in range(nScans):
        print("Scan %s ..." % (ii+1))
        rxd, msgs = expt.run()
        rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
        # Get data
        if dummyPulses>0:
            dummyData = np.concatenate((dummyData, rxd['rx0'][0:nRD*nPH*oversamplingFactor]), axis = 0)
            overData = np.concatenate((overData, rxd['rx0'][nRD*nPH*oversamplingFactor::]), axis = 0)
        else:
            overData = np.concatenate((overData, rxd['rx0']), axis = 0)
    expt.__del__()
    print('Scans done!')
    rawData['overData'] = overData
    
    # Fix the echo position using oversampled data
    if dummyPulses>0:
        dummyData = np.reshape(dummyData,  (nScans, nPH, nRD*oversamplingFactor))
        dummyData = np.average(dummyData, axis=0)
        rawData['dummyData'] = dummyData
        overData = np.reshape(overData, (nScans, 1, nPH,  nRD*oversamplingFactor))
        for ii in range(nScans):
            overData[ii, :, :, :] = fixEchoPosition(dummyData, overData[ii, :, :, :])
        
    # Generate dataFull
    overData = np.squeeze(np.reshape(overData, (1, nRD*oversamplingFactor*nPH*nScans)))
    dataFull = sig.decimate(overData, oversamplingFactor, ftype='fir', zero_phase=True)
    
    # Get index for krd = 0
    # Average data
    dataProv = np.reshape(dataFull, (nScans, nRD*nPH))
    dataProv = np.average(dataProv, axis=0)
    # Reorganize the data acording to sweep mode
    dataProv = np.reshape(dataProv, (nPH, nRD))
    dataTemp = dataProv*0
    for ii in range(nPH):
        dataTemp[ind[ii], :] = dataProv[ii, :]
    dataProv = dataTemp
    # Check where is krd = 0
    dataProv = dataProv[int(nPoints[1]/2), :]
    indkrd0 = np.argmax(np.abs(dataProv))
    if  indkrd0 < nRD/2-addRdPoints or indkrd0 > nRD+addRdPoints:
        indkrd0 = int(nRD/2)

    # Get individual images
    dataFull = np.reshape(dataFull, (nScans, nPH, nRD))
    dataFull = dataFull[:, :, indkrd0-int(nPoints[0]/2):indkrd0+int(nPoints[0]/2)]
    dataTemp = dataFull*0
    for ii in range(nPH):
        dataTemp[:, ind[ii], :] = dataFull[:, ii, :]
    dataFull = dataTemp
    imgFull = dataFull*0
    for ii in range(nScans):
        imgFull[ii, :, :] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataFull[ii, :, :])))
    rawData['dataFull'] = dataFull
    rawData['imgFull'] = imgFull    
    
    # Average data
    data = np.average(dataFull, axis=0)
    data = np.reshape(data, (nPH, nPoints[0]))
    
    # Do zero padding
    dataTemp = np.zeros((nPoints[1], nPoints[0]))
    dataTemp = dataTemp+1j*dataTemp
    dataTemp[0:nPH, :] = data
    data = np.reshape(dataTemp, (1, nPoints[0]*nPoints[1]))
    
    # Fix the position of the sample according to dfov
    kMax = np.array(nPoints)/(2*np.array(fov))*np.array(axesEnable)
    kRD = np.linspace(-kMax[0],kMax[0],num=nPoints[0],endpoint=False)
    kPH = np.linspace(-kMax[1],kMax[1],num=nPoints[1],endpoint=False)
    kPH, kRD = np.meshgrid(kPH, kRD)
    kRD = np.reshape(kRD, (1, nPoints[0]*nPoints[1]))
    kPH = np.reshape(kPH, (1, nPoints[0]*nPoints[1]))
    dPhase = np.exp(-2*np.pi*1j*(dfov[0]*kRD+dfov[1]*kPH))
    data = np.reshape(data*dPhase, (nPoints[1], nPoints[0]))
    rawData['kSpace3D'] = data
    img=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
    rawData['image3D'] = img
    data = np.reshape(data, (1, nPoints[0]*nPoints[1]))
    
    # Create sampled data
    kRD = np.reshape(kRD, (nPoints[0]*nPoints[1], 1))
    kPH = np.reshape(kPH, (nPoints[0]*nPoints[1], 1))
    data = np.reshape(data, (nPoints[0]*nPoints[1], 1))
    rawData['kMax'] = kMax
    rawData['sampled'] = np.concatenate((kRD, kPH, data), axis=1)
    data = np.reshape(data, (nPoints[1], nPoints[0]))
    
    # Save data
    dt = datetime.now()
    dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
    dt2 = date.today()
    dt2_string = dt2.strftime("%Y.%m.%d")
    if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
        os.makedirs('experiments/acquisitions/%s' % (dt2_string))
            
    if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
        os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
    rawData['fileName'] = "%s.%s.mat" % ("RARE",dt_string)
    savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, "Old_RARE",dt_string),  rawData) 
        
    # Plot data for 1D case
    if (nPH==1 and nSL==1):
        # Plot k-space
        plt.figure(3)
        dataPlot = data[0, :]
        plt.subplot(1, 2, 1)
        if axesEnable[0]==0:
            tVector = np.linspace(-acqTime/2, acqTime/2, num=nPoints[0],endpoint=False)*1e-3
            sMax = np.max(np.abs(dataPlot))
            indMax = np.argmax(np.abs(dataPlot))
            timeMax = tVector[indMax]
            sMax3 = sMax/3
            dataPlot3 = np.abs(np.abs(dataPlot)-sMax3)
            indMin = np.argmin(dataPlot3)
            timeMin = tVector[indMin]
            T2 = np.abs(timeMax-timeMin)
            plt.plot(tVector, np.abs(dataPlot))
            plt.plot(tVector, np.real(dataPlot))
            plt.plot(tVector, np.imag(dataPlot))
            plt.xlabel('t (ms)')
            plt.ylabel('Signal (mV)')
            print("T2 = %s us" % (T2))
        else:
            plt.plot(kRD[:, 0], np.abs(dataPlot))
            plt.yscale('log')
            plt.xlabel('krd (mm^-1)')
            plt.ylabel('Signal (mV)')
            echoTime = np.argmax(np.abs(dataPlot))
            echoTime = kRD[echoTime, 0]
            print("Echo position = %s mm^{-1}" %round(echoTime, 1))
        # Plot image
        plt.subplot(122)
        img = img[0, :]
        if axesEnable[0]==0:
            xAxis = np.linspace(-BW/2, BW/2, num=nPoints[0], endpoint=False)*1e3
            plt.plot(xAxis, np.abs(img), '.')
            plt.xlabel('Frequency (kHz)')
            plt.ylabel('Density (a.u.)')
            print("Smax = %s mV" % (np.max(np.abs(img))))
        else:
            xAxis = np.linspace(-fov[0]/2*1e2, fov[0]/2*1e2, num=nPoints[0], endpoint=False)
            plt.plot(xAxis, np.abs(img))
            plt.xlabel('Position RD (cm)')
            plt.ylabel('Density (a.u.)')
    
    # Plot data for 2D case
    else:
        # Plot k-space
        plt.figure(3)
        dataPlot = data
        plt.subplot(131)
        plt.imshow(np.log(np.abs(dataPlot)),cmap='gray')
        plt.axis('off')
        # Plot image
        if sweepMode==3:
            imgPlot = img[round(nPH/4):round(3*nPH/4), :]
        else:
            imgPlot = img
        plt.subplot(132)
        plt.imshow(np.abs(imgPlot), cmap='gray')
        plt.axis('off')
        plt.title("RARE.%s.mat" % (dt_string))
        plt.subplot(133)
        plt.imshow(np.angle(imgPlot), cmap='gray')
        plt.axis('off')
        
    plt.figure(5)
    plt.subplot(121)
    data1d = data[:, int(nPoints[0]/2)]
    plt.plot(abs(data1d))
    plt.subplot(122)
    img1d = img[:, int(nPoints[0]/2)]
    plt.plot(np.abs(img1d)*1e3)
    
    plt.show()


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


if __name__ == "__main__":

    haste_standalone()
