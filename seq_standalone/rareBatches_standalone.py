# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 12:40:05 2021
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@Summary: this code used rare_standalone.py on Feb 4 2022 as source. It divide the 3d acquisition
in batches in such a way that each batch acquires a k-space slice.
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
from mrilabMethods.mrilabMethods import *
st = pdb.set_trace



#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def rare_standalone(
    init_gpa=False, # Starts the gpa
    nScans = 1, # NEX
    larmorFreq = 3.0743, # MHz, Larmor frequency
    rfExAmp = 0.4, # a.u., rf excitation pulse amplitude
    rfReAmp = 0.4, # a.u., rf refocusing pulse amplitude
    rfExTime = 25, # us, rf excitation pulse time
    rfReTime = 50, # us, rf refocusing pulse time
    echoSpacing = 10., # ms, time between echoes
    preExTime = 0., # ms, Time from preexcitation pulse to inversion pulse
    inversionTime = 0., # ms, Inversion recovery time
    repetitionTime = 1000., # ms, TR
    fov = np.array([120., 120., 40.]), # mm, FOV along readout, phase and slice
    dfov = np.array([0., 0., 0.]), # mm, displacement of fov center
    nPoints = np.array([60, 60, 4]), # Number of points along readout, phase and slice
    etl = 60, # Echo train length
    acqTime = 4, # ms, acquisition time
    axes = np.array([2, 0, 1]), # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    axesEnable = np.array([1, 1, 1]), # 1-> Enable, 0-> Disable
    sweepMode = 1, # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    rdGradTime = 6,  # ms, readout gradient time
    rdDephTime = 1,  # ms, readout dephasing time
    phGradTime = 1, # ms, phase and slice dephasing time
    rdPreemphasis = 1.005, # readout dephasing gradient is multiplied by this factor
    drfPhase = 0, # degrees, phase of the excitation pulse
    dummyPulses = 1, # number of dummy pulses for T1 stabilization
    shimming = np.array([-70., -90., 10.]), # a.u.*1e4, shimming along the X,Y and Z axes
    parAcqLines = 0 # number of additional lines, Full sweep if 0
    ):
    
    freqCal = 1
    
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
    preExTime = preExTime*1e-3
    inversionTime = inversionTime*1e-3
    rdGradTime = rdGradTime*1e-3
    rdDephTime = rdDephTime*1e-3
    phGradTime = phGradTime*1e-3
    
    # Inputs for rawData
    rawData['nScans'] = nScans
    rawData['larmorFreq'] = larmorFreq      # Larmor frequency
    rawData['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    rawData['rfReAmp'] = rfReAmp             # rf refocusing pulse amplitude
    rawData['rfExTime'] = rfExTime          # rf excitation pulse time
    rawData['rfReTime'] = rfReTime            # rf refocusing pulse time
    rawData['echoSpacing'] = echoSpacing        # time between echoes
    rawData['preExTime'] = preExTime
    rawData['inversionTime'] = inversionTime       # Inversion recovery time
    rawData['repetitionTime'] = repetitionTime     # TR
    rawData['fov'] = fov           # FOV along readout, phase and slice
    rawData['dfov'] = dfov            # Displacement of fov center
    rawData['nPoints'] = nPoints                 # Number of points along readout, phase and slice
    rawData['etl'] = etl                    # Echo train length
    rawData['acqTime'] = acqTime             # Acquisition time
    rawData['axesOrientation'] = axes       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    rawData['axesEnable'] = axesEnable # 1-> Enable, 0-> Disable
    rawData['sweepMode'] = sweepMode               # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    rawData['rdPreemphasis'] = rdPreemphasis
    rawData['drfPhase'] = drfPhase 
    rawData['dummyPulses'] = dummyPulses                    # Dummy pulses for T1 stabilization
    rawData['partialAcquisition'] = parAcqLines
    rawData['rdDephTime'] = rdDephTime
    
    # Miscellaneous
    blkTime = 10             # Deblanking time (us)
    larmorFreq = larmorFreq*1e-6
    gradRiseTime = 400e-6       # Estimated gradient rise time
    gSteps = int(gradRiseTime*1e6/5)*0+1
    gradDelay = 9            # Gradient amplifier delay
    addRdPoints = 10             # Initial rd points to avoid artifact at the begining of rd
    gammaB = 42.56e6            # Gyromagnetic ratio in Hz/T
    oversamplingFactor = 6
    randFactor = 0e-3                        # Random amplitude to add to the phase gradients
    if rfReAmp==0:
        rfReAmp = rfExAmp
    if rfReTime==0:
        rfReTime = 2*rfExTime
    resolution = fov/nPoints
    rawData['resolution'] = resolution
    rawData['gradDelay'] = gradDelay*1e-6
    rawData['gradRiseTime'] = gradRiseTime
    rawData['oversamplingFactor'] = oversamplingFactor
    rawData['randFactor'] = randFactor
    rawData['addRdPoints'] = addRdPoints
    
    # Matrix size
    nRD = nPoints[0]+2*addRdPoints
    nPH = nPoints[1]*axesEnable[1]+(1-axesEnable[1])
    nSL = nPoints[2]*axesEnable[2]+(1-axesEnable[2])
    
    # ETL if nPH = 1
    if etl>nPH:
        etl = nPH
    
    # parAcqLines in case parAcqLines = 0
    if parAcqLines==0:
        parAcqLines = int(nSL/2)
    
    # BW
    BW = nPoints[0]/acqTime*1e-6
    BWov = BW*oversamplingFactor
    samplingPeriod = 1/BWov
    
    # Readout gradient time
    if rdGradTime>0 and rdGradTime<acqTime:
        rdGradTime = acqTime
    rawData['rdGradTime'] = rdGradTime
    
    # Phase and slice de- and re-phasing time
    if phGradTime==0 or phGradTime>echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime:
        phGradTime = echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime
    rawData['phGradTime'] = phGradTime
    
    # Max gradient amplitude
    rdGradAmplitude = nPoints[0]/(gammaB*fov[0]*acqTime)*axesEnable[0]
    phGradAmplitude = nPH/(2*gammaB*fov[1]*(phGradTime+gradRiseTime))*axesEnable[1]
    slGradAmplitude = nSL/(2*gammaB*fov[2]*(phGradTime+gradRiseTime))*axesEnable[2]
    rawData['rdGradAmplitude'] = rdGradAmplitude
    rawData['phGradAmplitude'] = phGradAmplitude
    rawData['slGradAmplitude'] = slGradAmplitude

    # Readout dephasing amplitude
    rdDephAmplitude = 0.5*rdGradAmplitude*(gradRiseTime+rdGradTime)/(gradRiseTime+rdDephTime)
    rawData['rdDephAmplitude'] = rdDephAmplitude

    # Get factors to OCRA1 units
    gFactor = reorganizeGfactor(axes)
    rawData['gFactor'] = gFactor
    
    # Phase and slice gradient vector
    phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
    slGradients = np.linspace(-slGradAmplitude,slGradAmplitude,num=nSL,endpoint=False)
    
    # Now fix the number of slices to partailly acquired k-space
    nSL = (int(nPoints[2]/2)+parAcqLines)*axesEnable[2]+(1-axesEnable[2])
    
    # Add random displacemnt to phase encoding lines
    for ii in range(nPH):
        if ii<np.ceil(nPH/2-nPH/20) or ii>np.ceil(nPH/2+nPH/20):
            phGradients[ii] = phGradients[ii]+randFactor*np.random.randn()
    kPH = gammaB*phGradients*(gradRiseTime+phGradTime)
    rawData['phGradients'] = phGradients
    rawData['slGradients'] = slGradients
    
    # Change units to OCRA1 board
    rdGradAmplitude = rdGradAmplitude/gFactor[0]*1000/5
    rdDephAmplitude = rdDephAmplitude/gFactor[0]*1000/5
    phGradients = phGradients/gFactor[1]*1000/5
    slGradients = slGradients/gFactor[2]*1000/5
    
    # Set phase vector to given sweep mode
    ind = getIndex(etl, nPH, sweepMode)
    rawData['sweepOrder'] = ind
    phGradients = phGradients[ind]

    def createSequence(rewrite=True):
        phIndex = 0
        nRepetitions = int(nPH/etl+dummyPulses)
        scanTime = 20e3+nRepetitions*repetitionTime
        rawData['scanTime'] = scanTime*nSL*1e-6
        if rdGradTime==0:   # Check if readout gradient is dc or pulsed
            dc = True
        else:
            dc = False
        # Set shimming
        if rewrite==True:
            setGradient(expt, t0=20, gAmp=shimming[axes[0]], gAxis=axes[0])
            setGradient(expt, t0=20, gAmp=shimming[axes[1]], gAxis=axes[1])
        setGradient(expt, t0=20, gAmp=shimming[axes[2]], gAxis=axes[2], rewrite=rewrite)
        for repeIndex in range(nRepetitions):
            # Initialize time
            tEx = 20e3+repetitionTime*repeIndex+inversionTime+preExTime
            
            # Pre-excitation pulse
            if repeIndex>=dummyPulses and preExTime!=0:
                t0 = tEx-preExTime-inversionTime-rfExTime/2-blkTime
                if rewrite==True:
                    rfRecPulse(expt, t0, rfExTime, rfExAmp/90*90, 0)
                    gradTrap(expt, t0+blkTime+rfReTime, gradRiseTime, preExTime*0.5, -0.2, gSteps, axes[0], shimming)
                    gradTrap(expt, t0+blkTime+rfReTime, gradRiseTime, preExTime*0.5, -0.2, gSteps, axes[1], shimming)
                gradTrap(expt, t0+blkTime+rfReTime, gradRiseTime, preExTime*0.5, -0.2, gSteps, axes[2], shimming)
                
            # Inversion pulse
            if repeIndex>=dummyPulses and inversionTime!=0:
                t0 = tEx-inversionTime-rfReTime/2-blkTime
                if rewrite==True:
                    rfPulse(expt, t0, rfReTime, rfReAmp/180*180, 0)
                    gradTrap(expt, t0+blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.2, gSteps, axes[0], shimming)
                    gradTrap(expt, t0+blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.2, gSteps, axes[1], shimming)
                gradTrap(expt, t0+blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.2, gSteps, axes[2], shimming)
            
            # DC gradient if desired
            if (repeIndex==0 or repeIndex>=dummyPulses) and dc==True and rewrite==True:
                t0 = tEx-10e3
                gradTrap(expt, t0, gradRiseTime, 10e3+echoSpacing*(etl+1), rdGradAmplitude, gSteps, axes[0], shimming)
            
            # Excitation pulse
            if rewrite==True:
                t0 = tEx-blkTime-rfExTime/2
                rfRecPulse(expt, t0,rfExTime,rfExAmp,drfPhase*np.pi/180)
        
            # Dephasing readout
            if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False and rewrite==True:
                t0 = tEx+rfExTime/2-gradDelay
                gradTrap(expt, t0, gradRiseTime, rdDephTime, rdDephAmplitude*rdPreemphasis, gSteps, axes[0], shimming)
            
            # Echo train
            for echoIndex in range(etl):
                tEcho = tEx+echoSpacing*(echoIndex+1)
                
                # Refocusing pulse
                if rewrite==True:
                    t0 = tEcho-echoSpacing/2-rfReTime/2-blkTime
                    rfRecPulse(expt, t0, rfReTime, rfReAmp, np.pi/2)
    
                # Dephasing phase and slice gradients
                t0 = tEcho-echoSpacing/2+rfReTime/2-gradDelay
                if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                    if rewrite==True:
                        gradTrap(expt, t0, gradRiseTime, phGradTime, phGradients[phIndex], gSteps, axes[1], shimming)
                    gradTrap(expt, t0, gradRiseTime, phGradTime, slGradients[slIndex], gSteps, axes[2], shimming)
                
                # Readout gradient
                t0 = tEcho-rdGradTime/2-gradRiseTime-gradDelay
                if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False and rewrite==True:         # This is to account for dummy pulses
                    gradTrap(expt, t0, gradRiseTime, rdGradTime, rdGradAmplitude, gSteps, axes[0], shimming)
    
                # Rx gate
                if (repeIndex==0 or repeIndex>=dummyPulses) and rewrite==True:
                    t0 = tEcho-acqTime/2-addRdPoints/BW
                    rxGate(expt, t0, acqTime+2*addRdPoints/BW)
    
                # Rephasing phase and slice gradients
                t0 = tEcho+acqTime/2+addRdPoints/BW-gradDelay
                if (echoIndex<etl-1 and repeIndex>=dummyPulses):
                    if rewrite==True:
                        gradTrap(expt, t0, gradRiseTime, phGradTime, -phGradients[phIndex], gSteps, axes[1], shimming)
                    gradTrap(expt, t0, gradRiseTime, phGradTime, -slGradients[slIndex], gSteps, axes[2], shimming)
                elif(echoIndex==etl-1 and repeIndex>=dummyPulses):
                    if rewrite==True:
                        gradTrap(expt, t0, gradRiseTime, phGradTime, +phGradients[phIndex], gSteps, axes[1], shimming)
                    gradTrap(expt, t0, gradRiseTime, phGradTime, +slGradients[slIndex], gSteps, axes[2], shimming)
    
                # Update the phase and slice gradient
                if repeIndex>=dummyPulses:
                    phIndex += 1
                
            if repeIndex==nRepetitions-1:
                if rewrite==True:
                    setGradient(expt, scanTime, 0, axes[0])
                    setGradient(expt, scanTime, 0, axes[1])
                setGradient(expt, scanTime, 0, axes[2])
    
    
    def createFreqCalSequence():
        t0 = 20
        
        # Shimming
        iniSequence(expt, t0, shimming)
            
        # Excitation pulse
        rfRecPulse(expt, t0,rfExTime,rfExAmp,drfPhase*np.pi/180)
        
        # Refocusing pulse
        t0 += rfExTime/2+echoSpacing/2-rfReTime/2
        rfRecPulse(expt, t0, rfReTime, rfReAmp, np.pi/2)
        
        # Rx
        t0 += blkTime+rfReTime/2+echoSpacing/2-acqTime/2-addRdPoints/BW
        rxGate(expt, t0, acqTime+2*addRdPoints/BW)
        
        # Finalize sequence
        endSequence(expt, repetitionTime)
        
    
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
    preExTime = preExTime*1e6
    
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
    # Run the experiment
    dataFull = []
    dummyData = []
    overData = []
    for slIndex in range(nSL):
        if slIndex==0:
            createSequence()
        else:
            createSequence(False)
        # Plot sequence:
#        expt.plot_sequence()
        
        for ii in range(nScans):
            print("Scan %s ..." % (ii+1))
            rxd, msgs = expt.run()
            rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
            # Get data
            if dummyPulses>0:
                dummyData = np.concatenate((dummyData, rxd['rx0'][0:nRD*etl*oversamplingFactor]), axis = 0)
                overData = np.concatenate((overData, rxd['rx0'][nRD*etl*oversamplingFactor::]), axis = 0)
            else:
                overData = np.concatenate((overData, rxd['rx0']), axis = 0)
    expt.__del__()
    print('Scans done!')
    rawData['overData'] = overData
    
    # Fix the echo position using oversampled data
    if dummyPulses>0:
        dummyData = np.reshape(dummyData,  (nSL*nScans, etl, nRD*oversamplingFactor))
        dummyData = np.average(dummyData, axis=0)
        rawData['dummyData'] = dummyData
        overData = np.reshape(overData, (nScans*nSL, int(nPH/etl), etl,  nRD*oversamplingFactor))
        for ii in range(nScans):
            overData[ii, :, :, :] = fixEchoPosition(dummyData, overData[ii, :, :, :])
        overData = np.squeeze(np.reshape(overData, (1, nRD*oversamplingFactor*nPH*nSL*nScans)))
    
    # Generate dataFull
    dataFull = sig.decimate(overData, oversamplingFactor, ftype='fir', zero_phase=True)
    
    # Get index for krd = 0
    # Average data
    dataProv = np.reshape(dataFull, (nSL, nScans, nRD*nPH))
    dataProv = np.average(dataProv, axis=1)
    # Reorganize the data acording to sweep mode
    dataProv = np.reshape(dataProv, (nSL, nPH, nRD))
    dataTemp = dataProv*0
    for ii in range(nPH):
        dataTemp[:, ind[ii], :] = dataProv[:,  ii, :]
    dataProv = dataTemp
    # Check where is krd = 0
    dataProv = dataProv[int(nPoints[2]/2), int(nPH/2), :]
    indkrd0 = np.argmax(np.abs(dataProv))
    if  indkrd0 < nRD/2-addRdPoints or indkrd0 > nRD+addRdPoints:
        indkrd0 = int(nRD/2)
    indkrd0 = int(nRD/2)

    # Get individual images
    dataFull = np.reshape(dataFull, (nSL, nScans, nPH, nRD))
    dataFull = dataFull[:, :, :, indkrd0-int(nPoints[0]/2):indkrd0+int(nPoints[0]/2)]
    dataTemp = dataFull*0
    for ii in range(nPH):
        dataTemp[:, :, ind[ii], :] = dataFull[:, :,  ii, :]
    dataFull = dataTemp
    imgFull = dataFull*0
    for ii in range(nScans):
        imgFull[:, ii, :, :] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataFull[:, ii, :, :])))
    rawData['dataFull'] = dataFull
    rawData['imgFull'] = imgFull    
    
    # Average data
    data = np.average(dataFull, axis=1)
    data = np.reshape(data, (nSL, nPH, nPoints[0]))
    
    # Do zero padding
    dataTemp = np.zeros((nPoints[2], nPoints[1], nPoints[0]))
    dataTemp = dataTemp+1j*dataTemp
    if nSL==1 or (nSL>1 and parAcqLines==0):
        dataTemp = data
    elif nSL>1 and parAcqLines>0:
        dataTemp[0:nSL-1, :, :] = data[0:nSL-1, :, :]
    data = np.reshape(dataTemp, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    
    # Fix the position of the sample according to dfov
    kMax = np.array(nPoints)/(2*np.array(fov))*np.array(axesEnable)
    kRD = np.linspace(-kMax[0],kMax[0],num=nPoints[0],endpoint=False)
#        kPH = np.linspace(-kMax[1],kMax[1],num=nPoints[1],endpoint=False)
    kSL = np.linspace(-kMax[2],kMax[2],num=nPoints[2],endpoint=False)
    kPH = kPH[::-1]
    kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
    kRD = np.reshape(kRD, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    kPH = np.reshape(kPH, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    kSL = np.reshape(kSL, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    dPhase = np.exp(-2*np.pi*1j*(dfov[0]*kRD+dfov[1]*kPH+dfov[2]*kSL))
    data = np.reshape(data*dPhase, (nPoints[2], nPoints[1], nPoints[0]))
    rawData['kSpace3D'] = data
    img=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
    rawData['image3D'] = img
    data = np.reshape(data, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    
    # Create sampled data
    kRD = np.reshape(kRD, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    kPH = np.reshape(kPH, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    kSL = np.reshape(kSL, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    data = np.reshape(data, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    rawData['kMax'] = kMax
    rawData['sampled'] = np.concatenate((kRD, kPH, kSL, data), axis=1)
    data = np.reshape(data, (nPoints[2], nPoints[1], nPoints[0]))
    
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
        dataPlot = data[0, 0, :]
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
        img = img[0, 0, :]
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
    else:
        # Plot k-space
        plt.figure(3)
        dataPlot = data[round(nSL/2), :, :]
        plt.subplot(131)
        plt.imshow(np.log(np.abs(dataPlot)),cmap='gray')
        plt.axis('off')
        # Plot image
        if sweepMode==3:
            imgPlot = img[round(nSL/2), round(nPH/4):round(3*nPH/4), :]
        else:
            imgPlot = img[round(nSL/2), :, :]
        plt.subplot(132)
        plt.imshow(np.abs(imgPlot), cmap='gray')
        plt.axis('off')
        plt.title("RARE.%s.mat" % (dt_string))
        plt.subplot(133)
        plt.imshow(np.angle(imgPlot), cmap='gray')
        plt.axis('off')
        
    # plot full image
    if nSL>1:
        plt.figure(4)
        img2d = np.zeros((nPoints[1], nPoints[0]*nPoints[2]))
        img2d = img2d+1j*img2d
        for ii in range(nPoints[2]):
            img2d[:, ii*nPoints[0]:(ii+1)*nPoints[0]] = img[ii, :, :]
        plt.imshow(np.abs(img2d), cmap='gray')
        plt.axis('off')
        plt.title("RARE.%s.mat" % (dt_string))
    
    plt.show()
    

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


#def getIndex(echos_per_tr, n_ph, sweep_mode):
#    n2ETL=int(n_ph/2/echos_per_tr)
#    ind:int = [];
#    if n_ph==1:
#         ind = np.linspace(int(n_ph)-1, 0, n_ph)
#    
#    else: 
#        if sweep_mode==0:   # Sequential for T2 contrast
#            for ii in range(int(n_ph/echos_per_tr)):
#               ind = np.concatenate((ind, np.arange(1, n_ph+1, n_ph/echos_per_tr)+ii))
#            ind = ind-1
#
#        elif sweep_mode==1: # Center-out for T1 contrast
#            if echos_per_tr==n_ph:
#                for ii in range(int(n_ph/2)):
#                    cont = 2*ii
#                    ind = np.concatenate((ind, np.array([n_ph/2-cont/2])), axis=0);
#                    ind = np.concatenate((ind, np.array([n_ph/2+1+cont/2])), axis=0);
#            else:
#                for ii in range(n2ETL):
#                    ind = np.concatenate((ind,np.arange(n_ph/2, 0, -n2ETL)-(ii)), axis=0);
#                    ind = np.concatenate((ind,np.arange(n_ph/2+1, n_ph+1, n2ETL)+(ii)), axis=0);
#            ind = ind-1
#        elif sweep_mode==2: # Out-to-center for T2 contrast
#            if echos_per_tr==n_ph:
#                ind=np.arange(1, n_ph+1, 1)
#            else:
#                for ii in range(n2ETL):
#                    ind = np.concatenate((ind,np.arange(1, n_ph/2+1, n2ETL)+(ii)), axis=0);
#                    ind = np.concatenate((ind,np.arange(n_ph, n_ph/2, -n2ETL)-(ii)), axis=0);
#            ind = ind-1
#        elif sweep_mode==3:
#            if echos_per_tr==n_ph:
#                ind = np.arange(0, n_ph, 1)
#            else:
#                for ii in range(int(n2ETL)):
#                    ind = np.concatenate((ind, np.arange(0, n_ph, 2*n2ETL)+2*ii), axis=0)
#                    ind = np.concatenate((ind, np.arange(n_ph-1, 0, -2*n2ETL)-2*ii), axis=0)
#
#    return np.int32(ind)


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


#def reorganizeGfactor(axes):
#    gFactor = np.array([0., 0., 0.])
#    
#    # Set the normalization factor for readout, phase and slice gradient
#    for ii in range(3):
#        if axes[ii]==0:
#            gFactor[ii] = Gx_factor
#        elif axes[ii]==1:
#            gFactor[ii] = Gy_factor
#        elif axes[ii]==2:
#            gFactor[ii] = Gz_factor
#    
#    return(gFactor)

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


#def fixEchoPosition(echoes, data0):
#    etl = np.size(echoes, axis=0)
#    n = np.size(echoes, axis=1)
#    idx = np.argmax(np.abs(echoes), axis=1)
#    idx = idx-int(n/2)
#    data1 = data0*0
#    for ii in range(etl):
#        if idx[ii]>0:
#            idx[ii] = 0
#        echoes[ii, -idx[ii]::] = echoes[ii, 0:n+idx[ii]]
#        data1[:, ii, -idx[ii]::] = data0[:, ii, 0:n+idx[ii]]
##    plt.figure(5)
##    plt.imshow(np.abs(echoes), cmap='gray')
##    plt.show()
#    return(data1)


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


if __name__ == "__main__":

    rare_standalone()
