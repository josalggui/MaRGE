# -*- coding: utf-8 -*-
"""
Created on Sat Nov  13 13:45:05 2021

@author: José Miguel Algarín Guisado
MRILAB @ I3M
"""

import sys
import os
#******************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        # sys.path.append(path[0:ii])
        print("Path: ",path[0:ii+1])
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************
import numpy as np
import experiment as ex
import matplotlib.pyplot as plt
import scipy.signal as sig
import os
import pdb
import configs.hw_config as hw # Import the scanner hardware config
import mrilabMethods.mrilabMethods as mri # This import all methods inside the mrilabMethods module
st = pdb.set_trace



#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def gre_standalone(
    init_gpa=False, # Starts the gpa
    nScans = 1, # NEX
    larmorFreq = 3.078, # MHz, Larmor frequency
    rfExAmp = 0.2, # a.u. rf excitation pulse amplitude
    rfExTime = 35., # us, rf excitation pulse time
    echoTime = 2.5, # ms, TE
    repetitionTime = 200., # ms, TR
    fov = np.array([120., 120., 120.]), # mm, FOV along readout, phase and slice
    dfov = np.array([0., 0., 0.]), # mm, Displacement of fov center
    nPoints = np.array([60, 60, 1]), # Number of points along readout, phase and slice
    acqTime = 1., # ms, Acquisition time
    axes = np.array([1, 0, 2]), # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    rdGradTime = 1.1, # ms, readout rephasing gradient time
    dephGradTime = 1.0, # ms, Phase and slice dephasing time
    rdPreemphasis = 1.00, # Preemphasis factor for readout dephasing
    drfPhase = 0., # degrees, phase of the excitation pulse
    dummyPulses = 0, # Dummy pulses for T1 stabilization
    shimming = np.array([-70., -90., 10.]), # a.u.*1e4, Shimming along the X,Y and Z axes
    parFourierFraction = 1.0, # fraction of acquired k-space along phase direction
    spoiler = 0 # set 1 or 0 if you want or do not want to apply spoiler gradients
    ):
    
    freqCal = False
    demo = False
    
    # rawData fields
    rawData = {}
    
    # Conversion of variables to non-multiplied units
    larmorFreq = larmorFreq*1e6
    rfExTime = rfExTime*1e-6
    fov = np.array(fov)*1e-3
    dfov = np.array(dfov)*1e-3
    echoTime = echoTime*1e-3
    acqTime = acqTime*1e-3
    shimming = np.array(shimming)*1e-4
    repetitionTime = repetitionTime*1e-3
    rdGradTime = rdGradTime*1e-3
    dephGradTime = dephGradTime*1e-3
    
    # Inputs for rawData
    rawData['seqName'] = 'GRE_standalone'
    rawData['nScans'] = nScans
    rawData['larmorFreq'] = larmorFreq      # Larmor frequency
    rawData['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    rawData['rfExTime'] = rfExTime          # rf excitation pulse time
    rawData['echoTime'] = echoTime        # time between echoes
    rawData['repetitionTime'] = repetitionTime     # TR
    rawData['fov'] = fov           # FOV along readout, phase and slice
    rawData['dfov'] = dfov            # Displacement of fov center
    rawData['nPoints'] = nPoints                 # Number of points along readout, phase and slice
    rawData['acqTime'] = acqTime             # Acquisition time
    rawData['axesOrientation'] = axes       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    rawData['rdGradTime'] = rdGradTime
    rawData['dephGradTime'] = dephGradTime
    rawData['rdPreemphasis'] = rdPreemphasis
    rawData['drfPhase'] = drfPhase 
    rawData['dummyPulses'] = dummyPulses                    # Dummy pulses for T1 stabilization
    rawData['shimming'] = shimming
    rawData['partialFourierFraction'] = parFourierFraction
    rawData['spoiler'] = spoiler
    
    # Miscellaneous
    larmorFreq = larmorFreq*1e-6
    gradRiseTime = 200e-6       # Estimated gradient rise time
    gSteps = int(gradRiseTime*1e6/5)*0+1
    addRdPoints = 5             # Initial rd points to avoid artifact at the begining of rd
    resolution = fov/nPoints
    randFactor = 0.
    axesEnable = np.array([1, 1, 1])
    for ii in range(3):
        if nPoints[ii]==1: axesEnable[ii] = 0
    if fov[0]>1 and nPoints[1]==1: axesEnable[0] = 0
    rawData['randFactor'] = randFactor
    rawData['resolution'] = resolution
    rawData['gradRiseTime'] = gradRiseTime
    rawData['addRdPoints'] = addRdPoints
    
    # Matrix size
    nRD = nPoints[0]+2*addRdPoints
    nPH = nPoints[1]
    nSL = nPoints[2]
    
    # parAcqLines
    nSLreal = int(nPoints[2]*parFourierFraction)
    parAcqLines = int(nSLreal-nPoints[2]/2)
    rawData['parAcqLines'] = parAcqLines
    del nSLreal
    
    # BW
    BW = nPoints[0]/acqTime*1e-6
    BWov = BW*hw.oversamplingFactor
    samplingPeriod = 1/BWov
    rawData['samplingPeriod'] = samplingPeriod
    
    # Check if dephasing grad time is ok
    maxDephGradTime = echoTime-(rfExTime+rdGradTime)-3*gradRiseTime
    if dephGradTime==0 or dephGradTime>maxDephGradTime:
        dephGradTime = maxDephGradTime
    
    # Max gradient amplitude
    rdGradAmplitude = nPoints[0]/(hw.gammaB*fov[0]*acqTime)
    rdDephAmplitude = -rdGradAmplitude*(rdGradTime+gradRiseTime)/(2*(dephGradTime+gradRiseTime))
    phGradAmplitude = nPH/(2*hw.gammaB*fov[1]*(dephGradTime+gradRiseTime))*axesEnable[1]
    slGradAmplitude = nSL/(2*hw.gammaB*fov[2]*(dephGradTime+gradRiseTime))*axesEnable[2]
    rawData['rdGradAmplitude'] = rdGradAmplitude
    rawData['rdDephAmplitude'] = rdDephAmplitude
    rawData['phGradAmplitude'] = phGradAmplitude
    rawData['slGradAmplitude'] = slGradAmplitude

    # Phase and slice gradient vector
    phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
    slGradients = np.linspace(-slGradAmplitude,slGradAmplitude,num=nSL,endpoint=False)
    
    # Now fix the number of slices to partailly acquired k-space
    if nPoints[2]==1:
        nSL = 1
    else:
        nSL = int(nPoints[2]/2)+parAcqLines
    nRepetitions = nPH*nSL
    
    # Add random displacemnt to phase encoding lines
    for ii in range(nPH):
        if ii<np.ceil(nPH/2-nPH/20) or ii>np.ceil(nPH/2+nPH/20):
            phGradients[ii] = phGradients[ii]+randFactor*np.random.randn()
    kPH = hw.gammaB*phGradients*(gradRiseTime+dephGradTime)
    rawData['phGradients'] = phGradients
    rawData['slGradients'] = slGradients
    
    # Changing time parameters to us
    rfExTime = rfExTime*1e6
    echoTime = echoTime*1e6
    repetitionTime = repetitionTime*1e6
    gradRiseTime = gradRiseTime*1e6
    dephGradTime = dephGradTime*1e6
    rdGradTime = rdGradTime*1e6
    scanTime = nRepetitions*repetitionTime
    rawData['scanTime'] = scanTime*nSL*1e-6
    
    # Create demo
    def createSequenceDemo(phIndex=0, slIndex=0, repeIndexGlobal=0):
        repeIndex = 0
        acqPoints = 0
        orders = 0
        data = []
        
        if(dummyPulses>0 and nRD*3>hw.maxRdPoints) or (dummyPulses==0 and nRD*2>hw.maxRdPoints):
            print('ERROR: Too many acquired points or orders to the red pitaya.')
            return()
        
        while acqPoints+nRD<=hw.maxRdPoints and orders<=hw.maxOrders and repeIndexGlobal<nRepetitions:
            # First I do a noise measurement
            if repeIndex==0:
                acqPoints += nRD
                data = np.concatenate((data, np.random.randn(nRD*hw.oversamplingFactor)), axis = 0)
            
            # Dephasing readout, phase and slice
            if (repeIndex==0 or repeIndex>=dummyPulses):
                orders = orders+gSteps*2
            if repeIndex>=dummyPulses:
                orders = orders+gSteps*4
    
            # Rephasing readout gradient
            if (repeIndex==0 or repeIndex>=dummyPulses):
                orders = orders+gSteps*2
            
            # Rx gate
            if (repeIndex==0 or repeIndex>=dummyPulses):
                acqPoints += nRD
                data = np.concatenate((data, np.random.randn(nRD*hw.oversamplingFactor)), axis = 0)
        
            # Spoiler
            if (repeIndex==0 or repeIndex>=dummyPulses):
                orders = orders+gSteps*2
            if repeIndex>=dummyPulses:
                orders = orders+gSteps*4
                
            # Update the phase and slice gradient
            if repeIndex>=dummyPulses:
                if phIndex == nPH-1:
                    phIndex = 0
                    slIndex += 1
                else:
                    phIndex += 1
        
            if repeIndex>=dummyPulses: repeIndexGlobal += 1 # Update the global repeIndex
            repeIndex+=1 # Update the repeIndex after the ETL
        
        # Return the output variables
        return(phIndex, slIndex, repeIndexGlobal, acqPoints, data)
        
    # Create sequence instructions
    def createSequence(phIndex=0, slIndex=0, repeIndexGlobal=0, rewrite=True):
        repeIndex = 0
        acqPoints = 0
        orders = 0
        
        # check in case of dummy pulse filling the cache
        if(dummyPulses>0 and nRD*3>hw.maxRdPoints) or (dummyPulses==0 and nRD*2>hw.maxRdPoints):
            print('ERROR: Too many acquired points or orders to the red pitaya.')
            return()
        
        # Set shimming
        mri.iniSequence(expt, 20, shimming, rewrite=rewrite)
        
        # Run sequence batch
        while acqPoints+nRD<=hw.maxRdPoints and orders<=hw.maxOrders and repeIndexGlobal<nRepetitions:
            # Initialize time
            tEx = 20e3+repetitionTime*repeIndex
            
            # First I do a noise measurement
            if repeIndex==0:
                t0 = tEx-4*acqTime
                mri.rxGate(expt, t0, acqTime+2*addRdPoints/BW)
                acqPoints += nRD
            
            # Excitation pulse
            t0 = tEx-hw.blkTime-rfExTime/2
            mri.rfRecPulse(expt, t0, rfExTime, rfExAmp, drfPhase)
            
            # Dephasing readout, phase and slice
            if (repeIndex==0 or repeIndex>=dummyPulses):
                t0 = tEx+rfExTime/2-hw.gradDelay
                mri.gradTrap(expt, t0, gradRiseTime, dephGradTime, rdDephAmplitude*rdPreemphasis, gSteps,  axes[0],  shimming)
                orders = orders+gSteps*2
            if repeIndex>=dummyPulses:
                t0 = tEx+rfExTime/2-hw.gradDelay
                mri.gradTrap(expt, t0, gradRiseTime,  dephGradTime, phGradients[phIndex], gSteps,  axes[1], shimming)
                mri.gradTrap(expt, t0, gradRiseTime,  dephGradTime, slGradients[slIndex], gSteps,  axes[2], shimming)
                orders = orders+gSteps*4
            
            # Rephasing readout gradient
            if (repeIndex==0 or repeIndex>=dummyPulses):
                t0 = tEx+echoTime-rdGradTime/2-gradRiseTime-hw.gradDelay
                mri.gradTrap(expt, t0, gradRiseTime, rdGradTime, rdGradAmplitude, gSteps, axes[0], shimming)
                orders = orders+gSteps*2
            
            # Rx gate
            if (repeIndex==0 or repeIndex>=dummyPulses):
                t0 = tEx+echoTime-acqTime/2-addRdPoints/BW
                mri.rxGate(expt, t0, acqTime+2*addRdPoints/BW)
                acqPoints += nRD
        
            # Spoiler
            if spoiler:
                t0 = tEx+echoTime+rdGradTime/2+gradRiseTime-hw.gradDelay
                if (repeIndex==0 or repeIndex>=dummyPulses):
                    mri.gradTrap(expt, t0, gradRiseTime, dephGradTime, -rdDephAmplitude*rdPreemphasis, gSteps,  axes[0],  shimming)
                    orders = orders+gSteps*2
                if repeIndex>=dummyPulses:
                    mri.gradTrap(expt, t0, gradRiseTime,  dephGradTime, phGradients[phIndex], gSteps,  axes[1], shimming)
                    mri.gradTrap(expt, t0, gradRiseTime,  dephGradTime, slGradients[slIndex], gSteps,  axes[2], shimming)
                    orders = orders+gSteps*4
                
            # Update the phase and slice gradient
            if repeIndex>=dummyPulses:
                if phIndex == nPH-1:
                    phIndex = 0
                    slIndex += 1
                else:
                    phIndex += 1
        
            if repeIndex>=dummyPulses: repeIndexGlobal += 1 # Update the global repeIndex
            repeIndex+=1 # Update the repeIndex after the ETL
        
        # Turn off the gradients after the end of the batch
        mri.endSequence(expt, repeIndex*repetitionTime)
        
        # Return the output variables
        return(phIndex, slIndex, repeIndexGlobal, acqPoints)
    
    # Calibrate frequency
    if (not demo) and freqCal: 
        mri.freqCalibration(rawData, bw=0.05)
        mri.freqCalibration(rawData, bw=0.005)
        larmorFreq = rawData['larmorFreq']*1e-6
        drfPhase = rawData['drfPhase']
        
    # Initialize the experiment
    dataFull = []
    dummyData = []
    overData = []
    noise = []
    nBatches = 0
    repeIndexArray = np.array([0])
    repeIndexGlobal = repeIndexArray[0]
    phIndex = 0
    slIndex = 0
    acqPointsPerBatch = []
    while repeIndexGlobal<nRepetitions:
        nBatches += 1
        if not demo:
            expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
            samplingPeriod = expt.get_rx_ts()[0]
            BW = 1/samplingPeriod/hw.oversamplingFactor
            acqTime = nPoints[0]/BW        # us
            phIndex, slIndex, repeIndexGlobal, aa = createSequence(phIndex=phIndex,
                                                               slIndex=slIndex,
                                                               repeIndexGlobal=repeIndexGlobal,
                                                               rewrite=False)
            repeIndexArray = np.concatenate((repeIndexArray, np.array([repeIndexGlobal-1])), axis=0)
            acqPointsPerBatch.append(aa)
            expt.plot_sequence()
        else:
            phIndex, slIndex, repeIndexGlobal, aa, dataA = createSequenceDemo(phIndex=phIndex,
                                                               slIndex=slIndex,
                                                               repeIndexGlobal=repeIndexGlobal)
            repeIndexArray = np.concatenate((repeIndexArray, np.array([repeIndexGlobal-1])), axis=0)
            acqPointsPerBatch.append(aa)
        
        for ii in range(nScans):
            print('Batch ', nBatches, ', Scan ', ii, ' runing...')
            if not demo:
                rxd, msgs = expt.run()
                rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
                # Get noise data
                noise = np.concatenate((noise, rxd['rx0'][0:nRD*hw.oversamplingFactor]), axis = 0)
                rxd['rx0'] = rxd['rx0'][nRD*hw.oversamplingFactor::]
                # Get data
                if dummyPulses>0:
                    dummyData = np.concatenate((dummyData, rxd['rx0'][0:nRD*hw.oversamplingFactor]), axis = 0)
                    overData = np.concatenate((overData, rxd['rx0'][nRD*hw.oversamplingFactor::]), axis = 0)
                else:
                    overData = np.concatenate((overData, rxd['rx0']), axis = 0)
            else:
                data = dataA
                noise = np.concatenate((noise, data[0:nRD*hw.oversamplingFactor]), axis = 0)
                data = data[nRD*hw.oversamplingFactor::]
                # Get data
                if dummyPulses>0:
                    dummyData = np.concatenate((dummyData, data[0:nRD*hw.oversamplingFactor]), axis = 0)
                    overData = np.concatenate((overData, data[nRD*hw.oversamplingFactor::]), axis = 0)
                else:
                    overData = np.concatenate((overData, data), axis = 0)
                
        if not demo: expt.__del__()
    del aa
    acqPointsPerBatch = (acqPointsPerBatch-nRD*(dummyPulses>0)-nRD)*nScans
    print('Scans done!')
    rawData['noiseData'] = noise
    rawData['overData'] = overData
    
    # Fix the echo position using oversampled data
    if dummyPulses>0:
        dummyData = np.reshape(dummyData,  (nBatches*nScans, 1, nRD*hw.oversamplingFactor))
        dummyData = np.average(dummyData, axis=0)
        rawData['dummyData'] = dummyData
        overData = np.reshape(overData, (-1, 1, nRD*hw.oversamplingFactor))
        overData = mri.fixEchoPosition(dummyData, overData)
        overData = np.reshape(overData, -1)
    
    # Generate dataFull
    dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
    if nBatches>1:
        dataFullA = dataFull[0:sum(acqPointsPerBatch[0:-1])]
        dataFullB = dataFull[sum(acqPointsPerBatch[0:-1])::]
    
    # Reorganize dataFull
    dataProv = np.zeros([nScans,nSL*nPH*nRD])
    dataProv = dataProv+1j*dataProv
    dataFull = np.reshape(dataFull, (nBatches, nScans, -1, nRD))
    if nBatches>1:
        dataFullA = np.reshape(dataFullA, (nBatches-1, nScans, -1, nRD))
        dataFullB = np.reshape(dataFullB, (1, nScans, -1, nRD))
    for scan in range(nScans):
        if nBatches>1:
            dataProv[ii, :] = np.concatenate((np.reshape(dataFullA[:,ii,:,:],-1), np.reshape(dataFullB[:,ii,:,:],-1)), axis=0)
        else:
            dataProv[ii, :] = np.reshape(dataFull[:,ii,:,:],-1)
    dataFull = np.reshape(dataProv,-1)
    
    # Get index for krd = 0
    # Average data
    dataProv = np.reshape(dataFull, (nScans, nRD*nPH*nSL))
    dataProv = np.average(dataProv, axis=0)
    dataProv = np.reshape(dataProv, (nSL, nPH, nRD))
    # Check where is krd = 0
    dataProv = dataProv[int(nPoints[2]/2), int(nPH/2), :]
    indkrd0 = np.argmax(np.abs(dataProv))
    if  indkrd0 < nRD/2-addRdPoints or indkrd0 > nRD/2+addRdPoints:
        indkrd0 = int(nRD/2)
    indkrd0 = int(nRD/2)
        
    # Get individual images
    dataFull = np.reshape(dataFull, (nScans, nSL, nPH, nRD))
    dataFull = dataFull[:, :, :, indkrd0-int(nPoints[0]/2):indkrd0+int(nPoints[0]/2)]
    imgFull = dataFull*0
    for ii in range(nScans):
        imgFull[ii, :, :, :] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataFull[ii, :, :, :])))
    rawData['dataFull'] = dataFull
    rawData['imgFull'] = imgFull    
    
    # Average data
    data = np.average(dataFull, axis=0)
    data = np.reshape(data, (nSL, nPH, nPoints[0]))
    
    # Do zero padding
    dataTemp = np.zeros((nPoints[2], nPoints[1], nPoints[0]))
    dataTemp = dataTemp+1j*dataTemp
    dataTemp[0:nSL, :, :] = data
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
    mri.saveRawData(rawData)
    
    
    
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
            plt.title(rawData['fileName'])
            plt.legend(['Abs', 'Real', 'Imag'])
        else:
            plt.plot(kRD[:, 0], np.abs(dataPlot))
            plt.yscale('log')
            plt.xlabel('krd (mm^-1)')
            plt.ylabel('Signal (mV)')
            echoTime = np.argmax(np.abs(dataPlot))
            echoTime = kRD[echoTime, 0]
            print("Echo position = %s mm^{-1}" %round(echoTime, 1))
            plt.title(rawData['fileName'])
        
        # Plot image
        plt.subplot(122)
        img = img[0, 0, :]
        if axesEnable[0]==0:
            xAxis = np.linspace(-BW/2, BW/2, num=nPoints[0], endpoint=False)*1e3
            plt.plot(xAxis, np.abs(img), '.')
            plt.xlabel('Frequency (kHz)')
            plt.ylabel('Density (a.u.)')
            print("Smax = %s mV" % (np.max(np.abs(img))))
            plt.title(rawData['fileName'])
        else:
            xAxis = np.linspace(-fov[0]/2*1e2, fov[0]/2*1e2, num=nPoints[0], endpoint=False)
            plt.plot(xAxis, np.abs(img))
            plt.xlabel('Position RD (cm)')
            plt.ylabel('Density (a.u.)')
            plt.title(rawData['fileName'])
    else:
        # Plot k-space
        plt.figure(3)
        dataPlot = data[round(nSL/2), :, :]
        plt.subplot(131)
        plt.imshow(np.log(np.abs(dataPlot)),cmap='gray')
        plt.axis('off')
        # Plot image
        imgPlot = img[round(nSL/2), :, :]
        plt.subplot(132)
        plt.imshow(np.abs(imgPlot), cmap='gray')
        plt.axis('off')
        plt.title(rawData['fileName'])
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
        plt.title(rawData['fileName'])
    
    plt.show()
    

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


if __name__ == "__main__":

    gre_standalone()
