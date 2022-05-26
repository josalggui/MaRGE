# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:05:53 2022
@author: José Miguel Algarín Guisado
MRILAB @ I3M
"""

import numpy as np
import experiment as ex
import matplotlib.pyplot as plt
import scipy.signal as sig
import pdb
import configs.hw_config as hw # Import the scanner hardware config
import mrilabMethods.mrilabMethods as mri # This import all methods inside the mrilabMethods module
st = pdb.set_trace



#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def haste(self, plotSeq):
    init_gpa=False # Starts the gpa
    nScans = self.nScans # NEX
    larmorFreq = self.larmorFreq # MHz, Larmor frequency
    rfExAmp = self.rfExAmp # a.u., rf excitation pulse amplitude
    rfReAmp = self.rfReAmp # a.u., rf refocusing pulse amplitude
    rfExTime = self.rfExTime # us, rf excitation pulse time
    rfReTime = self.rfReTime # us, rf refocusing pulse time
    rfEnvelope = self.rfEnvelope  # 'Rec' -> square pulse, 'Sinc' -> sinc pulse
    echoSpacing = self.echoSpacing # ms, time between echoes
    inversionTime = self.inversionTime # ms, Inversion recovery time
    repetitionTime = self.repetitionTime # ms, TR
    fov = np.array(self.fov) # mm, FOV along readout, phase and slice
    dfov = np.array(self.dfov) # mm, displacement of fov center
    nPoints = np.array(self.nPoints) # Number of points along readout, phase and slice
    acqTime = self.acqTime # ms, acquisition time
    axes = np.array(self.axes) # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    axesEnable = self.axesEnable # 1-> Enable, 0-> Disable
    sweepMode = self.sweepMode # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    rdGradTime = self.rdGradTime  # ms, readout gradient time
    rdDephTime = self.rdDephTime  # ms, readout dephasing time
    phGradTime = self.phGradTime # ms, phase and slice dephasing time
    rdPreemphasis = self.rdPreemphasis # readout dephasing gradient is multiplied by this factor
    ssPreemphasis = self.ssPreemphasis # slice rephasing gradient is multiplied by this factor
    crusherDelay = self.crusherDelay # delay of the crusher gradient
    drfPhase = self.drfPhase # degrees, phase of the excitation pulse
    dummyPulses = self.dummyPulses # number of dummy pulses for T1 stabilization
    shimming = self.shimming # a.u.*1e4, shimming along the X,Y and Z axes
    parFourierFraction = self.parFourierFraction # fraction of acquired k-space along phase direction
    
    freqCal = True
    demo = False
    
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
    shimming = np.array(shimming)*1e-4
    repetitionTime= repetitionTime*1e-3
    inversionTime = inversionTime*1e-3
    rdGradTime = rdGradTime*1e-3
    rdDephTime = rdDephTime*1e-3
    phGradTime = phGradTime*1e-3
    crusherDelay = crusherDelay*1e-6
    
    # Inputs for rawData
    rawData['seqName'] = 'HASTE'
    rawData['nScans'] = nScans
    rawData['larmorFreq'] = larmorFreq      # Larmor frequency
    rawData['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    rawData['rfReAmp'] = rfReAmp             # rf refocusing pulse amplitude
    rawData['rfExTime'] = rfExTime          # rf excitation pulse time
    rawData['rfReTime'] = rfReTime            # rf refocusing pulse time
    rawData['rfEnvelope'] = rfEnvelope
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
    rawData['crusherDelay'] = crusherDelay
    rawData['shimming'] = shimming
    
    # Miscellaneous
    slThickness = fov[2]
    rfSincLobes = 7     # Number of lobes for sinc rf excitation, BW = rfSincLobes/rfTime
    larmorFreq = larmorFreq*1e-6
    gradRiseTime = 200e-6       # Estimated gradient rise time
    gSteps = int(gradRiseTime*1e6/5)
    addRdPoints = 10             # Initial rd points to avoid artifact at the begining of rd
    if rfReAmp==0:
        rfReAmp = 2*rfExAmp
    if rfReTime==0:
        rfReTime = rfExTime
    resolution = fov/nPoints
    rawData['resolution'] = resolution
    rawData['gradRiseTime'] = gradRiseTime
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
    BWov = BW*hw.oversamplingFactor
    samplingPeriod = 1/BWov
    rawData['samplingPeriod'] = samplingPeriod
    
    # Readout gradient time
    if rdGradTime<acqTime:
        rdGradTime = acqTime
    rawData['rdGradTime'] = rdGradTime
    
    # Phase de- and re-phasing time
    if phGradTime==0 or phGradTime>echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime:
        phGradTime = echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime
    rawData['phGradTime'] = phGradTime
    
    # Slice selection dephasing gradient time
    ssDephGradTime = (rfExTime-gradRiseTime)/2
    rawData['ssDephGradTime'] = ssDephGradTime
    
    # Max redaout and phase gradient amplitude
    rdGradAmplitude = nPoints[0]/(hw.gammaB*fov[0]*acqTime)*axesEnable[0]
    phGradAmplitude = nPH/(2*hw.gammaB*fov[1]*(phGradTime+gradRiseTime))*axesEnable[1]
    rawData['rdGradAmplitude'] = rdGradAmplitude
    rawData['phGradAmplitude'] = phGradAmplitude
    
    # Slice selection gradient
    if rfEnvelope=='Sinc':
        ssGradAmplitude = rfSincLobes/(hw.gammaB*slThickness*rfExTime)*axesEnable[2]
    elif rfEnvelope=='Rec':
        ssGradAmplitude = 1/(hw.gammaB*slThickness*rfExTime)*axesEnable[2]
    rawData['ssGradAmplitude'] = ssGradAmplitude
    
    # Readout dephasing amplitude
    rdDephAmplitude = 0.5*rdGradAmplitude*(gradRiseTime+rdGradTime)/(gradRiseTime+rdDephTime)
    rawData['rdDephAmplitude'] = rdDephAmplitude

    # Phase gradient vector
    phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
    
    # Get phase indexes for the given sweep mode
    ind = mri.getIndex(2*parAcqLines, 2*parAcqLines, sweepMode)
    ind = ind-parAcqLines+int(nPH/2)
    ind = np.int32(np.concatenate((ind, np.linspace(int(nPH/2)-parAcqLines-1, -1, num=int(nPH/2)-parAcqLines, endpoint=False)), axis=0))
    rawData['sweepOrder'] = ind
    
    # Now fix the number of phases to partailly acquired k-space
    nPH = (int(nPoints[1]/2)+parAcqLines)*axesEnable[1]+(1-axesEnable[1])
    phGradients = phGradients[0:nPH]
    phGradients = phGradients[ind]
    rawData['phGradients'] = phGradients
    
    def createSequenceDemo():
        nRepetitions = int(1+dummyPulses)
        scanTime = 20e3+nRepetitions*repetitionTime
        rawData['scanTime'] = scanTime*1e-6
        acqPoints = 0
        data = []
        for repeIndex in range(nRepetitions):
            for echoIndex in range(nPH):
                if (repeIndex==0 or repeIndex>=dummyPulses):
                    acqPoints += nRD*hw.oversamplingFactor
                    data = np.concatenate((data, np.random.randn(nRD*hw.oversamplingFactor)+1j*np.random.randn(nRD*hw.oversamplingFactor)), axis=0)
        return data, acqPoints
                
        
    def createSequence():
        nRepetitions = int(1+dummyPulses)
        scanTime = 20e3+nRepetitions*repetitionTime
        rawData['scanTime'] = scanTime*1e-6
        print('Scan Time = ', (scanTime*1e-6),  's')
        if rdGradTime==0:   # Check if readout gradient is dc or pulsed
            dc = True
        else:
            dc = False
        
        # Set shimming
        mri.iniSequence(expt, 20, shimming)
        for repeIndex in range(nRepetitions):
            # Initialize time
            tEx = 20e3+repetitionTime*repeIndex+inversionTime
            
            # Inversion pulse
            if repeIndex>=dummyPulses and inversionTime!=0:
                t0 = tEx-inversionTime-rfReTime/2-hw.blkTime
                mri.rfRecPulse(expt, t0,rfReTime,rfReAmp/180*180,0)
                mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[0], shimming)
                mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[1], shimming)
                mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[2], shimming)
            
            # DC radout gradient if desired
            if (repeIndex==0 or repeIndex>=dummyPulses) and dc==True:
                t0 = tEx-rfExTime/2-gradRiseTime-hw.gradDelay
                mri.gradTrap(expt, t0, echoSpacing*(nPH+1), rdGradAmplitude, axes[0])
            
            # Slice selection gradient dephasing
            if (slThickness!=0 and repeIndex>=dummyPulses):
                t0 = tEx-rfExTime/2-gradRiseTime-hw.gradDelay
                mri.gradTrap(expt, t0, gradRiseTime, rfExTime, ssGradAmplitude, gSteps, axes[2], shimming)
            
            # Excitation pulse
            t0 = tEx-hw.blkTime-rfExTime/2
            if rfEnvelope=='Rec': 
                mri.rfRecPulse(expt, t0,rfExTime,rfExAmp,drfPhase*np.pi/180)
            elif rfEnvelope=='Sinc':
                mri.rfSincPulse(expt, t0, rfExTime, rfSincLobes, rfExAmp, drfPhase*np.pi/180)
            
            # Slice selection gradient rephasing
            if (slThickness!=0 and repeIndex>=dummyPulses):
                t0 = tEx+rfExTime/2+gradRiseTime-hw.gradDelay
                if rfEnvelope=='Rec':
                    mri.gradTrap(expt, t0, gradRiseTime, 0., -ssGradAmplitude*ssPreemphasis, gSteps, axes[2], shimming)
                elif rfEnvelope=='Sinc': 
                    mri.gradTrap(expt, t0, gradRiseTime, ssDephGradTime, -ssGradAmplitude*ssPreemphasis, gSteps, axes[2], shimming)

            # Dephasing readout
            t0 = tEx+rfExTime/2-hw.gradDelay
            if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False:
                mri.gradTrap(expt, t0, gradRiseTime, rdDephTime, rdDephAmplitude*rdPreemphasis, gSteps, axes[0], shimming)
            
            # Echo train
            for echoIndex in range(nPH):
                tEcho = tEx+echoSpacing*(echoIndex+1)
                
                # Crusher gradient
                if repeIndex>=dummyPulses:
                    t0 = tEcho-echoSpacing/2-rfReTime/2-gradRiseTime-hw.gradDelay-crusherDelay
                    mri.gradTrap(expt, t0, gradRiseTime, rfReTime+2*crusherDelay, ssGradAmplitude, gSteps, axes[2], shimming)
                
                # Refocusing pulse
                t0 = tEcho-echoSpacing/2-rfReTime/2-hw.blkTime
                if rfEnvelope=='Rec':
                    mri.rfRecPulse(expt, t0, rfReTime, rfReAmp, np.pi/2+drfPhase*np.pi/180)
                if rfEnvelope=='Sinc':
                    mri.rfSincPulse(expt, t0, rfReTime, rfSincLobes, rfReAmp, np.pi/2+drfPhase*np.pi/180)
                
                # Dephasing phase gradient
                t0 = tEcho-echoSpacing/2+rfReTime/2-hw.gradDelay
                if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                    mri.gradTrap(expt, t0, gradRiseTime, phGradTime, phGradients[echoIndex], gSteps, axes[1], shimming)
                    
                # Readout gradient
                t0 = tEcho-rdGradTime/2-gradRiseTime-hw.gradDelay
                if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False:         # This is to account for dummy pulses
                    mri.gradTrap(expt, t0, gradRiseTime, rdGradTime, rdGradAmplitude, gSteps, axes[0], shimming)
    
                # Rx gate
                if (repeIndex==0 or repeIndex>=dummyPulses):
                    t0 = tEcho-acqTime/2-addRdPoints/BW
                    mri.rxGate(expt, t0, acqTime+2*addRdPoints/BW)
    
                # Rephasing phase and slice gradients
                t0 = tEcho+acqTime/2+addRdPoints/BW-hw.gradDelay
                if (echoIndex<nPH-1 and repeIndex>=dummyPulses):
                    mri.gradTrap(expt, t0, gradRiseTime, phGradTime, -phGradients[echoIndex], gSteps, axes[1], shimming)
                elif(echoIndex==nPH-1 and repeIndex>=dummyPulses):
                    mri.gradTrap(expt, t0, gradRiseTime, phGradTime, +phGradients[echoIndex], gSteps, axes[1], shimming)

            if repeIndex==nRepetitions-1:
                mri.endSequence(expt, scanTime)

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
    ssDephGradTime = ssDephGradTime*1e6
    
    # Calibrate frequency
    if not demo and freqCal: 
        mri.freqCalibration(rawData, bw=0.05)
        mri.freqCalibration(rawData, bw=0.005)
        larmorFreq = rawData['larmorFreq']*1e-6
    
    # Create full sequence
    if not demo:
        expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = expt.get_rx_ts()[0]
        BW = 1/samplingPeriod/hw.oversamplingFactor
        acqTime = nPoints[0]/BW        # us
        createSequence()

    # Plot sequence:
    expt.plot_sequence()
        
    # Run the experiment
    dataFull = []
    dummyData = []
    overData = []
    for ii in range(nScans):
        print("Scan %s ..." % (ii+1))
        if not demo:
            rxd, msgs = expt.run()
            rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
        else:
            data, acqPoints = createSequenceDemo()
        # Get data
        if not demo:
            if dummyPulses>0:
                dummyData = np.concatenate((dummyData, rxd['rx0'][0:nRD*nPH*hw.oversamplingFactor]), axis = 0)
                overData = np.concatenate((overData, rxd['rx0'][nRD*nPH*hw.oversamplingFactor::]), axis = 0)
            else:
                overData = np.concatenate((overData, rxd['rx0']), axis = 0)
        else:
            if dummyPulses>0:
                dummyData = np.concatenate((dummyData, data[0:nRD*nPH*hw.oversamplingFactor]), axis = 0)
                overData = np.concatenate((overData, data[nRD*nPH*hw.oversamplingFactor::]), axis = 0)
            else:
                overData = np.concatenate((overData, data), axis = 0)
    if not demo: expt.__del__()
    print('Scans done!')
    rawData['overData'] = overData
    
    # Fix the echo position using oversampled data
    if dummyPulses>0:
        dummyData = np.reshape(dummyData,  (nScans, nPH, nRD*hw.oversamplingFactor))
        dummyData = np.average(dummyData, axis=0)
        rawData['dummyData'] = dummyData
        overData = np.reshape(overData, (nScans, 1, nPH,  nRD*hw.oversamplingFactor))
        for ii in range(nScans):
            overData[ii, :, :, :] = mri.fixEchoPosition(dummyData, overData[ii, :, :, :])
        
    # Generate dataFull
    overData = np.squeeze(np.reshape(overData, (1, nRD*hw.oversamplingFactor*nPH*nScans)))
    dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
    
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
    if  indkrd0 < nRD/2-addRdPoints or indkrd0 > nRD/2+addRdPoints:
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
    data = np.reshape(data, -1)
    
    # Save data
    mri.saveRawData(rawData)

    return rawData, msgs, data,  BW
