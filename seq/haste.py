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
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from plotview.spectrumplot import SpectrumPlot # To plot nice 1d images
from PyQt5.QtWidgets import QLabel  # To set the figure title
from PyQt5 import QtCore            # To set the figure title
import pyqtgraph as pg              # To plot nice 3d images+


class HASTE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(HASTE, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='HASTEInfo', val='HASTE')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.03, field='RF')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.06, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=200.0, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=200.0, field='RF')
        self.addParameter(key='rfEnvelope', string='RF envelope 0->Rec, 1->Sinc', val=0, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, field='SEQ')
        self.addParameter(key='inversionTime', string='Inversion time (ms)', val=0.0, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ')
        self.addParameter(key='fov', string='FOV (cm)', val=[12.0, 12.0, 12.0], field='IM')
        self.addParameter(key='dfov', string='dFOV (mm)', val=[0.0, 0.0, 0.0], field='IM')
        self.addParameter(key='nPoints', string='nPoints (rd, ph, sl)', val=[60, 60, 1], field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='SEQ')
        self.addParameter(key='axes', string='Axes', val=[0, 1, 2], field='IM')
        self.addParameter(key='axesEnable', string='Axes enable', val=[1, 1, 0], field='IM')
        self.addParameter(key='sweepMode', string='Sweep mode, 0->k20, 1->02k, 2->k2k', val=1, field='SEQ')
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=5.0, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, field='OTH')
        self.addParameter(key='phGradTime', string='Ph gradient time (ms)', val=1.0, field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')
        self.addParameter(key='ssPreemphasis', string='SS preemphasis', val=1.0, field='OTH')
        self.addParameter(key='crusherDelay', string='Crusher delay (us)', val=0.0, field='OTH')
        self.addParameter(key='drfPhase', string='Phase of exciation pulse (º)', val=0.0, field='RF')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=1, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='parFourierFraction', string='Partial fourier fraction', val=1.0, field='OTH')


    def sequenceInfo(self):
        print(" ")
        print("2D HASTE sequence")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")


    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime']
        return(repetitionTime*1e-3*nScans/60)  # minutes, scanTime


    def sequenceRun(self, plotSeq):
        init_gpa=False # Starts the gpa
        nScans = self.mapVals['nScans'] # NEX
        larmorFreq = self.mapVals['larmorFreq'] # MHz, Larmor frequency
        rfExAmp = self.mapVals['rfExAmp'] # a.u., rf excitation pulse amplitude
        rfReAmp = self.mapVals['rfReAmp'] # a.u., rf refocusing pulse amplitude
        rfExTime = self.mapVals['rfExTime'] # us, rf excitation pulse time
        rfReTime = self.mapVals['rfReTime'] # us, rf refocusing pulse time
        rfEnvelope = self.mapVals['rfEnvelope'] # 0 -> square pulse, 1 -> sinc pulse
        echoSpacing = self.mapVals['echoSpacing'] # ms, time between echoes
        inversionTime = self.mapVals['inversionTime'] # ms, Inversion recovery time
        repetitionTime = self.mapVals['repetitionTime'] # ms, TR
        fov = np.array(self.mapVals['fov']) # mm, FOV along readout, phase and slice
        dfov = np.array(self.mapVals['dfov']) # mm, displacement of fov center
        nPoints = np.array(self.mapVals['nPoints']) # Number of points along readout, phase and slice
        acqTime = self.mapVals['acqTime'] # ms, acquisition time
        axes = np.array(self.mapVals['axes']) # 0->x, 1->y and 2->z defined as [rd,ph,sl]
        axesEnable = np.array(self.mapVals['axesEnable']) # 1-> Enable, 0-> Disable
        sweepMode = self.mapVals['sweepMode'] # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
        rdGradTime = self.mapVals['rdGradTime'] # ms, readout gradient time
        rdDephTime = self.mapVals['rdDephTime'] # ms, readout dephasing time
        phGradTime = self.mapVals['phGradTime'] # ms, phase and slice dephasing time
        rdPreemphasis = self.mapVals['rdPreemphasis'] # readout dephasing gradient is multiplied by this factor
        ssPreemphasis = self.mapVals['ssPreemphasis'] # slice rephasing gradient is multiplied by this factor
        crusherDelay = self.mapVals['crusherDelay'] # delay of the crusher gradient
        drfPhase = self.mapVals['drfPhase'] # degrees, phase of the excitation pulse
        dummyPulses = self.mapVals['dummyPulses'] # number of dummy pulses for T1 stabilization
        shimming = self.mapVals['shimming'] # a.u.*1e4, shimming along the X,Y and Z axes
        parFourierFraction = self.mapVals['parFourierFraction'] # fraction of acquired k-space along phase direction

        freqCal = True
        demo = False

        # Conversion of variables to non-multiplied units
        larmorFreq = larmorFreq*1e6
        rfExTime = rfExTime*1e-6
        rfReTime = rfReTime*1e-6
        fov = np.array(fov)*1e-2
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
        self.mapVals['resolution'] = resolution
        self.mapVals['gradRiseTime'] = gradRiseTime
        self.mapVals['addRdPoints'] = addRdPoints

        # Matrix size
        nRD = nPoints[0]+2*addRdPoints
        nPH = nPoints[1]*axesEnable[1]+(1-axesEnable[1])

        # parAcqLines
        nPHreal = int(nPoints[1]*parFourierFraction)
        parAcqLines = int(nPHreal-nPoints[1]/2)
        self.mapVals['parAcqLines'] = parAcqLines
        print(parAcqLines)
        del nPHreal

        # BW
        BW = nPoints[0]/acqTime*1e-6
        BWov = BW*hw.oversamplingFactor
        samplingPeriod = 1/BWov
        self.mapVals['samplingPeriod'] = samplingPeriod

        # Readout gradient time
        if rdGradTime<acqTime:
            rdGradTime = acqTime
        self.mapVals['rdGradTime'] = rdGradTime

        # Phase de- and re-phasing time
        if phGradTime==0 or phGradTime>echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime:
            phGradTime = echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime
        self.mapVals['phGradTime'] = phGradTime

        # Slice selection dephasing gradient time
        ssDephGradTime = (rfExTime-gradRiseTime)/2
        self.mapVals['ssDephGradTime'] = ssDephGradTime

        # Max redaout and phase gradient amplitude
        rdGradAmplitude = nPoints[0]/(hw.gammaB*fov[0]*acqTime)*axesEnable[0]
        phGradAmplitude = nPH/(2*hw.gammaB*fov[1]*(phGradTime+gradRiseTime))*axesEnable[1]
        self.mapVals['rdGradAmplitude'] = rdGradAmplitude
        self.mapVals['phGradAmplitude'] = phGradAmplitude

        # Slice selection gradient
        if rfEnvelope==1:
            ssGradAmplitude = rfSincLobes/(hw.gammaB*slThickness*rfExTime)*axesEnable[2]
        elif rfEnvelope==0:
            ssGradAmplitude = 1/(hw.gammaB*slThickness*rfExTime)*axesEnable[2]
        self.mapVals['ssGradAmplitude'] = ssGradAmplitude

        # Readout dephasing amplitude
        rdDephAmplitude = 0.5*rdGradAmplitude*(gradRiseTime+rdGradTime)/(gradRiseTime+rdDephTime)
        self.mapVals['rdDephAmplitude'] = rdDephAmplitude

        # Phase gradient vector
        phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)

        # Get phase indexes for the given sweep mode
        ind = self.getIndex(2*parAcqLines, 2*parAcqLines, sweepMode)
        ind = ind-parAcqLines+int(nPH/2)
        ind = np.int32(np.concatenate((ind, np.linspace(int(nPH/2)-parAcqLines-1, -1, num=int(nPH/2)-parAcqLines, endpoint=False)), axis=0))
        self.mapVals['sweepOrder'] = ind

        # Now fix the number of phases to partailly acquired k-space
        nPH = (int(nPoints[1]/2)+parAcqLines)*axesEnable[1]+(1-axesEnable[1])
        phGradients = phGradients[0:nPH]
        phGradients = phGradients[ind]
        self.mapVals['phGradients'] = phGradients

        def createSequenceDemo():
            nRepetitions = int(1+dummyPulses)
            scanTime = 20e3+nRepetitions*repetitionTime
            self.mapVals['scanTime'] = scanTime*1e-6
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
            if rdGradTime==0:   # Check if readout gradient is dc or pulsed
                dc = True
            else:
                dc = False

            # Set shimming
            self.iniSequence(20, shimming)
            for repeIndex in range(nRepetitions):
                # Initialize time
                tEx = 20e3+repetitionTime*repeIndex+inversionTime

                # Inversion pulse
                if repeIndex>=dummyPulses and inversionTime!=0:
                    t0 = tEx-inversionTime-rfReTime/2-hw.blkTime
                    self.rfRecPulse(t0,rfReTime,rfReAmp/180*180,0)
                    self.gradTrap(t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[0], shimming)
                    self.gradTrap(t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[1], shimming)
                    self.gradTrap(t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[2], shimming)

                # DC radout gradient if desired
                if (repeIndex==0 or repeIndex>=dummyPulses) and dc==True:
                    t0 = tEx-rfExTime/2-gradRiseTime-hw.gradDelay
                    self.gradTrap(t0, echoSpacing*(nPH+1), rdGradAmplitude, axes[0])

                # Slice selection gradient dephasing
                if (slThickness!=0 and repeIndex>=dummyPulses):
                    t0 = tEx-rfExTime/2-gradRiseTime-hw.gradDelay
                    self.gradTrap(t0, gradRiseTime, rfExTime, ssGradAmplitude, gSteps, axes[2], shimming)

                # Excitation pulse
                t0 = tEx-hw.blkTime-rfExTime/2
                if rfEnvelope==0:
                    self.rfRecPulse(t0,rfExTime,rfExAmp,drfPhase*np.pi/180)
                elif rfEnvelope==1:
                    self.rfSincPulse(t0, rfExTime, rfSincLobes, rfExAmp, drfPhase*np.pi/180)

                # Slice selection gradient rephasing
                if (slThickness!=0 and repeIndex>=dummyPulses):
                    t0 = tEx+rfExTime/2+gradRiseTime-hw.gradDelay
                    if rfEnvelope==0:
                        self.gradTrap(t0, gradRiseTime, 0., -ssGradAmplitude*ssPreemphasis, gSteps, axes[2], shimming)
                    elif rfEnvelope==1:
                        self.gradTrap(t0, gradRiseTime, ssDephGradTime, -ssGradAmplitude*ssPreemphasis, gSteps, axes[2], shimming)

                # Dephasing readout
                t0 = tEx+rfExTime/2-hw.gradDelay
                if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False:
                    self.gradTrap(t0, gradRiseTime, rdDephTime, rdDephAmplitude*rdPreemphasis, gSteps, axes[0], shimming)

                # Echo train
                for echoIndex in range(nPH):
                    tEcho = tEx+echoSpacing*(echoIndex+1)

                    # Crusher gradient
                    if repeIndex>=dummyPulses:
                        t0 = tEcho-echoSpacing/2-rfReTime/2-gradRiseTime-hw.gradDelay-crusherDelay
                        self.gradTrap(t0, gradRiseTime, rfReTime+2*crusherDelay, ssGradAmplitude, gSteps, axes[2], shimming)

                    # Refocusing pulse
                    t0 = tEcho-echoSpacing/2-rfReTime/2-hw.blkTime
                    if rfEnvelope==0:
                        self.rfRecPulse(t0, rfReTime, rfReAmp, np.pi/2+drfPhase*np.pi/180)
                    if rfEnvelope==1:
                        self.rfSincPulse(t0, rfReTime, rfSincLobes, rfReAmp, np.pi/2+drfPhase*np.pi/180)

                    # Dephasing phase gradient
                    t0 = tEcho-echoSpacing/2+rfReTime/2-hw.gradDelay
                    if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                        self.gradTrap(t0, gradRiseTime, phGradTime, phGradients[echoIndex], gSteps, axes[1], shimming)

                    # Readout gradient
                    t0 = tEcho-rdGradTime/2-gradRiseTime-hw.gradDelay
                    if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False:         # This is to account for dummy pulses
                        self.gradTrap(t0, gradRiseTime, rdGradTime, rdGradAmplitude, gSteps, axes[0], shimming)

                    # Rx gate
                    if (repeIndex==0 or repeIndex>=dummyPulses):
                        t0 = tEcho-acqTime/2-addRdPoints/BW
                        self.rxGate(t0, acqTime+2*addRdPoints/BW)

                    # Rephasing phase and slice gradients
                    t0 = tEcho+acqTime/2+addRdPoints/BW-hw.gradDelay
                    if (echoIndex<nPH-1 and repeIndex>=dummyPulses):
                        self.gradTrap(t0, gradRiseTime, phGradTime, -phGradients[echoIndex], gSteps, axes[1], shimming)
                    elif(echoIndex==nPH-1 and repeIndex>=dummyPulses):
                        self.gradTrap(t0, gradRiseTime, phGradTime, +phGradients[echoIndex], gSteps, axes[1], shimming)

                if repeIndex==nRepetitions-1:
                    self.endSequence(scanTime)

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

        if repetitionTime<nPH*echoSpacing:
            print('Warning: Repetition time should be longer')
            return(0)

        # Calibrate frequency
        if freqCal and (not plotSeq) and (not demo):
            self.freqCalibration(bw=0.05)
            self.freqCalibration(bw=0.005)
            drfPhase = self.mapVals['drfPhase']

        # Create full sequence
        if not demo:
            self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
            samplingPeriod = self.expt.get_rx_ts()[0] # us
            BW = 1/samplingPeriod/hw.oversamplingFactor # MHz
            self.mapVals['bw'] = BW
            acqTime = nPoints[0]/BW        # us
            createSequence()

        # Run the experiment
        dataFull = []
        dummyData = []
        overData = []

        for ii in range(nScans):
            if plotSeq:
                self.expt.plot_sequence()
                plt.show()
                break
            else:
                print("Scan %s ..." % (ii+1))
                if not demo:
                    rxd, msgs = self.expt.run()
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
        if not demo: self.expt.__del__()

        # Process data
        if not plotSeq:
            print('Scans done!')
            self.mapVals['overData'] = overData

            # Fix the echo position using oversampled data
            if dummyPulses>0:
                dummyData = np.reshape(dummyData,  (nScans, nPH, nRD*hw.oversamplingFactor))
                dummyData = np.average(dummyData, axis=0)
                self.mapVals['dummyData'] = dummyData
                overData = np.reshape(overData, (nScans, 1, nPH,  nRD*hw.oversamplingFactor))
                for ii in range(nScans):
                    overData[ii, :, :, :] = self.fixEchoPosition(dummyData, overData[ii, :, :, :])

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
            self.mapVals['dataFull'] = dataFull
            self.mapVals['imgFull'] = imgFull

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
            self.mapVals['kSpace3D'] = data
            img=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
            self.mapVals['image3D'] = img
            data = np.reshape(data, (1, nPoints[0]*nPoints[1]))

            # Create sampled data
            kRD = np.reshape(kRD, (nPoints[0]*nPoints[1], 1))
            kPH = np.reshape(kPH, (nPoints[0]*nPoints[1], 1))
            data = np.reshape(data, (nPoints[0]*nPoints[1], 1))
            self.mapVals['kMax'] = kMax
            self.mapVals['sampled'] = np.concatenate((kRD, kPH, data), axis=1)

    def sequenceAnalysis(self, obj=''):
        self.saveRawData()
        if obj!='':
            nPoints = self.mapVals['nPoints']
            axesEnable = self.mapVals['axesEnable']
            if not hasattr(obj.parent, 'batch'):
                if (axesEnable[1] == 0 and axesEnable[2] == 0):
                    bw = self.mapVals['bw'] * 1e-3  # kHz
                    acqTime = self.mapVals['acqTime']  # ms
                    tVector = np.linspace(-acqTime / 2, acqTime / 2, nPoints[0])
                    sVector = self.mapVals['sampled'][:, 3]
                    fVector = np.linspace(-bw / 2, bw / 2, nPoints[0])
                    iVector = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(sVector)))

                    f_plotview = SpectrumPlot(fVector, [np.abs(iVector)], ['Spectrum magnitude'],
                                              "Frequency (kHz)", "Amplitude (a.u.)",
                                              "%s Spectrum" % (obj.sequence.mapVals['seqName']), )
                    t_plotview = SpectrumPlot(tVector, [np.abs(sVector), np.real(sVector), np.imag(sVector)],
                                              ['Magnitude', 'Real', 'Imaginary'],
                                              'Time (ms)', "Signal amplitude (mV)",
                                              "%s Signal" % (obj.sequence.mapVals['seqName']), )
                    obj.parent.plotview_layout.addWidget(t_plotview)
                    obj.parent.plotview_layout.addWidget(f_plotview)
                    obj.parent.f_plotview = f_plotview
                    obj.parent.t_plotview = t_plotview

                else:
                    # Create label with rawdata name
                    obj.label = QLabel(self.mapVals['fileName'])
                    obj.label.setAlignment(QtCore.Qt.AlignCenter)
                    obj.label.setStyleSheet("background-color: black;color: white")
                    obj.parent.plotview_layout.addWidget(obj.label)

                    # Plot image
                    obj.parent.plotview_layout.addWidget(pg.image(np.abs(self.mapVals['image3D'])))

                    # Plot k-space
                    obj.parent.plotview_layout.addWidget(pg.image(np.log10(np.abs(self.mapVals['kSpace3D']))))