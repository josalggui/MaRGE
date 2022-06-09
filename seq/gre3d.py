# -*- coding: utf-8 -*-
"""
Created on Sat Nov  13 13:45:05 2021

@author: José Miguel Algarín Guisado
MRILAB @ I3M
"""

import numpy as np
import experiment as ex
import scipy.signal as sig
import matplotlib.pyplot as plt
import pdb
import configs.hw_config as hw # Import the scanner hardware config
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from plotview.spectrumplot import SpectrumPlot # To plot nice 1d images
from PyQt5.QtWidgets import QLabel  # To set the figure title
from PyQt5 import QtCore            # To set the figure title
import pyqtgraph as pg              # To plot nice 3d images

class GRE3D(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(GRE3D, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='GRE3DInfo', val='GRE3D')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=4.0, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=500., field='SEQ')
        self.addParameter(key='fov', string='FOV (cm)', val=[12.0, 12.0, 12.0], field='IM')
        self.addParameter(key='dfov', string='dFOV (mm)', val=[0.0, 0.0, 0.0], field='IM')
        self.addParameter(key='nPoints', string='nPoints (rd, ph, sl)', val=[60, 60, 1], field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=1.0, field='SEQ')
        self.addParameter(key='axes', string='Axes', val=[0, 1, 2], field='IM')
        self.addParameter(key='axesEnable', string='Axes enable', val=[1, 1, 0], field='IM')
        self.addParameter(key='sweepMode', string='Sweep mode, 0->k20, 1->02k, 2->k2k', val=1, field='SEQ')
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=1.5, field='OTH')
        self.addParameter(key='dephGradTime', string='Rd dephasing time (ms)', val=1.0, field='OTH')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=1, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='parFourierFractionSl', string='Partial fourier fraction', val=1.0, field='OTH')
        self.addParameter(key='spoiler', string='Spoiler gradient', val=0, field='SEQ')


    # ******************************************************************************************************************
    # ******************************************************************************************************************
    # ******************************************************************************************************************


    def sequenceInfo(self):
        print(" ")
        print("3D GRE sequence")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")


    # ******************************************************************************************************************
    # ******************************************************************************************************************
    # ******************************************************************************************************************


    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        nPoints = np.array(self.mapVals['nPoints'])
        repetitionTime = self.mapVals['repetitionTime']
        return(nPoints[1]*nPoints[2]*repetitionTime*1e-3*nScans/60)  # minutes, scanTime


    #*********************************************************************************
    #*********************************************************************************
    #*********************************************************************************


    def sequenceRun(self, plotSeq):
        init_gpa=False, # Starts the gpa
        nScans = self.mapVals['nScans'] # NEX
        larmorFreq = self.mapVals['larmorFreq'] # MHz, Larmor frequency
        rfExAmp = self.mapVals['rfExAmp'] # a.u. rf excitation pulse amplitude
        rfExTime = self.mapVals['rfExTime'] # us, rf excitation pulse time
        echoTime = self.mapVals['echoTime'] # ms, TE
        repetitionTime = self.mapVals['repetitionTime'] # ms, TR
        fov = np.array(self.mapVals['fov']) # cm, FOV along readout, phase and slice
        dfov = np.array(self.mapVals['dfov']) # mm, Displacement of fov center
        nPoints = np.array(self.mapVals['nPoints']) # Number of points along readout, phase and slice
        acqTime = self.mapVals['acqTime'] # ms, Acquisition time
        axes = np.array(self.mapVals['axes']) # 0->x, 1->y and 2->z defined as [rd,ph,sl]
        rdGradTime = self.mapVals['rdGradTime'] # ms, readout rephasing gradient time
        dephGradTime = self.mapVals['dephGradTime'] # ms, Phase and slice dephasing time
        dummyPulses = self.mapVals['dummyPulses'] # Dummy pulses for T1 stabilization
        shimming = self.mapVals['shimming'] # a.u.*1e4, Shimming along the X,Y and Z axes
        parFourierFraction = self.mapVals['parFourierFractionSl'] # fraction of acquired k-space along phase direction
        spoiler = self.mapVals['spoiler'] # set 1 or 0 if you want or do not want to apply spoiler gradients

        freqCal = False
        demo = False

        # Conversion of variables to non-multiplied units
        larmorFreq = larmorFreq*1e6
        rfExTime = rfExTime*1e-6
        fov = np.array(fov)*1e-2
        dfov = np.array(dfov)*1e-3
        echoTime = echoTime*1e-3
        acqTime = acqTime*1e-3
        shimming = np.array(shimming)*1e-4
        nPoints = np.array(nPoints)
        repetitionTime = repetitionTime*1e-3
        rdGradTime = rdGradTime*1e-3
        dephGradTime = dephGradTime*1e-3

        # Miscellaneous
        larmorFreq = larmorFreq*1e-6
        gradRiseTime = 100e-6       # Estimated gradient rise time
        gSteps = int(gradRiseTime*1e6/5)*0+1
        addRdPoints = 5             # Initial rd points to avoid artifact at the begining of rd
        resolution = fov/nPoints
        randFactor = 0.
        axesEnable = np.array([1, 1, 1])
        for ii in range(3):
            if nPoints[ii]==1: axesEnable[ii] = 0
        if fov[0]>1 and nPoints[1]==1: axesEnable[0] = 0
        self.mapVals['randFactor'] = randFactor
        self.mapVals['resolution'] = resolution
        self.mapVals['gradRiseTime'] = gradRiseTime
        self.mapVals['addRdPoints'] = addRdPoints

        # Matrix size
        nRD = nPoints[0]+2*addRdPoints
        nPH = nPoints[1]
        nSL = nPoints[2]

        # parAcqLines
        nSLreal = int(nPoints[2]*parFourierFraction)
        parAcqLines = int(nSLreal-nPoints[2]/2)
        self.mapVals['parAcqLines'] = parAcqLines
        del nSLreal

        # BW
        BW = nPoints[0]/acqTime*1e-6
        BWov = BW*hw.oversamplingFactor
        samplingPeriod = 1/BWov
        self.mapVals['samplingPeriod'] = samplingPeriod

        # Check if dephasing grad time is ok
        maxDephGradTime = echoTime-(rfExTime+rdGradTime)-3*gradRiseTime
        if dephGradTime==0 or dephGradTime>maxDephGradTime:
            dephGradTime = maxDephGradTime

        # Max gradient amplitude
        rdGradAmplitude = nPoints[0]/(hw.gammaB*fov[0]*acqTime)
        rdDephAmplitude = -rdGradAmplitude*(rdGradTime+gradRiseTime)/(2*(dephGradTime+gradRiseTime))
        phGradAmplitude = nPH/(2*hw.gammaB*fov[1]*(dephGradTime+gradRiseTime))*axesEnable[1]
        slGradAmplitude = nSL/(2*hw.gammaB*fov[2]*(dephGradTime+gradRiseTime))*axesEnable[2]
        self.mapVals['rdGradAmplitude'] = rdGradAmplitude
        self.mapVals['rdDephAmplitude'] = rdDephAmplitude
        self.mapVals['phGradAmplitude'] = phGradAmplitude
        self.mapVals['slGradAmplitude'] = slGradAmplitude

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
        self.mapVals['phGradients'] = phGradients
        self.mapVals['slGradients'] = slGradients

        # Changing time parameters to us
        rfExTime = rfExTime*1e6
        echoTime = echoTime*1e6
        repetitionTime = repetitionTime*1e6
        gradRiseTime = gradRiseTime*1e6
        dephGradTime = dephGradTime*1e6
        rdGradTime = rdGradTime*1e6
        scanTime = nRepetitions*repetitionTime
        self.mapVals['scanTime'] = scanTime*nSL*1e-6

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
            self.iniSequence(20, shimming, rewrite=rewrite)

            # Run sequence batch
            while acqPoints+nRD<=hw.maxRdPoints and orders<=hw.maxOrders and repeIndexGlobal<nRepetitions:
                # Initialize time
                tEx = 20e3+repetitionTime*repeIndex

                # First I do a noise measurement
                if repeIndex==0:
                    t0 = tEx-4*acqTime
                    self.rxGate(t0, acqTime+2*addRdPoints/BW)
                    acqPoints += nRD

                # Excitation pulse
                t0 = tEx-hw.blkTime-rfExTime/2
                self.rfRecPulse(t0, rfExTime, rfExAmp, 0.)

                # Dephasing readout, phase and slice
                if (repeIndex==0 or repeIndex>=dummyPulses):
                    t0 = tEx+rfExTime/2-hw.gradDelay
                    self.gradTrap(t0, gradRiseTime, dephGradTime, rdDephAmplitude, gSteps,  axes[0],  shimming)
                    orders = orders+gSteps*2
                if repeIndex>=dummyPulses:
                    t0 = tEx+rfExTime/2-hw.gradDelay
                    self.gradTrap(t0, gradRiseTime,  dephGradTime, phGradients[phIndex], gSteps,  axes[1], shimming)
                    self.gradTrap(t0, gradRiseTime,  dephGradTime, slGradients[slIndex], gSteps,  axes[2], shimming)
                    orders = orders+gSteps*4

                # Rephasing readout gradient
                if (repeIndex==0 or repeIndex>=dummyPulses):
                    t0 = tEx+echoTime-rdGradTime/2-gradRiseTime-hw.gradDelay
                    self.gradTrap(t0, gradRiseTime, rdGradTime, rdGradAmplitude, gSteps, axes[0], shimming)
                    orders = orders+gSteps*2

                # Rx gate
                if (repeIndex==0 or repeIndex>=dummyPulses):
                    t0 = tEx+echoTime-acqTime/2-addRdPoints/BW
                    self.rxGate(t0, acqTime+2*addRdPoints/BW)
                    acqPoints += nRD

                # Spoiler
                if spoiler:
                    t0 = tEx+echoTime+rdGradTime/2+gradRiseTime-hw.gradDelay
                    self.gradTrap(t0, gradRiseTime, dephGradTime, -rdDephAmplitude, gSteps,  axes[0],  shimming)
                    orders = orders+gSteps*2

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
            self.endSequence(repeIndex*repetitionTime)

            # Return the output variables
            return(phIndex, slIndex, repeIndexGlobal, acqPoints)

        # Calibrate frequency
        if freqCal and (not plotSeq) and (not demo):
            larmorFreq = self.freqCalibration(bw=0.05)
            larmorFreq = self.freqCalibration(bw=0.005)
            drfPhase = self.mapVals['drfPhase']

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
                self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
                samplingPeriod = self.expt.get_rx_ts()[0]
                BW = 1/samplingPeriod/hw.oversamplingFactor
                self.mapVals['bw'] = BW
                acqTime = nPoints[0]/BW        # us
                phIndex, slIndex, repeIndexGlobal, aa = createSequence(phIndex=phIndex,
                                                                   slIndex=slIndex,
                                                                   repeIndexGlobal=repeIndexGlobal,
                                                                   rewrite=False)
                repeIndexArray = np.concatenate((repeIndexArray, np.array([repeIndexGlobal-1])), axis=0)
                acqPointsPerBatch.append(aa)
                self.expt.plot_sequence()
            else:
                phIndex, slIndex, repeIndexGlobal, aa, dataA = createSequenceDemo(phIndex=phIndex,
                                                                   slIndex=slIndex,
                                                                   repeIndexGlobal=repeIndexGlobal)
                repeIndexArray = np.concatenate((repeIndexArray, np.array([repeIndexGlobal-1])), axis=0)
                acqPointsPerBatch.append(aa)

            for ii in range(nScans):
                print('Batch ', nBatches, ', Scan ', ii, ' runing...')
                if not demo:
                    if plotSeq==1:                  # What is the meaning of plotSeq??
                        print('Ploting sequence...')
                        self.expt.plot_sequence()
                        plt.show()
                        self.expt.__del__()
                        break
                    else:
                        rxd, msgs = self.expt.run()
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
                else: # Demo
                    data = dataA
                    noise = np.concatenate((noise, data[0:nRD*hw.oversamplingFactor]), axis = 0)
                    data = data[nRD*hw.oversamplingFactor::]
                    # Get data
                    if dummyPulses>0:
                        dummyData = np.concatenate((dummyData, data[0:nRD*hw.oversamplingFactor]), axis = 0)
                        overData = np.concatenate((overData, data[nRD*hw.oversamplingFactor::]), axis = 0)
                    else:
                        overData = np.concatenate((overData, data), axis = 0)

            if not demo: self.expt.__del__()
            if plotSeq ==1:
                break
        del aa

        if not plotSeq:
            acqPointsPerBatch = (acqPointsPerBatch-nRD*(dummyPulses>0)-nRD)*nScans
            print('Scans done!')
            self.mapVals['noiseData'] = noise
            self.mapVals['overData'] = overData

            # Fix the echo position using oversampled data
            if dummyPulses>0:
                dummyData = np.reshape(dummyData,  (nBatches*nScans, 1, nRD*hw.oversamplingFactor))
                dummyData = np.average(dummyData, axis=0)
                self.mapVals['dummyData'] = dummyData
                overData = np.reshape(overData, (-1, 1, nRD*hw.oversamplingFactor))
                overData = self.fixEchoPosition(dummyData, overData)
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

            # Get individual images
            dataFull = np.reshape(dataFull, (nScans, nSL, nPH, nRD))
            dataFull = dataFull[:, :, :, indkrd0-int(nPoints[0]/2):indkrd0+int(nPoints[0]/2)]
            imgFull = dataFull*0
            for ii in range(nScans):
                imgFull[ii, :, :, :] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataFull[ii, :, :, :])))
            self.mapVals['dataFull'] = dataFull
            self.mapVals['imgFull'] = imgFull

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
            self.mapVals['kSpace3D'] = data
            img=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
            self.mapVals['image3D'] = img
            data = np.reshape(data, (1, nPoints[0]*nPoints[1]*nPoints[2]))

            # Create sampled data
            kRD = np.reshape(kRD, (nPoints[0]*nPoints[1]*nPoints[2], 1))
            kPH = np.reshape(kPH, (nPoints[0]*nPoints[1]*nPoints[2], 1))
            kSL = np.reshape(kSL, (nPoints[0]*nPoints[1]*nPoints[2], 1))
            data = np.reshape(data, (nPoints[0]*nPoints[1]*nPoints[2], 1))
            self.mapVals['kMax'] = kMax
            self.mapVals['sampled'] = np.concatenate((kRD, kPH, kSL, data), axis=1)
            data = np.reshape(data, (nPoints[2], nPoints[1], nPoints[0]))

    def sequenceAnalysis(self, obj):
        self.saveRawData()
        nPoints = self.mapVals['nPoints']
        axesEnable = self.mapVals['axesEnable']
        if not hasattr(obj.parent, 'batch'):
            if (axesEnable[1] == 0 and axesEnable[2] == 0):
                bw = self.mapVals['bw']*1e-3 # kHz
                acqTime = self.mapVals['acqTime'] # ms
                tVector = np.linspace(-acqTime/2, acqTime/2, nPoints[0])
                sVector = self.mapVals['sampled'][:, 3]
                fVector = np.linspace(-bw/2, bw/2, nPoints[0])
                iVector = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(sVector)))

                f_plotview = SpectrumPlot(fVector, np.abs(iVector), [], [],
                                          "Frequency (kHz)", "Amplitude (a.u.)",
                                          "%s Spectrum" % (obj.sequence.mapVals['seqName']), )
                t_plotview = SpectrumPlot(tVector, np.abs(sVector), np.real(sVector),
                                          np.imag(sVector), 'Time (ms)', "Signal amplitude (mV)",
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