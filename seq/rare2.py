# -*- coding: utf-8 -*-
"""
Created on Thu Jun 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@Summary: first attempt to do sequence as class
"""

import numpy as np
import experiment as ex
import matplotlib.pyplot as plt
import scipy.signal as sig
import pdb
import configs.hw_config as hw # Import the scanner hardware config
import mrilabMethods.mrilabMethods as mri   # This import all methods inside the mrilabMethods module
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
st = pdb.set_trace


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class RARE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        # Input the parameters
        self.addParameter(key='seqName', string='RARE')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='larmorFreq', string='Larmor frequency', val=3.08, unit='MHz', field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude', val=0.3, unit='a.u.', field='RF')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude', val=0.3, unit='a.u.', field='RF')
        self.addParameter(key='rfReTime', string='RF excitation time', val=30, unit='ms', field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time', val=60, unit='ms', field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing', val=10, unit='ms', field='SEQ')
        self.addParameter(key='preExTime', string='Preexitation time', val=0, unit='ms', field='SEQ')
        self.addParameter(key='inversionTime', string='Inversion time', val=0, unit='ms', field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time', val=500, unit='ms', field='SEQ')
        self.addParameter(key='fov', string='FOV', val=0, unit='mm', field='IM')
        self.addParameter(key='dfov', string='dFOV', val=0, unit='mm', field='IM')
        self.addParameter(key='nPoints', string='nPoints (rd, ph, sl)', val=[60, 0, 0], field='IM')
        self.addParameter(key='etl', string='Echo train length', val=5, field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time', val=4, unit='ms', field='SEQ')
        self.addParameter(key='axes', string='Axes', val=[0, 1, 2], field='IM')
        self.addParameter(key='axesEnable', string='Axes enable', val=[0, 0, 0], field='IM')
        self.addParameter(key='sweepMode', string='Sweep mode, 0->k20, 1->02k, 2->k2k', val=1, field='SEQ')
        self.addParameter(key='rdGradTime', string='Rd gradient time', val=5, unit='ms', field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time', val=1, unit='ms', field='OTH')
        self.addParameter(key='phGradTime', string='Ph gradient time', val=1, unit='ms', field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1., field='OTH')
        self.addParameter(key='drfPhase', string='Phase of exciation pulse', val=0., unit='degrees', field='RF')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=1, field='SEQ')
        self.addParameter(key='shimming', string='Shimming', val=[-70, -90, 10], unit='*1e4', field='OTH')
        self.addParameter(key='parFourierFraction', string='Partial fourier fraction', val=1., field='OTH')


        #*********************************************************************************
        #*********************************************************************************
        #*********************************************************************************


    def sequencePlot(self):
        init_gpa = False

        # Create experiment
        self.expt = ex.Experiment(lo_freq=3.0, rx_t=20, init_gpa=False, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))

        # Introduce instructions into de code
        self.iniSequence(20, shimming, rewrite=True)
        self.rxGate(1000, 1000)
        self.endSequence(3000)

        # Run experiment
        self.rxd, self.msgs = expt.run()


    # def sequenceRun(self):
    #     init_gpa=False # Starts the gpa
    #     nScans = self.nScans # NEX
    #     larmorFreq = self.larmorFreq # MHz, Larmor frequency
    #     rfExAmp = self.rfExAmp # a.u., rf excitation pulse amplitude
    #     rfReAmp = self.rfReAmp # a.u., rf refocusing pulse amplitude
    #     rfExTime = self.rfExTime # us, rf excitation pulse time
    #     rfReTime = self.rfReTime # us, rf refocusing pulse time
    #     echoSpacing = self.echoSpacing # ms, time between echoes
    #     preExTime = self.preExTime # ms, Time from preexcitation pulse to inversion pulse
    #     inversionTime = self.inversionTime # ms, Inversion recovery time
    #     repetitionTime = self.repetitionTime # ms, TR
    #     fov = self.fov # mm, FOV along readout, phase and slice
    #     dfov = self.dfov # mm, displacement of fov center
    #     nPoints = self.nPoints # Number of points along readout, phase and slice
    #     etl = self.etl # Echo train length
    #     acqTime = self.acqTime # ms, acquisition time
    #     axes = self.axes # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    #     axesEnable = self.axesEnable # 1-> Enable, 0-> Disable
    #     sweepMode = self.sweepMode # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    #     rdGradTime = self.rdGradTime  # ms, readout gradient time
    #     rdDephTime = self.rdDephTime  # ms, readout dephasing time
    #     phGradTime = self.phGradTime # ms, phase and slice dephasing time
    #     rdPreemphasis = self.rdPreemphasis # readout dephasing gradient is multiplied by this factor
    #     drfPhase = self.drfPhase # degrees, phase of the excitation pulse
    #     dummyPulses = self.dummyPulses # number of dummy pulses for T1 stabilization
    #     shimming = self.shimming # a.u.*1e4, shimming along the X,Y and Z axes
    #     parFourierFraction = self.parFourierFractionSl # fraction of acquired k-space along phase direction
    #
    #     freqCal = True
    #     demo = False
    #
    #     # rawData fields
    #     rawData = {}
    #
    #     # Conversion of variables to non-multiplied units
    #     larmorFreq = larmorFreq*1e6
    #     rfExTime = rfExTime*1e-6
    #     rfReTime = rfReTime*1e-6
    #     fov = np.array(fov)*1e-3
    #     dfov = np.array(dfov)*1e-3
    #     echoSpacing = echoSpacing*1e-3
    #     acqTime = acqTime*1e-3
    #     shimming = np.array(shimming)*1e-4
    #     repetitionTime= repetitionTime*1e-3
    #     preExTime = preExTime*1e-3
    #     inversionTime = inversionTime*1e-3
    #     rdGradTime = rdGradTime*1e-3
    #     rdDephTime = rdDephTime*1e-3
    #     phGradTime = phGradTime*1e-3
    #
    # # Inputs for rawData
    #     rawData['seqName'] = 'RARE'
    #     rawData['nScans'] = nScans
    #     rawData['larmorFreq'] = larmorFreq      # Larmor frequency
    #     rawData['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    #     rawData['rfReAmp'] = rfReAmp             # rf refocusing pulse amplitude
    #     rawData['rfExTime'] = rfExTime          # rf excitation pulse time
    #     rawData['rfReTime'] = rfReTime            # rf refocusing pulse time
    #     rawData['echoSpacing'] = echoSpacing        # time between echoes
    #     rawData['preExTime'] = preExTime
    #     rawData['inversionTime'] = inversionTime       # Inversion recovery time
    #     rawData['repetitionTime'] = repetitionTime     # TR
    #     rawData['fov'] = fov           # FOV along readout, phase and slice
    #     rawData['dfov'] = dfov            # Displacement of fov center
    #     rawData['nPoints'] = nPoints                 # Number of points along readout, phase and slice
    #     rawData['etl'] = etl                    # Echo train length
    #     rawData['acqTime'] = acqTime             # Acquisition time
    #     rawData['axesOrientation'] = axes       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    #     rawData['axesEnable'] = axesEnable # 1-> Enable, 0-> Disable
    #     rawData['sweepMode'] = sweepMode               # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    #     rawData['rdPreemphasis'] = rdPreemphasis
    #     rawData['drfPhase'] = drfPhase
    #     rawData['dummyPulses'] = dummyPulses                    # Dummy pulses for T1 stabilization
    #     rawData['parFourierFraction'] = parFourierFraction
    #     rawData['rdDephTime'] = rdDephTime
    #     rawData['shimming'] = shimming
    #
    #     # Miscellaneous
    #     larmorFreq = larmorFreq*1e-6
    #     gradRiseTime = 400e-6       # Estimated gradient rise time
    #     gSteps = int(gradRiseTime*1e6/5)*0+1
    #     addRdPoints = 10             # Initial rd points to avoid artifact at the begining of rd
    #     randFactor = 0e-3                        # Random amplitude to add to the phase gradients
    #     if rfReAmp==0:
    #         rfReAmp = rfExAmp
    #     if rfReTime==0:
    #         rfReTime = 2*rfExTime
    #     resolution = fov/nPoints
    #     rawData['resolution'] = resolution
    #     rawData['gradRiseTime'] = gradRiseTime
    #     rawData['randFactor'] = randFactor
    #     rawData['addRdPoints'] = addRdPoints
    #
    #     # Matrix size
    #     nRD = nPoints[0]+2*addRdPoints
    #     nPH = nPoints[1]*axesEnable[1]+(1-axesEnable[1])
    #     nSL = nPoints[2]*axesEnable[2]+(1-axesEnable[2])
    #
    #     # ETL if nPH = 1
    #     if etl>nPH:
    #         etl = nPH
    #
    #     # parAcqLines in case parAcqLines = 0
    #     parAcqLines = int(int(nPoints[2]*parFourierFraction)-nPoints[2]/2)
    #     rawData['partialAcquisition'] = parAcqLines
    #
    #     # BW
    #     BW = nPoints[0]/acqTime*1e-6
    #     BWov = BW*hw.oversamplingFactor
    #     samplingPeriod = 1/BWov
    #
    #     # Readout gradient time
    #     if rdGradTime>0 and rdGradTime<acqTime:
    #         rdGradTime = acqTime
    #     rawData['rdGradTime'] = rdGradTime
    #
    #     # Phase and slice de- and re-phasing time
    #     if phGradTime==0 or phGradTime>echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime:
    #         phGradTime = echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime
    #     rawData['phGradTime'] = phGradTime
    #
    #     # Max gradient amplitude
    #     rdGradAmplitude = nPoints[0]/(hw.gammaB*fov[0]*acqTime)*axesEnable[0]
    #     phGradAmplitude = nPH/(2*hw.gammaB*fov[1]*(phGradTime+gradRiseTime))*axesEnable[1]
    #     slGradAmplitude = nSL/(2*hw.gammaB*fov[2]*(phGradTime+gradRiseTime))*axesEnable[2]
    #     rawData['rdGradAmplitude'] = rdGradAmplitude
    #     rawData['phGradAmplitude'] = phGradAmplitude
    #     rawData['slGradAmplitude'] = slGradAmplitude
    #
    #     # Readout dephasing amplitude
    #     rdDephAmplitude = 0.5*rdGradAmplitude*(gradRiseTime+rdGradTime)/(gradRiseTime+rdDephTime)
    #     rawData['rdDephAmplitude'] = rdDephAmplitude
    #
    #     # Phase and slice gradient vector
    #     phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
    #     slGradients = np.linspace(-slGradAmplitude,slGradAmplitude,num=nSL,endpoint=False)
    #
    #     # Now fix the number of slices to partailly acquired k-space
    #     nSL = (int(nPoints[2]/2)+parAcqLines)*axesEnable[2]+(1-axesEnable[2])
    #
    #     # Add random displacemnt to phase encoding lines
    #     for ii in range(nPH):
    #         if ii<np.ceil(nPH/2-nPH/20) or ii>np.ceil(nPH/2+nPH/20):
    #             phGradients[ii] = phGradients[ii]+randFactor*np.random.randn()
    #     kPH = hw.gammaB*phGradients*(gradRiseTime+phGradTime)
    #     rawData['phGradients'] = phGradients
    #     rawData['slGradients'] = slGradients
    #
    #     # Set phase vector to given sweep mode
    #     ind = mri.getIndex(etl, nPH, sweepMode)
    #     rawData['sweepOrder'] = ind
    #     phGradients = phGradients[ind]
    #
    #     def createSequenceDemo(phIndex=0, slIndex=0, repeIndexGlobal=0, rewrite=True):
    #         repeIndex = 0
    #         acqPoints = 0
    #         orders = 0
    #         data = []
    #         while acqPoints+etl*nRD<=hw.maxRdPoints and orders<=hw.maxOrders and repeIndexGlobal<nRepetitions:
    #             if repeIndex==0:
    #                 acqPoints += nRD
    #                 data = np.concatenate((data, np.random.randn(nRD*hw.oversamplingFactor)), axis = 0)
    #
    #             for echoIndex in range(etl):
    #                 if (repeIndex==0 or repeIndex>=dummyPulses):
    #                     acqPoints += nRD
    #                     data = np.concatenate((data, np.random.randn(nRD*hw.oversamplingFactor)), axis = 0)
    #
    #                 # Update the phase and slice gradient
    #                 if repeIndex>=dummyPulses:
    #                     if phIndex == nPH-1:
    #                         phIndex = 0
    #                         slIndex += 1
    #                     else:
    #                         phIndex += 1
    #             if repeIndex>=dummyPulses: repeIndexGlobal += 1 # Update the global repeIndex
    #             repeIndex+=1 # Update the repeIndex after the ETL
    #
    #         # Return the output variables
    #         return(phIndex, slIndex, repeIndexGlobal, acqPoints, data)
    #
    #     def createSequence(phIndex=0, slIndex=0, repeIndexGlobal=0, rewrite=True):
    #         repeIndex = 0
    #         if rdGradTime==0:   # Check if readout gradient is dc or pulsed
    #             dc = True
    #         else:
    #             dc = False
    #         acqPoints = 0
    #         orders = 0
    #         # Check in case of dummy pulse fill the cache
    #         if (dummyPulses>0 and etl*nRD*2>hw.maxRdPoints) or (dummyPulses==0 and etl*nRD>hw.maxRdPoints):
    #             print('ERROR: Too many acquired points.')
    #             return()
    #         # Set shimming
    #         mri.iniSequence(expt, 20, shimming, rewrite=rewrite)
    #         while acqPoints+etl*nRD<=hw.maxRdPoints and orders<=hw.maxOrders and repeIndexGlobal<nRepetitions:
    #             # Initialize time
    #             tEx = 20e3+repetitionTime*repeIndex+inversionTime+preExTime
    #
    #             # First I do a noise measurement.
    #             if repeIndex==0:
    #                 t0 = tEx-preExTime-inversionTime-4*acqTime
    #                 mri.rxGate(expt, t0, acqTime+2*addRdPoints/BW)
    #                 acqPoints += nRD
    #
    #             # Pre-excitation pulse
    #             if repeIndex>=dummyPulses and preExTime!=0:
    #                 t0 = tEx-preExTime-inversionTime-rfExTime/2-hw.blkTime
    #                 mri.rfRecPulse(expt, t0, rfExTime, rfExAmp/90*90, 0)
    #                 mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, preExTime*0.5, -0.005, gSteps, axes[0], shimming)
    #                 mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, preExTime*0.5, -0.005, gSteps, axes[1], shimming)
    #                 mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, preExTime*0.5, -0.005, gSteps, axes[2], shimming)
    #                 orders = orders+gSteps*6
    #
    #             # Inversion pulse
    #             if repeIndex>=dummyPulses and inversionTime!=0:
    #                 t0 = tEx-inversionTime-rfReTime/2-hw.blkTime
    #                 mri.rfPulse(expt, t0, rfReTime, rfReAmp/180*180, 0)
    #                 mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[0], shimming)
    #                 mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[1], shimming)
    #                 mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[2], shimming)
    #                 orders = orders+gSteps*6
    #
    #             # DC gradient if desired
    #             if (repeIndex==0 or repeIndex>=dummyPulses) and dc==True:
    #                 t0 = tEx-10e3
    #                 mri.gradTrap(expt, t0, gradRiseTime, 10e3+echoSpacing*(etl+1), rdGradAmplitude, gSteps, axes[0], shimming)
    #                 orders = orders+gSteps*2
    #
    #             # Excitation pulse
    #             t0 = tEx-hw.blkTime-rfExTime/2
    #             mri.rfRecPulse(expt, t0,rfExTime,rfExAmp,drfPhase)
    #
    #             # Dephasing readout
    #             if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False:
    #                 t0 = tEx+rfExTime/2-hw.gradDelay
    #                 mri.gradTrap(expt, t0, gradRiseTime, rdDephTime, rdDephAmplitude*rdPreemphasis, gSteps, axes[0], shimming)
    #                 orders = orders+gSteps*2
    #
    #             # Echo train
    #             for echoIndex in range(etl):
    #                 tEcho = tEx+echoSpacing*(echoIndex+1)
    #
    #                 # Refocusing pulse
    #                 t0 = tEcho-echoSpacing/2-rfReTime/2-hw.blkTime
    #                 mri.rfRecPulse(expt, t0, rfReTime, rfReAmp, drfPhase+np.pi/2)
    #
    #                 # Dephasing phase and slice gradients
    #                 if repeIndex>=dummyPulses:         # This is to account for dummy pulses
    #                     t0 = tEcho-echoSpacing/2+rfReTime/2-hw.gradDelay
    #                     mri.gradTrap(expt, t0, gradRiseTime, phGradTime, phGradients[phIndex], gSteps, axes[1], shimming)
    #                     mri.gradTrap(expt, t0, gradRiseTime, phGradTime, slGradients[slIndex], gSteps, axes[2], shimming)
    #                     orders = orders+gSteps*4
    #
    #                 # Readout gradient
    #                 if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False:         # This is to account for dummy pulses
    #                     t0 = tEcho-rdGradTime/2-gradRiseTime-hw.gradDelay
    #                     mri.gradTrap(expt, t0, gradRiseTime, rdGradTime, rdGradAmplitude, gSteps, axes[0], shimming)
    #                     orders = orders+gSteps*2
    #
    #                 # Rx gate
    #                 if (repeIndex==0 or repeIndex>=dummyPulses):
    #                     t0 = tEcho-acqTime/2-addRdPoints/BW
    #                     mri.rxGate(expt, t0, acqTime+2*addRdPoints/BW)
    #                     acqPoints += nRD
    #
    #                 # Rephasing phase and slice gradients
    #                 t0 = tEcho+acqTime/2+addRdPoints/BW-hw.gradDelay
    #                 if (echoIndex<etl-1 and repeIndex>=dummyPulses):
    #                     mri.gradTrap(expt, t0, gradRiseTime, phGradTime, -phGradients[phIndex], gSteps, axes[1], shimming)
    #                     mri.gradTrap(expt, t0, gradRiseTime, phGradTime, -slGradients[slIndex], gSteps, axes[2], shimming)
    #                     orders = orders+gSteps*4
    #                 elif(echoIndex==etl-1 and repeIndex>=dummyPulses):
    #                     mri.gradTrap(expt, t0, gradRiseTime, phGradTime, +phGradients[phIndex], gSteps, axes[1], shimming)
    #                     mri.gradTrap(expt, t0, gradRiseTime, phGradTime, +slGradients[slIndex], gSteps, axes[2], shimming)
    #                     orders = orders+gSteps*4
    #
    #                 # Update the phase and slice gradient
    #                 if repeIndex>=dummyPulses:
    #                     if phIndex == nPH-1:
    #                         phIndex = 0
    #                         slIndex += 1
    #                     else:
    #                         phIndex += 1
    #             if repeIndex>=dummyPulses: repeIndexGlobal += 1 # Update the global repeIndex
    #             repeIndex+=1 # Update the repeIndex after the ETL
    #
    #         # Turn off the gradients after the end of the batch
    #         mri.endSequence(expt, repeIndex*repetitionTime)
    #
    #         # Return the output variables
    #         return(phIndex, slIndex, repeIndexGlobal, acqPoints)
    #
    #
    #     # Changing time parameters to us
    #     rfExTime = rfExTime*1e6
    #     rfReTime = rfReTime*1e6
    #     echoSpacing = echoSpacing*1e6
    #     repetitionTime = repetitionTime*1e6
    #     gradRiseTime = gradRiseTime*1e6
    #     phGradTime = phGradTime*1e6
    #     rdGradTime = rdGradTime*1e6
    #     rdDephTime = rdDephTime*1e6
    #     inversionTime = inversionTime*1e6
    #     preExTime = preExTime*1e6
    #     nRepetitions = int(nSL*nPH/etl)
    #     scanTime = nRepetitions*repetitionTime
    #     rawData['scanTime'] = scanTime*nSL*1e-6
    #
    #     # Calibrate frequency
    #     if (not demo) and freqCal and (not plotSeq):
    #         mri.freqCalibration(rawData, bw=0.05)
    #         mri.freqCalibration(rawData, bw=0.005)
    #         larmorFreq = rawData['larmorFreq']*1e-6
    #         drfPhase = rawData['drfPhase']
    #
    #     # Create full sequence
    #     # Run the experiment
    #     dataFull = []
    #     dummyData = []
    #     overData = []
    #     noise = []
    #     nBatches = 0
    #     repeIndexArray = np.array([0])
    #     repeIndexGlobal = repeIndexArray[0]
    #     phIndex = 0
    #     slIndex = 0
    #     acqPointsPerBatch = []
    #     while repeIndexGlobal<nRepetitions:
    #
    #         nBatches += 1
    #         if not demo:
    #             expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    #             samplingPeriod = expt.get_rx_ts()[0]
    #             BW = 1/samplingPeriod/hw.oversamplingFactor
    #             acqTime = nPoints[0]/BW        # us
    #             phIndex, slIndex, repeIndexGlobal, aa = createSequence(phIndex=phIndex,
    #                                                                slIndex=slIndex,
    #                                                                repeIndexGlobal=repeIndexGlobal,
    #                                                                rewrite=nBatches==1)
    #             repeIndexArray = np.concatenate((repeIndexArray, np.array([repeIndexGlobal-1])), axis=0)
    #             acqPointsPerBatch.append(aa)
    #         else:
    #             phIndex, slIndex, repeIndexGlobal, aa, dataA = createSequenceDemo(phIndex=phIndex,
    #                                                                slIndex=slIndex,
    #                                                                repeIndexGlobal=repeIndexGlobal,
    #                                                                rewrite=nBatches==1)
    #             repeIndexArray = np.concatenate((repeIndexArray, np.array([repeIndexGlobal-1])), axis=0)
    #             acqPointsPerBatch.append(aa)
    #
    #         for ii in range(nScans):
    #             if not demo:
    #                 if plotSeq==1:                  # What is the meaning of plotSeq??
    #                     print('Ploting sequence...')
    #                     expt.plot_sequence()
    #                     plt.show()
    #                     expt.__del__()
    #                     break
    #                 elif plotSeq==0:
    #                     print('Batch ', nBatches, ', Scan ', ii, ' runing...')
    #                     rxd, msgs = expt.run()
    #                     rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
    #                     # Get noise data
    #                     noise = np.concatenate((noise, rxd['rx0'][0:nRD*hw.oversamplingFactor]), axis = 0)
    #                     rxd['rx0'] = rxd['rx0'][nRD*hw.oversamplingFactor::]
    #                     # Get data
    #                     if dummyPulses>0:
    #                         dummyData = np.concatenate((dummyData, rxd['rx0'][0:nRD*etl*hw.oversamplingFactor]), axis = 0)
    #                         overData = np.concatenate((overData, rxd['rx0'][nRD*etl*hw.oversamplingFactor::]), axis = 0)
    #                     else:
    #                         overData = np.concatenate((overData, rxd['rx0']), axis = 0)
    #             else:
    #                 print('Batch ', nBatches, ', Scan ', ii, ' runing...')
    #                 data = dataA
    #                 noise = np.concatenate((noise, data[0:nRD*hw.oversamplingFactor]), axis = 0)
    #                 data = data[nRD*hw.oversamplingFactor::]
    #                 # Get data
    #                 if dummyPulses>0:
    #                     dummyData = np.concatenate((dummyData, data[0:nRD*etl*hw.oversamplingFactor]), axis = 0)
    #                     overData = np.concatenate((overData, data[nRD*etl*hw.oversamplingFactor::]), axis = 0)
    #                 else:
    #                     overData = np.concatenate((overData, data), axis = 0)
    #
    #         if not demo: expt.__del__()
    #         if plotSeq ==1:
    #             break
    #     del aa
    #
    #     if plotSeq ==0:
    #         acqPointsPerBatch= (np.array(acqPointsPerBatch)-etl*nRD*(dummyPulses>0)-nRD)*nScans
    #         print('Scans done!')
    #         rawData['noiseData'] = noise
    #         rawData['overData'] = overData
    #
    #         # Fix the echo position using oversampled data
    #         if dummyPulses>0:
    #             dummyData = np.reshape(dummyData,  (nBatches*nScans, etl, nRD*hw.oversamplingFactor))
    #             dummyData = np.average(dummyData, axis=0)
    #             rawData['dummyData'] = dummyData
    #             overData = np.reshape(overData, (-1, etl, nRD*hw.oversamplingFactor))
    #             overData = mri.fixEchoPosition(dummyData, overData)
    #             overData = np.reshape(overData, -1)
    #
    #         # Generate dataFull
    #         dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
    #         if nBatches>1:
    #             dataFullA = dataFull[0:sum(acqPointsPerBatch[0:-1])]
    #             dataFullB = dataFull[sum(acqPointsPerBatch[0:-1])::]
    #
    #         # Reorganize dataFull
    #         dataProv = np.zeros([nScans,nSL*nPH*nRD])
    #         dataProv = dataProv+1j*dataProv
    #         dataFull = np.reshape(dataFull, (nBatches, nScans, -1, nRD))
    #         if nBatches>1:
    #             dataFullA = np.reshape(dataFullA, (nBatches-1, nScans, -1, nRD))
    #             dataFullB = np.reshape(dataFullB, (1, nScans, -1, nRD))
    #         for scan in range(nScans):
    #             if nBatches>1:
    #                 dataProv[ii, :] = np.concatenate((np.reshape(dataFullA[:,ii,:,:],-1), np.reshape(dataFullB[:,ii,:,:],-1)), axis=0)
    #             else:
    #                 dataProv[ii, :] = np.reshape(dataFull[:,ii,:,:],-1)
    #         dataFull = np.reshape(dataProv,-1)
    #
    #         # Get index for krd = 0
    #         # Average data
    #         dataProv = np.reshape(dataFull, (nScans, nRD*nPH*nSL))
    #         dataProv = np.average(dataProv, axis=0)
    #         # Reorganize the data acording to sweep mode
    #         dataProv = np.reshape(dataProv, (nSL, nPH, nRD))
    #         dataTemp = dataProv*0
    #         for ii in range(nPH):
    #             dataTemp[:, ind[ii], :] = dataProv[:,  ii, :]
    #         dataProv = dataTemp
    #         # Check where is krd = 0
    #         dataProv = dataProv[int(nPoints[2]/2), int(nPH/2), :]
    #         indkrd0 = np.argmax(np.abs(dataProv))
    #         if  indkrd0 < nRD/2-addRdPoints or indkrd0 > nRD/2+addRdPoints:
    #             indkrd0 = int(nRD/2)
    #
    #         # Get individual images
    #         dataFull = np.reshape(dataFull, (nScans, nSL, nPH, nRD))
    #         dataFull = dataFull[:, :, :, indkrd0-int(nPoints[0]/2):indkrd0+int(nPoints[0]/2)]
    #         dataTemp = dataFull*0
    #         for ii in range(nPH):
    #             dataTemp[:, :, ind[ii], :] = dataFull[:, :,  ii, :]
    #         dataFull = dataTemp
    #         imgFull = dataFull*0
    #         for ii in range(nScans):
    #             imgFull[ii, :, :, :] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataFull[ii, :, :, :])))
    #         rawData['dataFull'] = dataFull
    #         rawData['imgFull'] = imgFull
    #
    #         # Average data
    #         data = np.average(dataFull, axis=0)
    #         data = np.reshape(data, (nSL, nPH, nPoints[0]))
    #
    #         # Do zero padding
    #         dataTemp = np.zeros((nPoints[2], nPoints[1], nPoints[0]))
    #         dataTemp = dataTemp+1j*dataTemp
    #         dataTemp[0:nSL, :, :] = data
    #         data = np.reshape(dataTemp, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    #
    #         # Fix the position of the sample according to dfov
    #         kMax = np.array(nPoints)/(2*np.array(fov))*np.array(axesEnable)
    #         kRD = np.linspace(-kMax[0],kMax[0],num=nPoints[0],endpoint=False)
    #     #        kPH = np.linspace(-kMax[1],kMax[1],num=nPoints[1],endpoint=False)
    #         kSL = np.linspace(-kMax[2],kMax[2],num=nPoints[2],endpoint=False)
    #         kPH = kPH[::-1]
    #         kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
    #         kRD = np.reshape(kRD, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    #         kPH = np.reshape(kPH, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    #         kSL = np.reshape(kSL, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    #         dPhase = np.exp(-2*np.pi*1j*(dfov[0]*kRD+dfov[1]*kPH+dfov[2]*kSL))
    #         data = np.reshape(data*dPhase, (nPoints[2], nPoints[1], nPoints[0]))
    #         rawData['kSpace3D'] = data
    #         img=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
    #         rawData['image3D'] = img
    #         data = np.reshape(data, (1, nPoints[0]*nPoints[1]*nPoints[2]))
    #
    #         # Create sampled data
    #         kRD = np.reshape(kRD, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    #         kPH = np.reshape(kPH, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    #         kSL = np.reshape(kSL, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    #         data = np.reshape(data, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    #         rawData['kMax'] = kMax
    #         rawData['sampled'] = np.concatenate((kRD, kPH, kSL, data), axis=1)
    #         data = np.reshape(data, (nPoints[2], nPoints[1], nPoints[0]))
    #
    #         # Save data
    #         mri.saveRawData(rawData)
    #         # Reshape to 0 dimensional
    #         data = np.reshape(data, -1)
    #
    #     return rawData,  msgs, data,  BW
