"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: rare sequence class
"""

import numpy as np
import experiment as ex
import scipy.signal as sig
import configs.hw_config as hw # Import the scanner hardware config
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from plotview.spectrumplot import SpectrumPlot # To plot nice 1d images
from plotview.spectrumplot import Spectrum3DPlot # To show nice 2d or 3d images

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class RARE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RARE, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='RAREInfo', val='RARE')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.066, field='RF')
        self.addParameter(key='rfExFA', string='Exitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=35.0, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=70.0, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=20.0, field='SEQ')
        self.addParameter(key='preExTime', string='Preexitation time (ms)', val=0.0, field='SEQ')
        self.addParameter(key='inversionTime', string='Inversion time (ms)', val=0.0, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=300., field='SEQ')
        self.addParameter(key='fov', string='FOV (cm)', val=[15.0, 15.0, 15.0], field='IM')
        self.addParameter(key='dfov', string='dFOV (mm)', val=[0.0, 0.0, 0.0], field='IM')
        self.addParameter(key='nPoints', string='nPoints (rd, ph, sl)', val=[30, 30, 30], field='IM')
        self.addParameter(key='etl', string='Echo train length', val=5, field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=2.0, field='SEQ')
        self.addParameter(key='axes', string='Axes', val=[0, 1, 2], field='IM')
        self.addParameter(key='axesEnable', string='Axes enable', val=[1, 1, 1], field='IM')
        self.addParameter(key='sweepMode', string='Sweep mode, 0->k20, 1->02k, 2->k2k', val=1, field='SEQ')
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=2.5, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, field='OTH')
        self.addParameter(key='phGradTime', string='Ph gradient time (ms)', val=1.0, field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')
        self.addParameter(key='drfPhase', string='Phase of exciation pulse (º)', val=0.0, field='RF')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=1, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-40, -20, 10], field='OTH')
        self.addParameter(key='parFourierFraction', string='Partial fourier fraction', val=1.0, field='OTH')
        self.addParameter(key='freqCal', string='Calibrate frequency (0 or 1)', val=1, field='OTH')


    def sequenceInfo(self):
        print(" ")
        print("3D RARE sequence")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")


    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        nPoints = np.array(self.mapVals['nPoints'])
        etl = self.mapVals['etl']
        repetitionTime = self.mapVals['repetitionTime']
        parFourierFraction = self.mapVals['parFourierFraction']

        # check if rf amplitude is too high
        rfExFA = self.mapVals['rfExFA'] / 180 * np.pi  # rads
        rfReFA = self.mapVals['rfReFA'] / 180 * np.pi  # rads
        rfExTime = self.mapVals['rfExTime']  # us
        rfReTime = self.mapVals['rfReTime']  # us
        rfExAmp = rfExFA / (rfExTime * hw.b1Efficiency)
        rfReAmp = rfReFA / (rfReTime * hw.b1Efficiency)
        if rfExAmp>1 or rfReAmp>1:
            print("RF amplitude is too high, try with longer RF pulse time.")
            return(0)

        seqTime = nPoints[1]/etl*nPoints[2]*repetitionTime*1e-3*nScans*parFourierFraction/60
        seqTime = np.round(seqTime, decimals=1)
        return(seqTime)  # minutes, scanTime


    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa=False # Starts the gpa
        demo = True

        # Create the inputs automatically. For some reason it only works if there is a few code later...
        # for key in self.mapKeys:
        #     locals()[key] = self.mapVals[key]
        #     if not key in locals():
        #         print('Error')
        #         locals()[key] = self.mapVals[key]

        # Create the inputs manually, pufff
        seqName = self.mapVals['seqName']
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq']# MHz
        rfExFA = self.mapVals['rfExFA']/180*np.pi # rads
        rfReFA = self.mapVals['rfReFA']/180*np.pi # rads
        rfExTime = self.mapVals['rfExTime'] # us
        rfReTime = self.mapVals['rfReTime'] # us
        echoSpacing = self.mapVals['echoSpacing'] # ms
        preExTime = self.mapVals['preExTime'] # ms
        inversionTime = self.mapVals['inversionTime'] # ms
        repetitionTime = self.mapVals['repetitionTime'] # ms
        fov = np.array(self.mapVals['fov']) # cm
        dfov = np.array(self.mapVals['dfov']) # mm
        nPoints = np.array(self.mapVals['nPoints'])
        etl = self.mapVals['etl']
        acqTime = self.mapVals['acqTime'] # ms
        axes = self.mapVals['axes']
        axesEnable = self.mapVals['axesEnable']
        sweepMode = self.mapVals['sweepMode']
        rdGradTime = self.mapVals['rdGradTime'] # ms
        rdDephTime = self.mapVals['rdDephTime'] # ms
        phGradTime = self.mapVals['phGradTime'] # ms
        rdPreemphasis = self.mapVals['rdPreemphasis']
        drfPhase = self.mapVals['drfPhase'] # degrees
        dummyPulses = self.mapVals['dummyPulses']
        shimming = np.array(self.mapVals['shimming']) # *1e4
        parFourierFraction = self.mapVals['parFourierFraction']
        freqCal = self.mapVals['freqCal']

        # Conversion of variables to non-multiplied units
        larmorFreq = larmorFreq*1e6
        rfExTime = rfExTime*1e-6
        rfReTime = rfReTime*1e-6
        fov = fov*1e-2
        dfov = dfov*1e-3
        echoSpacing = echoSpacing*1e-3
        acqTime = acqTime*1e-3
        shimming = shimming*1e-4
        repetitionTime= repetitionTime*1e-3
        preExTime = preExTime*1e-3
        inversionTime = inversionTime*1e-3
        rdGradTime = rdGradTime*1e-3
        rdDephTime = rdDephTime*1e-3
        phGradTime = phGradTime*1e-3

        # Miscellaneous
        larmorFreq = larmorFreq*1e-6    # MHz
        gradRiseTime = 400e-6       # s
        gSteps = int(gradRiseTime*1e6/5)*0+1
        addRdPoints = 10             # Initial rd points to avoid artifact at the begining of rd
        randFactor = 0e-3                        # Random amplitude to add to the phase gradients
        resolution = fov/nPoints
        rfExAmp = rfExFA/(rfExTime*1e6*hw.b1Efficiency)
        rfReAmp = rfReFA/(rfReTime*1e6*hw.b1Efficiency)
        self.mapVals['rfExAmp'] = rfExAmp
        self.mapVals['rfReAmp'] = rfReAmp
        self.mapVals['resolution'] = resolution
        self.mapVals['gradRiseTime'] = gradRiseTime
        self.mapVals['randFactor'] = randFactor
        self.mapVals['addRdPoints'] = addRdPoints

        if rfExAmp>1 or rfReAmp>1:
            print("RF amplitude is too high, try with longer RF pulse time.")
            return(0)

        # Matrix size
        nRD = nPoints[0]+2*addRdPoints
        nPH = nPoints[1]
        nSL = nPoints[2]

        # ETL if etl>nPH
        if etl>nPH:
            etl = nPH

        # parAcqLines in case parAcqLines = 0
        parAcqLines = int(int(nPoints[2]*parFourierFraction)-nPoints[2]/2)
        self.mapVals['partialAcquisition'] = parAcqLines

        # BW
        BW = nPoints[0]/acqTime*1e-6        # MHz
        BWov = BW*hw.oversamplingFactor     # MHz
        samplingPeriod = 1/BWov             # us

        # Readout gradient time
        if rdGradTime<acqTime:
            rdGradTime = acqTime
        self.mapVals['rdGradTime'] = rdGradTime * 1e3 # ms

        # Phase and slice de- and re-phasing time
        if phGradTime==0 or phGradTime>echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime:
            phGradTime = echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime
        self.mapVals['phGradTime'] = phGradTime*1e3 # ms

        # Max gradient amplitude
        rdGradAmplitude = nPoints[0]/(hw.gammaB*fov[0]*acqTime)*axesEnable[0]
        phGradAmplitude = nPH/(2*hw.gammaB*fov[1]*(phGradTime+gradRiseTime))*axesEnable[1]
        slGradAmplitude = nSL/(2*hw.gammaB*fov[2]*(phGradTime+gradRiseTime))*axesEnable[2]
        self.mapVals['rdGradAmplitude'] = rdGradAmplitude
        self.mapVals['phGradAmplitude'] = phGradAmplitude
        self.mapVals['slGradAmplitude'] = slGradAmplitude

        # Readout dephasing amplitude
        rdDephAmplitude = 0.5*rdGradAmplitude*(gradRiseTime+rdGradTime)/(gradRiseTime+rdDephTime)
        self.mapVals['rdDephAmplitude'] = rdDephAmplitude

        # Phase and slice gradient vector
        phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
        slGradients = np.linspace(-slGradAmplitude,slGradAmplitude,num=nSL,endpoint=False)

        # Now fix the number of slices to partailly acquired k-space
        nSL = (int(nPoints[2]/2)+parAcqLines)*axesEnable[2]+(1-axesEnable[2])

        # Add random displacemnt to phase encoding lines
        for ii in range(nPH):
            if ii<np.ceil(nPH/2-nPH/20) or ii>np.ceil(nPH/2+nPH/20):
                phGradients[ii] = phGradients[ii]+randFactor*np.random.randn()
        kPH = hw.gammaB*phGradients*(gradRiseTime+phGradTime)
        self.mapVals['phGradients'] = phGradients
        self.mapVals['slGradients'] = slGradients

        # Set phase vector to given sweep mode
        ind = self.getIndex(etl, nPH, sweepMode)
        self.mapVals['sweepOrder'] = ind
        phGradients = phGradients[ind]

        def createSequenceDemo(phIndex=0, slIndex=0, repeIndexGlobal=0, rewrite=True):
            repeIndex = 0
            acqPoints = 0
            orders = 0
            data = []
            while acqPoints + etl * nRD <= hw.maxRdPoints and orders <= hw.maxOrders and repeIndexGlobal < nRepetitions:
                if repeIndex == 0:
                    acqPoints += nRD
                    data = np.concatenate((data, np.random.randn(nRD * hw.oversamplingFactor)), axis=0)

                for echoIndex in range(etl):
                    if (repeIndex == 0 or repeIndex >= dummyPulses):
                        acqPoints += nRD
                        data = np.concatenate((data, np.random.randn(nRD * hw.oversamplingFactor)), axis=0)

                    # Update the phase and slice gradient
                    if repeIndex >= dummyPulses:
                        if phIndex == nPH - 1:
                            phIndex = 0
                            slIndex += 1
                        else:
                            phIndex += 1
                if repeIndex >= dummyPulses: repeIndexGlobal += 1  # Update the global repeIndex
                repeIndex += 1  # Update the repeIndex after the ETL

            # Return the output variables
            return (phIndex, slIndex, repeIndexGlobal, acqPoints, data)

        def createSequence(phIndex=0, slIndex=0, repeIndexGlobal=0, rewrite=True):
            repeIndex = 0
            if rdGradTime==0:   # Check if readout gradient is dc or pulsed
                dc = True
            else:
                dc = False
            acqPoints = 0
            orders = 0
            # Check in case of dummy pulse fill the cache
            if (dummyPulses>0 and etl*nRD*2>hw.maxRdPoints) or (dummyPulses==0 and etl*nRD>hw.maxRdPoints):
                print('ERROR: Too many acquired points.')
                return()
            # Set shimming
            self.iniSequence(20, shimming, rewrite=rewrite)
            while acqPoints+etl*nRD<=hw.maxRdPoints and orders<=hw.maxOrders and repeIndexGlobal<nRepetitions:
                # Initialize time
                tEx = 20e3+repetitionTime*repeIndex+inversionTime+preExTime

                # First I do a noise measurement.
                if repeIndex==0:
                    t0 = tEx-preExTime-inversionTime-4*acqTime
                    self.rxGate(t0, acqTime+2*addRdPoints/BW)
                    acqPoints += nRD

                # Pre-excitation pulse
                if repeIndex>=dummyPulses and preExTime!=0:
                    t0 = tEx-preExTime-inversionTime-rfExTime/2-hw.blkTime
                    self.rfRecPulse(t0, rfExTime, rfExAmp/90*90, 0)
                    self.gradTrap(t0+hw.blkTime+rfReTime, gradRiseTime, preExTime*0.5, -0.005, gSteps, axes[0], shimming)
                    self.gradTrap(t0+hw.blkTime+rfReTime, gradRiseTime, preExTime*0.5, -0.005, gSteps, axes[1], shimming)
                    self.gradTrap(t0+hw.blkTime+rfReTime, gradRiseTime, preExTime*0.5, -0.005, gSteps, axes[2], shimming)
                    orders = orders+gSteps*6

                # Inversion pulse
                if repeIndex>=dummyPulses and inversionTime!=0:
                    t0 = tEx-inversionTime-rfReTime/2-hw.blkTime
                    self.rfRecPulse(t0, rfReTime, rfReAmp/180*180, 0)
                    self.gradTrap(t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[0], shimming)
                    self.gradTrap(t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[1], shimming)
                    self.gradTrap(t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[2], shimming)
                    orders = orders+gSteps*6

                # DC gradient if desired
                if (repeIndex==0 or repeIndex>=dummyPulses) and dc==True:
                    t0 = tEx-10e3
                    self.gradTrap(t0, gradRiseTime, 10e3+echoSpacing*(etl+1), rdGradAmplitude, gSteps, axes[0], shimming)
                    orders = orders+gSteps*2

                # Excitation pulse
                t0 = tEx-hw.blkTime-rfExTime/2
                self.rfRecPulse(t0,rfExTime,rfExAmp,drfPhase)

                # Dephasing readout
                if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False:
                    t0 = tEx+rfExTime/2-hw.gradDelay
                    self.gradTrap(t0, gradRiseTime, rdDephTime, rdDephAmplitude*rdPreemphasis, gSteps, axes[0], shimming)
                    orders = orders+gSteps*2

                # Echo train
                for echoIndex in range(etl):
                    tEcho = tEx+echoSpacing*(echoIndex+1)

                    # Refocusing pulse
                    t0 = tEcho-echoSpacing/2-rfReTime/2-hw.blkTime
                    self.rfRecPulse(t0, rfReTime, rfReAmp, drfPhase+np.pi/2)

                    # Dephasing phase and slice gradients
                    if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                        t0 = tEcho-echoSpacing/2+rfReTime/2-hw.gradDelay
                        self.gradTrap(t0, gradRiseTime, phGradTime, phGradients[phIndex], gSteps, axes[1], shimming)
                        self.gradTrap(t0, gradRiseTime, phGradTime, slGradients[slIndex], gSteps, axes[2], shimming)
                        orders = orders+gSteps*4

                    # Readout gradient
                    if (repeIndex==0 or repeIndex>=dummyPulses) and dc==False:         # This is to account for dummy pulses
                        t0 = tEcho-rdGradTime/2-gradRiseTime-hw.gradDelay
                        self.gradTrap(t0, gradRiseTime, rdGradTime, rdGradAmplitude, gSteps, axes[0], shimming)
                        orders = orders+gSteps*2

                    # Rx gate
                    if (repeIndex==0 or repeIndex>=dummyPulses):
                        t0 = tEcho-acqTime/2-addRdPoints/BW
                        self.rxGate(t0, acqTime+2*addRdPoints/BW)
                        acqPoints += nRD

                    # Rephasing phase and slice gradients
                    t0 = tEcho+acqTime/2+addRdPoints/BW-hw.gradDelay
                    if (echoIndex<etl-1 and repeIndex>=dummyPulses):
                        self.gradTrap(t0, gradRiseTime, phGradTime, -phGradients[phIndex], gSteps, axes[1], shimming)
                        self.gradTrap(t0, gradRiseTime, phGradTime, -slGradients[slIndex], gSteps, axes[2], shimming)
                        orders = orders+gSteps*4
                    elif(echoIndex==etl-1 and repeIndex>=dummyPulses):
                        self.gradTrap(t0, gradRiseTime, phGradTime, +phGradients[phIndex], gSteps, axes[1], shimming)
                        self.gradTrap(t0, gradRiseTime, phGradTime, +slGradients[slIndex], gSteps, axes[2], shimming)
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
            self.endSequence(repeIndex*repetitionTime)

            # Return the output variables
            return(phIndex, slIndex, repeIndexGlobal, acqPoints)


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
        nRepetitions = int(nSL*nPH/etl)
        scanTime = nRepetitions*repetitionTime
        self.mapVals['scanTime'] = scanTime*nSL*1e-6

        # Calibrate frequency
        if freqCal and (not plotSeq) and (not demo):
            larmorFreq = self.freqCalibration(bw=0.05)
            larmorFreq = self.freqCalibration(bw=0.005)
            drfPhase = self.mapVals['drfPhase']

        # Create full sequence
        # Run the experiment
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
                acqTime = nPoints[0]/BW        # us
                self.mapVals['bw'] = BW
                phIndex, slIndex, repeIndexGlobal, aa = createSequence(phIndex=phIndex,
                                                                   slIndex=slIndex,
                                                                   repeIndexGlobal=repeIndexGlobal,
                                                                   rewrite=nBatches==1)
                repeIndexArray = np.concatenate((repeIndexArray, np.array([repeIndexGlobal-1])), axis=0)
                acqPointsPerBatch.append(aa)
            else:
                phIndex, slIndex, repeIndexGlobal, aa, dataA = createSequenceDemo(phIndex=phIndex,
                                                                   slIndex=slIndex,
                                                                   repeIndexGlobal=repeIndexGlobal,
                                                                   rewrite=nBatches==1)
                repeIndexArray = np.concatenate((repeIndexArray, np.array([repeIndexGlobal-1])), axis=0)
                acqPointsPerBatch.append(aa)
                self.mapVals['bw'] = 1/samplingPeriod/hw.oversamplingFactor

            for ii in range(nScans):
                if not demo:
                    if plotSeq==1:                  # What is the meaning of plotSeq??
                        self.expt.__del__()
                        break
                    elif plotSeq==0:
                        print('Batch ', nBatches, ', Scan ', ii, ' runing...')
                        rxd, msgs = self.expt.run()
                        rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
                        # Get noise data
                        noise = np.concatenate((noise, rxd['rx0'][0:nRD*hw.oversamplingFactor]), axis = 0)
                        rxd['rx0'] = rxd['rx0'][nRD*hw.oversamplingFactor::]
                        # Get data
                        if dummyPulses>0:
                            dummyData = np.concatenate((dummyData, rxd['rx0'][0:nRD*etl*hw.oversamplingFactor]), axis = 0)
                            overData = np.concatenate((overData, rxd['rx0'][nRD*etl*hw.oversamplingFactor::]), axis = 0)
                        else:
                            overData = np.concatenate((overData, rxd['rx0']), axis = 0)
                else:
                    print('Batch ', nBatches, ', Scan ', ii, ' runing...')
                    data = dataA
                    noise = np.concatenate((noise, data[0:nRD*hw.oversamplingFactor]), axis = 0)
                    data = data[nRD*hw.oversamplingFactor::]
                    # Get data
                    if dummyPulses>0:
                        dummyData = np.concatenate((dummyData, data[0:nRD*etl*hw.oversamplingFactor]), axis = 0)
                        overData = np.concatenate((overData, data[nRD*etl*hw.oversamplingFactor::]), axis = 0)
                    else:
                        overData = np.concatenate((overData, data), axis = 0)

            if not demo: self.expt.__del__()
            if plotSeq ==1:
                break
        del aa

        if plotSeq ==0:
            acqPointsPerBatch= (np.array(acqPointsPerBatch)-etl*nRD*(dummyPulses>0)-nRD)*nScans
            print('Scans done!')
            self.mapVals['noiseData'] = noise
            self.mapVals['overData'] = overData

            # Fix the echo position using oversampled data
            if dummyPulses>0:
                dummyData = np.reshape(dummyData,  (nBatches*nScans, etl, nRD*hw.oversamplingFactor))
                dummyData = np.average(dummyData, axis=0)
                self.mapVals['dummyData'] = dummyData
                overData = np.reshape(overData, (-1, etl, nRD*hw.oversamplingFactor))
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
            # Reorganize the data acording to sweep mode
            dataProv = np.reshape(dataProv, (nSL, nPH, nRD))
            dataTemp = dataProv*0
            for ii in range(nPH):
                dataTemp[:, ind[ii], :] = dataProv[:,  ii, :]
            dataProv = dataTemp
            # Check where is krd = 0
            dataProv = dataProv[int(nPoints[2]/2), int(nPH/2), :]
            indkrd0 = np.argmax(np.abs(dataProv))
            if  indkrd0 < nRD/2-addRdPoints or indkrd0 > nRD/2+addRdPoints:
                indkrd0 = int(nRD/2)

            # Get individual images
            dataFull = np.reshape(dataFull, (nScans, nSL, nPH, nRD))
            dataFull = dataFull[:, :, :, indkrd0-int(nPoints[0]/2):indkrd0+int(nPoints[0]/2)]
            dataTemp = dataFull*0
            for ii in range(nPH):
                dataTemp[:, :, ind[ii], :] = dataFull[:, :,  ii, :]
            dataFull = dataTemp
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
            self.mapVals['sampledCartesian'] = self.mapVals['sampled']  # To sweep
            data = np.reshape(data, (nPoints[2], nPoints[1], nPoints[0]))


    def sequenceAnalysis(self, obj=''):
        self.saveRawData()
        nPoints = self.mapVals['nPoints']
        axesEnable = self.mapVals['axesEnable']

        # Get axes in strings
        axes = self.mapVals['axes']
        axesDict = {'x':0, 'y':1, 'z':2}
        axesKeys = list(axesDict.keys())
        axesVals = list(axesDict.values())
        axesStr = ['','','']
        n = 0
        for val in axes:
            index = axesVals.index(val)
            axesStr[n] = axesKeys[index]
            n += 1

        if (axesEnable[1] == 0 and axesEnable[2] == 0):
            bw = self.mapVals['bw']*1e-3 # kHz
            acqTime = self.mapVals['acqTime'] # ms
            tVector = np.linspace(-acqTime/2, acqTime/2, nPoints[0])
            sVector = self.mapVals['sampled'][:, 3]
            fVector = np.linspace(-bw/2, bw/2, nPoints[0])
            iVector = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(sVector)))

            # Plots to show into the GUI
            f_plotview = SpectrumPlot(fVector, [np.abs(iVector)], ['Spectrum magnitude'],
                                      "Frequency (kHz)", "Amplitude (a.u.)",
                                      "%s Spectrum" % self.mapVals['fileName'])
            t_plotview = SpectrumPlot(tVector, [np.abs(sVector), np.real(sVector), np.imag(sVector)],
                                      ['Magnitude', 'Real', 'Imaginary'],
                                      'Time (ms)', "Signal amplitude (mV)",
                                      "%s Signal" % self.mapVals['fileName'])
            self.out = [t_plotview, f_plotview]
            return(self.out)
        else:
            # Plot image
            # image = pg.image(np.abs(self.mapVals['image3D']))
            image = np.abs(self.mapVals['image3D'])
            image = image/np.max(np.reshape(image,-1))*100
            image = Spectrum3DPlot(image,
                                   title='Image magnitude',
                                   xLabel=axesStr[1]+" Axis",
                                   yLabel=axesStr[0]+" Axis")
            imageWidget = image.getImageWidget()

            kSpace = Spectrum3DPlot(np.abs(self.mapVals['kSpace3D']),
                                    title='k-Space',
                                    xLabel="k%s"%axesStr[1],
                                    yLabel="k%s"%axesStr[0])
            kSpaceWidget = kSpace.getImageWidget()

            self.out = [imageWidget, kSpaceWidget]
            return(self.out)