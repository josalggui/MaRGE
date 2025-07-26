"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: rare sequence class
"""

import os
import sys
#*****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory = os.path.dirname(parent_directory)

# Define the subdirectories you want to add to sys.path
subdirs = ['MaRGE', 'marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
#******************************************************************************
import numpy as np
import marge.marcos.marcos_client.experiment
import scipy.signal as sig
from scipy.stats import linregress
import marge.configs.hw_config as hw # Import the scanner hardware config
import marge.configs.units as units
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.

from datetime import date
from datetime import datetime
import ismrmrd
import ismrmrd.xsd
import datetime
import ctypes
import matplotlib
import xml.etree.ElementTree as ET
from scipy.io import loadmat

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class RARE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RARE, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='RAREInfo', val='RARE')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM') ## number of scans 
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, units=units.kHz, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=35.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=70.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=20.0, units=units.ms, field='SEQ')
        self.addParameter(key='preExTime', string='Preexitation time (ms)', val=0.0, units=units.ms, field='SEQ')
        self.addParameter(key='inversionTime', string='Inversion time (ms)', val=0.0, units=units.ms, field='SEQ', tip="0 to ommit this pulse")
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=300., units=units.ms, field='SEQ', tip="0 to ommit this pulse")
        self.addParameter(key='fov', string='FOV[x,y,z] (cm)', val=[15.0, 15.0, 15.0], units=units.cm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM', tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[40, 40, 10], field='IM')
        self.addParameter(key='angle', string='Angle (º)', val=0.0, field='IM')
        self.addParameter(key='rotationAxis', string='Rotation axis', val=[0, 0, 1], field='IM')
        self.addParameter(key='etl', string='Echo train length', val=5, field='SEQ') ## nm of peaks in 1 repetition
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=2.0, units=units.ms, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[0, 1, 2], field='IM', tip="0=x, 1=y, 2=z")
        self.addParameter(key='axesEnable', string='Axes enable', val=[1, 1, 0], tip="Use 0 for directions with matrix size 1, use 1 otherwise.")
        self.addParameter(key='sweepMode', string='Sweep mode', val=1, field='SEQ', tip="0: sweep from -kmax to kmax. 1: sweep from 0 to kmax. 2: sweep from kmax to 0")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=2.5, units=units.ms, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='phGradTime', string='Ph gradient time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')
        self.addParameter(key='rfPhase', string='RF phase (º)', val=0.0, field='OTH')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=1, field='SEQ', tip="Use last dummy pulse to calibrate k = 0")
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='parFourierFraction', string='Partial fourier fraction', val=1.0, field='OTH', tip="Fraction of k planes aquired in slice direction")
        self.addParameter(key='echo_shift', string='Echo time shift', val=0.0, units=units.us, field='OTH', tip='Shift the gradient echo time respect to the spin echo time.')
        self.addParameter(key='unlock_orientation', string='Unlock image orientation', val=0, field='OTH', tip='0: Images oriented according to standard. 1: Image raw orientation')
        self.acq = ismrmrd.Acquisition()
        self.img = ismrmrd.Image()
        self.header= ismrmrd.xsd.ismrmrdHeader() 
        
       
    def sequenceInfo(self):
        print("3D RARE sequence")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain \n")

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
            print("ERROR: RF amplitude is too high, try with longer RF pulse time.")
            return(0)

        seqTime = nPoints[1]/etl*nPoints[2]*repetitionTime*1e-3*nScans*parFourierFraction/60
        seqTime = np.round(seqTime, decimals=1)
        return(seqTime)  # minutes, scanTime

        # TODO: check for min and max values for all fields

    def sequenceRun(self, plotSeq=False, demo=False):
        init_gpa=False # Starts the gpa
        self.demo = demo

        # Set the fov
        self.dfov = self.getFovDisplacement()
        self.dfov = self.dfov[self.axesOrientation]
        self.fov = self.fov[self.axesOrientation]

        # Check for used axes
        axesEnable = []
        for ii in range(3):
            if self.nPoints[ii] == 1:
                axesEnable.append(0)
            else:
                axesEnable.append(1)
        self.mapVals['axesEnable'] = axesEnable
        self.axesEnable = axesEnable

        # Miscellaneous
        self.freqOffset = self.freqOffset*1e6 # MHz
        gradRiseTime = hw.grad_rise_time
        gSteps = hw.grad_steps
        addRdPoints = 10 # Initial rd points to avoid artifact at the begining of rd
        randFactor = 0e-3                        # Random amplitude to add to the phase gradients
        resolution = self.fov/self.nPoints
        rfExAmp = self.rfExFA/(self.rfExTime*1e6*hw.b1Efficiency)*np.pi/180
        rfReAmp = self.rfReFA/(self.rfReTime*1e6*hw.b1Efficiency)*np.pi/180
        self.mapVals['rfExAmp'] = rfExAmp
        self.mapVals['rfReAmp'] = rfReAmp
        self.mapVals['resolution'] = resolution
        self.mapVals['gradRiseTime'] = gradRiseTime
        self.mapVals['randFactor'] = randFactor
        self.mapVals['addRdPoints'] = addRdPoints
        self.mapVals['larmorFreq'] = hw.larmorFreq + self.freqOffset
        self.addRdPoints = addRdPoints
        if rfExAmp>1 or rfReAmp>1:
            print("ERROR: RF amplitude is too high, try with longer RF pulse time.")
            return(0)

        # Matrix size
        nRD = self.nPoints[0]+2*addRdPoints
        nPH = self.nPoints[1]
        nSL = self.nPoints[2]

        # ETL if etl>nPH
        if self.etl>nPH:
            self.etl = nPH

        # parAcqLines in case parAcqLines = 0
        parAcqLines = int(int(self.nPoints[2]*self.parFourierFraction)-self.nPoints[2]/2)
        self.mapVals['partialAcquisition'] = parAcqLines

        # BW
        BW = self.nPoints[0]/self.acqTime*1e-6        # MHz
        BWov = BW*hw.oversamplingFactor     # MHz
        samplingPeriod = 1/BWov             # us

        # Readout gradient time
        if self.rdGradTime<self.acqTime:
            self.rdGradTime = self.acqTime
        self.mapVals['rdGradTime'] = self.rdGradTime * 1e3 # ms

        # Phase and slice de- and re-phasing time
        if self.phGradTime==0 or self.phGradTime>self.echoSpacing/2-self.rfExTime/2-self.rfReTime/2-2*gradRiseTime:
            self.phGradTime = self.echoSpacing/2-self.rfExTime/2-self.rfReTime/2-2*gradRiseTime
        self.mapVals['phGradTime'] = self.phGradTime*1e3 # ms

        # Max gradient amplitude
        rdGradAmplitude = self.nPoints[0]/(hw.gammaB*self.fov[0]*self.acqTime)*axesEnable[0]
        phGradAmplitude = nPH/(2*hw.gammaB*self.fov[1]*(self.phGradTime+gradRiseTime))*axesEnable[1]
        slGradAmplitude = nSL/(2*hw.gammaB*self.fov[2]*(self.phGradTime+gradRiseTime))*axesEnable[2]
        self.mapVals['rdGradAmplitude'] = rdGradAmplitude
        self.mapVals['phGradAmplitude'] = phGradAmplitude
        self.mapVals['slGradAmplitude'] = slGradAmplitude

        # Readout dephasing amplitude
        rdDephAmplitude = 0.5*rdGradAmplitude*(gradRiseTime+self.rdGradTime)/(gradRiseTime+self.rdDephTime)
        self.mapVals['rdDephAmplitude'] = rdDephAmplitude

        # Phase and slice gradient vector
        phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
        slGradients = np.linspace(-slGradAmplitude,slGradAmplitude,num=nSL,endpoint=False)

        # Now fix the number of slices to partailly acquired k-space
        nSL = (int(self.nPoints[2]/2)+parAcqLines)*axesEnable[2]+(1-axesEnable[2])

        # Add random displacemnt to phase encoding lines
        for ii in range(nPH):
            if ii<np.ceil(nPH/2-nPH/20) or ii>np.ceil(nPH/2+nPH/20):
                phGradients[ii] = phGradients[ii]+randFactor*np.random.randn()
        kPH = hw.gammaB*phGradients*(gradRiseTime+self.phGradTime)
        self.mapVals['phGradients'] = phGradients
        self.mapVals['slGradients'] = slGradients

        # Set phase vector to given sweep mode
        ind = self.getIndex(self.etl, nPH, self.sweepMode)
        self.mapVals['sweepOrder'] = ind
        phGradients = phGradients[ind]

        # Get the rotation matrix
        rot = self.getRotationMatrix()
        gradAmp = np.array([0.0, 0.0, 0.0])
        gradAmp[self.axesOrientation[0]] = 1
        gradAmp = np.reshape(gradAmp, (3, 1))
        result = np.dot(rot, gradAmp)

        print("Readout direction:")
        print(np.reshape(result, (1, 3)))

        # Initialize k-vectors
        k_ph_sl_xyz = np.ones((3, self.nPoints[0]*self.nPoints[1]*nSL))*hw.gammaB*(self.phGradTime+hw.grad_rise_time)
        k_rd_xyz = np.ones((3, self.nPoints[0]*self.nPoints[1]*nSL))*hw.gammaB

        def createSequence(phIndex=0, slIndex=0, lnIndex=0, repeIndexGlobal=0):
            repeIndex = 0
            acqPoints = 0
            orders = 0

            # Check in case of dummy pulse fill the cache
            if (self.dummyPulses>0 and self.etl*nRD*2>hw.maxRdPoints) or (self.dummyPulses==0 and self.etl*nRD>hw.maxRdPoints):
                print('ERROR: Too many acquired points.')
                return 0

            # Set shimming
            self.iniSequence(20, self.shimming)
            while acqPoints+self.etl*nRD<=hw.maxRdPoints and orders<=hw.maxOrders and repeIndexGlobal<nRepetitions:
                # Initialize time
                tEx = self.repetitionTime+self.repetitionTime*repeIndex+self.inversionTime+self.preExTime

                # First I do a noise measurement.
                if repeIndex==0:
                    t0 = 40
                    self.rxGate(t0, self.acqTime+2*addRdPoints/BW)
                    acqPoints += nRD

                # Pre-excitation pulse
                if repeIndex>=self.dummyPulses and self.preExTime!=0:
                    t0 = tEx-self.preExTime-self.inversionTime-self.rfExTime/2-hw.blkTime
                    self.rfRecPulse(t0, self.rfExTime, rfExAmp, 0)
                    self.gradTrap(t0 + hw.blkTime + self.rfReTime, gradRiseTime, self.preExTime * 0.5, -0.005, gSteps,
                                  self.axesOrientation[0], self.shimming)
                    self.gradTrap(t0 + hw.blkTime + self.rfReTime, gradRiseTime, self.preExTime * 0.5, -0.005, gSteps,
                                  self.axesOrientation[1], self.shimming)
                    self.gradTrap(t0 + hw.blkTime + self.rfReTime, gradRiseTime, self.preExTime * 0.5, -0.005, gSteps,
                                  self.axesOrientation[2], self.shimming)
                orders = orders+gSteps*6

                # Inversion pulse
                if repeIndex>=self.dummyPulses and self.inversionTime!=0:
                    t0 = tEx-self.inversionTime-self.rfReTime/2-hw.blkTime
                    self.rfRecPulse(t0, self.rfReTime, rfReAmp, 0)
                    self.gradTrap(t0 + hw.blkTime + self.rfReTime, gradRiseTime, self.inversionTime * 0.5, 0.005,
                                  gSteps, self.axesOrientation[0], self.shimming)
                    self.gradTrap(t0 + hw.blkTime + self.rfReTime, gradRiseTime, self.inversionTime * 0.5, 0.005,
                                  gSteps, self.axesOrientation[1], self.shimming)
                    self.gradTrap(t0 + hw.blkTime + self.rfReTime, gradRiseTime, self.inversionTime * 0.5, 0.005,
                                  gSteps, self.axesOrientation[2], self.shimming)
                orders = orders+gSteps*6

                # Excitation pulse
                t0 = tEx-hw.blkTime-self.rfExTime/2
                self.rfRecPulse(t0,self.rfExTime,rfExAmp,0)

                # Dephasing readout
                gradAmp = np.array([0.0, 0.0, 0.0])
                gradAmp[self.axesOrientation[0]] = rdDephAmplitude
                gradAmp = np.dot(rot, np.reshape(gradAmp, (3, 1)))
                if repeIndex==(self.dummyPulses-1) or repeIndex>=self.dummyPulses:
                    t0 = tEx+self.rfExTime/2-hw.gradDelay
                    self.gradTrap(t0, gradRiseTime, self.rdDephTime, gradAmp[0] * self.rdPreemphasis, gSteps,
                                  0, self.shimming)
                    self.gradTrap(t0, gradRiseTime, self.rdDephTime, gradAmp[1] * self.rdPreemphasis, gSteps,
                                  1, self.shimming)
                    self.gradTrap(t0, gradRiseTime, self.rdDephTime, gradAmp[2] * self.rdPreemphasis, gSteps,
                                  2, self.shimming)
                    orders = orders+gSteps*6

                # Echo train
                for echoIndex in range(self.etl):
                    tEcho = tEx+self.echoSpacing*(echoIndex+1)

                    # Refocusing pulse
                    t0 = tEcho-self.echoSpacing/2-self.rfReTime/2-hw.blkTime
                    self.rfRecPulse(t0, self.rfReTime, rfReAmp, np.pi/2+self.rfPhase*np.pi/180)

                    # Dephasing phase and slice gradients
                    gradAmp = np.array([0.0, 0.0, 0.0])
                    gradAmp[self.axesOrientation[1]] = phGradients[phIndex]
                    gradAmp[self.axesOrientation[2]] = slGradients[slIndex]
                    gradAmp = np.dot(rot, np.reshape(gradAmp, (3, 1)))
                    if repeIndex>=self.dummyPulses:         # This is to account for dummy pulses
                        t0 = tEcho-self.echoSpacing/2+self.rfReTime/2-hw.gradDelay
                        self.gradTrap(t0, gradRiseTime, self.phGradTime, gradAmp[0], gSteps, 0, self.shimming)
                        self.gradTrap(t0, gradRiseTime, self.phGradTime, gradAmp[1], gSteps, 1, self.shimming)
                        self.gradTrap(t0, gradRiseTime, self.phGradTime, gradAmp[2], gSteps, 2, self.shimming)
                        orders = orders+gSteps*6
                        # get k-point
                        k_ph_sl_xyz[:, self.nPoints[0]*lnIndex:self.nPoints[0]*(lnIndex+1)] = \
                            np.diag(np.reshape(gradAmp, -1)) @ \
                            k_ph_sl_xyz[:, self.nPoints[0] * lnIndex:self.nPoints[0] * (lnIndex + 1)]

                    # Readout gradient
                    gradAmp = np.array([0.0, 0.0, 0.0])
                    gradAmp[self.axesOrientation[0]] = rdGradAmplitude
                    gradAmp = np.dot(rot, np.reshape(gradAmp, (3, 1)))
                    if repeIndex==(self.dummyPulses-1) or repeIndex>=self.dummyPulses:         # This is to account for dummy pulses
                        t0 = tEcho-self.rdGradTime/2-gradRiseTime-hw.gradDelay+self.echo_shift
                        self.gradTrap(t0, gradRiseTime, self.rdGradTime, gradAmp[0], gSteps, 0, self.shimming)
                        self.gradTrap(t0, gradRiseTime, self.rdGradTime, gradAmp[1], gSteps, 1, self.shimming)
                        self.gradTrap(t0, gradRiseTime, self.rdGradTime, gradAmp[2], gSteps, 2, self.shimming)
                        orders = orders+gSteps*6

                    # Rx gate
                    if repeIndex==(self.dummyPulses-1) or repeIndex>=self.dummyPulses:
                        t0 = tEcho-self.acqTime/2-addRdPoints/BW+self.echo_shift
                        self.rxGate(t0, self.acqTime+2*addRdPoints/BW)
                        acqPoints += nRD

                    if repeIndex>=self.dummyPulses:
                        k_rd_xyz[:, self.nPoints[0] * lnIndex:self.nPoints[0] * (lnIndex + 1)] = \
                            np.diag(np.reshape(gradAmp, -1)) @ \
                            k_rd_xyz[:, self.nPoints[0] * lnIndex:self.nPoints[0] * (lnIndex + 1)]  @ \
                            np.diag(self.time_vector)

                    # Rephasing phase and slice gradients
                    gradAmp = np.array([0.0, 0.0, 0.0])
                    gradAmp[self.axesOrientation[1]] = phGradients[phIndex]
                    gradAmp[self.axesOrientation[2]] = slGradients[slIndex]
                    gradAmp = np.dot(rot, np.reshape(gradAmp, (3, 1)))
                    t0 = tEcho+self.rdGradTime/2+gradRiseTime-hw.gradDelay+self.echo_shift
                    if (echoIndex<self.etl-1 and repeIndex>=self.dummyPulses):
                        self.gradTrap(t0, gradRiseTime, self.phGradTime, -gradAmp[0], gSteps, 0, self.shimming)
                        self.gradTrap(t0, gradRiseTime, self.phGradTime, -gradAmp[1], gSteps, 1, self.shimming)
                        self.gradTrap(t0, gradRiseTime, self.phGradTime, -gradAmp[2], gSteps, 2, self.shimming)
                        orders = orders+gSteps*6
                    elif(echoIndex==self.etl-1 and repeIndex>=self.dummyPulses):
                        self.gradTrap(t0, gradRiseTime, self.phGradTime, gradAmp[0], gSteps, 0, self.shimming)
                        self.gradTrap(t0, gradRiseTime, self.phGradTime, gradAmp[1], gSteps, 1, self.shimming)
                        self.gradTrap(t0, gradRiseTime, self.phGradTime, gradAmp[2], gSteps, 2, self.shimming)
                        orders = orders+gSteps*6

                    # Update the phase and slice gradient
                    if repeIndex>=self.dummyPulses:
                        lnIndex +=1
                        if phIndex == nPH-1:
                            phIndex = 0
                            slIndex += 1
                        else:
                            phIndex += 1
                if repeIndex>=self.dummyPulses: 
                    repeIndexGlobal += 1 # Update the global repeIndex
                repeIndex+=1 # Update the repeIndex after the ETL

            # Turn off the gradients after the end of the batch
            self.endSequence((repeIndex+1)*self.repetitionTime)

            # Return the output variables
            return(phIndex, slIndex, lnIndex, repeIndexGlobal, acqPoints)

        # Changing time parameters to us
        self.rfExTime = self.rfExTime*1e6
        self.rfReTime = self.rfReTime*1e6
        self.echoSpacing = self.echoSpacing*1e6
        self.repetitionTime = self.repetitionTime*1e6
        gradRiseTime = gradRiseTime*1e6
        self.phGradTime = self.phGradTime*1e6
        self.rdGradTime = self.rdGradTime*1e6
        self.rdDephTime = self.rdDephTime*1e6
        self.inversionTime = self.inversionTime*1e6
        self.preExTime = self.preExTime*1e6
        self.echo_shift = self.echo_shift*1e6
        nRepetitions = int(nSL*nPH/self.etl)
        scanTime = nRepetitions*self.repetitionTime
        self.mapVals['Scan Time (s)'] = scanTime*self.nScans*1e-6

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
        lnIndex = 0
        acqPointsPerBatch = []
        while repeIndexGlobal<nRepetitions:
            nBatches += 1
            # Create the experiment if it is not a demo
            if not self.demo:
                self.expt = ex.Experiment(lo_freq=hw.larmorFreq+self.freqOffset, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
                samplingPeriod = self.expt.get_rx_ts()[0]
                BW = 1/samplingPeriod/hw.oversamplingFactor

            # Time vector for main points
            self.time_vector = np.linspace(-self.nPoints[0]/BW/2 + 0.5/BW, self.nPoints[0]/BW/2 - 0.5/BW,
                                           self.nPoints[0]) * 1e-6 # s
            
            # Run the createSequence method
            self.acqTime = self.nPoints[0]/BW        # us
            self.mapVals['bw'] = BW
            phIndex, slIndex, lnIndex, repeIndexGlobal, aa = createSequence(phIndex=phIndex,
                                                                            slIndex=slIndex,
                                                                            lnIndex=lnIndex,
                                                                            repeIndexGlobal=repeIndexGlobal)

            # Save instructions into MaRCoS if not a demo
            if self.floDict2Exp(rewrite=nBatches==1, demo=self.demo):
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("ERROR: sequence waveforms out of hardware bounds")
                return False

            repeIndexArray = np.concatenate((repeIndexArray, np.array([repeIndexGlobal-1])), axis=0)
            acqPointsPerBatch.append(aa)

            if not plotSeq:
                for ii in range(self.nScans):
                    print("Batch %i, scan %i running..." % (nBatches, ii+1))
                    if not self.demo:
                        acq_points = 0
                        while acq_points != (aa * hw.oversamplingFactor):
                            rxd, msgs = self.expt.run()
                            rxd['rx0'] = rxd['rx0']*hw.adcFactor   # Here I normalize to get the result in mV
                            acq_points = np.size(rxd['rx0'])
                            print("Acquired points = %i" % acq_points)
                            print("Expected points = %i" % (aa * hw.oversamplingFactor))
                            if acq_points != (aa * hw.oversamplingFactor):
                                print("WARNING: missed points. Repeating batch...")
                        print("Batch %i, scan %i ready!" % (nBatches, ii+1))
                    else:
                        rxd = {}
                        rxd['rx0'] = np.random.randn(aa*hw.oversamplingFactor) + 1j * np.random.randn(aa*hw.oversamplingFactor)
                        print("Batch %i, scan %i ready!" % (nBatches, ii+1))
                    # Get noise data
                    noise = np.concatenate((noise, rxd['rx0'][0:nRD*hw.oversamplingFactor]), axis = 0)
                    rxd['rx0'] = rxd['rx0'][nRD*hw.oversamplingFactor::]
                    # Get data
                    if self.dummyPulses>0:
                        dummyData = np.concatenate((dummyData, rxd['rx0'][0:nRD*self.etl*hw.oversamplingFactor]), axis = 0)
                        overData = np.concatenate((overData, rxd['rx0'][nRD*self.etl*hw.oversamplingFactor::]), axis = 0)
                    else:
                        overData = np.concatenate((overData, rxd['rx0']), axis = 0)
            # elif plotSeq and standalone:
            #     self.plotSequence()

            if not self.demo: self.expt.__del__() 
        del aa

        if not plotSeq:
            acqPointsPerBatch= (np.array(acqPointsPerBatch)-self.etl*nRD*(self.dummyPulses>0)-nRD)*self.nScans
            print('Scans ready!')
            self.mapVals['noiseData'] = noise
            self.mapVals['overData'] = overData

            # Fix the echo position using oversampled data
            if self.dummyPulses>0:
                dummyData = np.reshape(dummyData,  (nBatches*self.nScans, self.etl, nRD*hw.oversamplingFactor))
                dummyData = np.average(dummyData, axis=0)
                self.mapVals['dummyData'] = dummyData
                overData = np.reshape(overData, (-1, self.etl, nRD*hw.oversamplingFactor))
                #overData = self.fixEchoPosition(dummyData, overData)
                overData = np.reshape(overData, -1)
                if self.etl > 1:
                    self.dummyAnalysis()

            # Generate dataFull
            dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
        
            if nBatches>1:
                dataFullA = dataFull[0:sum(acqPointsPerBatch[0:-1])]
                dataFullB = dataFull[sum(acqPointsPerBatch[0:-1])::]
                
            
            # Reorganize dataFull
            dataProv = np.zeros([self.nScans,nSL*nPH*nRD])
            dataProv = dataProv+1j*dataProv
            if nBatches>1:
                dataFullA = np.reshape(dataFullA, (nBatches-1, self.nScans, -1, nRD))
                dataFullB = np.reshape(dataFullB, (1, self.nScans, -1, nRD))
            else:
                dataFull = np.reshape(dataFull, (nBatches, self.nScans, -1, nRD))
            for scan in range(self.nScans):
                if nBatches>1:
                    dataProv[scan, :] = np.concatenate((np.reshape(dataFullA[:,scan,:,:],-1), np.reshape(dataFullB[:,scan,:,:],-1)), axis=0)
                else:
                    dataProv[scan, :] = np.reshape(dataFull[:,scan,:,:],-1)
            dataFull = np.reshape(dataProv,-1)
            
            # Save dataFull to save it in .h5 ########################################################""
            self.dataFullmat = dataFull

            # Get index for krd = 0
            # Average data
            dataProv = np.reshape(dataFull, (self.nScans, nRD*nPH*nSL))
            dataProv = np.average(dataProv, axis=0)
            # Reorganize the data acording to sweep mode
            dataProv = np.reshape(dataProv, (nSL, nPH, nRD))
            
            dataTemp = dataProv*0
            for ii in range(nPH):
                dataTemp[:, ind[ii], :] = dataProv[:,  ii, :]
            dataProv = dataTemp
           

            dataProv = dataProv[int(self.nPoints[2]/2), int(nPH/2), :] 
            indkrd0 = np.argmax(np.abs(dataProv))
            if indkrd0 < nRD/2-addRdPoints or indkrd0 > nRD/2+addRdPoints:
                indkrd0 = int(nRD/2)

            # Get individual images
            dataFull = np.reshape(dataFull, (self.nScans, nSL, nPH, nRD))
            dataFull = dataFull[:, :, :, indkrd0-int(self.nPoints[0]/2):indkrd0+int(self.nPoints[0]/2)] 
            dataTemp = dataFull*0
            for ii in range(nPH):
                dataTemp[:, :, ind[ii], :] = dataFull[:, :,  ii, :]
            dataFull = dataTemp
            imgFull = dataFull*0
            for ii in range(self.nScans):
                imgFull[ii, :, :, :] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataFull[ii, :, :, :])))
            self.mapVals['dataFull'] = dataFull 
            self.mapVals['imgFull'] = imgFull 

            # Average data
            data = np.average(dataFull, axis=0)

            # Concatenate with k_xyz
            for ii in range(3):
                k_prov = np.reshape(k_ph_sl_xyz[ii, :], (nSL, nPH, self.nPoints[0]))
                k_temp = k_prov * 0
                for jj in range(nPH):
                    k_temp[:, ind[jj], :] = k_prov[:, jj, :]
                k_ph_sl_xyz[ii, :] = np.reshape(k_temp, -1)
            k_xyz = k_ph_sl_xyz + k_rd_xyz
            self.mapVals['sampled_xyz'] = np.concatenate((k_xyz.T, np.reshape(data, (nSL*nPH*self.nPoints[0], 1))), axis=1)
            data = np.reshape(data, (nSL, nPH, self.nPoints[0]))

            # Do zero padding
            dataTemp = np.zeros((self.nPoints[2], self.nPoints[1], self.nPoints[0]), dtype=complex)
            dataTemp[0:nSL, :, :] = data
            data = np.reshape(dataTemp, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))

            # if self.demo:
            #     data = self.myPhantom()

            # Fix the position of the sample according to dfov
            kMax = np.array(self.nPoints)/(2*np.array(self.fov))*np.array(axesEnable)
            kRD = self.time_vector*hw.gammaB*rdGradAmplitude
            kSL = np.linspace(-kMax[2],kMax[2],num=self.nPoints[2],endpoint=False)
            kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
            kRD = np.reshape(kRD, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))
            kPH = np.reshape(kPH, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))
            kSL = np.reshape(kSL, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))
            dPhase = np.exp(-2*np.pi*1j*(self.dfov[0]*kRD+self.dfov[1]*kPH+self.dfov[2]*kSL))
            data = np.reshape(data*dPhase, (self.nPoints[2], self.nPoints[1], self.nPoints[0]))
            self.mapVals['kSpace3D'] = data
            img=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
            self.mapVals['image3D'] = img
            data = np.reshape(data, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))

            # Create sampled data
            kRD = np.reshape(kRD, (self.nPoints[0]*self.nPoints[1]*self.nPoints[2], 1))
            kPH = np.reshape(kPH, (self.nPoints[0]*self.nPoints[1]*self.nPoints[2], 1))
            kSL = np.reshape(kSL, (self.nPoints[0]*self.nPoints[1]*self.nPoints[2], 1))
            data = np.reshape(data, (self.nPoints[0]*self.nPoints[1]*self.nPoints[2], 1))
            self.mapVals['kMax'] = kMax
            self.mapVals['sampled'] = np.concatenate((kRD, kPH, kSL, data), axis=1) ## this is what is taken from .mat to show the image
            self.mapVals['sampledCartesian'] = self.mapVals['sampled']  # To sweep
            data = np.reshape(data, (self.nPoints[2], self.nPoints[1], self.nPoints[0]))

        return True

    def sequenceAnalysis(self, mode=None):
        nPoints = self.mapVals['nPoints']
        axesEnable = self.mapVals['axesEnable']
        self.mode = mode

        # Get axes in strings
        axes = self.mapVals['axesOrientation']
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
            result1 = {}
            result1['widget'] = 'curve'
            result1['xData'] = tVector
            result1['yData'] = [np.abs(sVector), np.real(sVector), np.imag(sVector)]
            result1['xLabel'] = 'Time (ms)'
            result1['yLabel'] = 'Signal amplitude (mV)'
            result1['title'] = "Signal"
            result1['legend'] = ['Magnitude', 'Real', 'Imaginary']
            result1['row'] = 0
            result1['col'] = 0

            result2 = {}
            result2['widget'] = 'curve'
            result2['xData'] = fVector
            result2['yData'] = [np.abs(iVector)]
            result2['xLabel'] = 'Frequency (kHz)'
            result2['yLabel'] = "Amplitude (a.u.)"
            result2['title'] = "Spectrum"
            result2['legend'] = ['Spectrum magnitude']
            result2['row'] = 1
            result2['col'] = 0

            self.output = [result1, result2]
            
        else:
            # Plot image
            image = np.abs(self.mapVals['image3D'])
            image = image/np.max(np.reshape(image,-1))*100

            # Image orientation
            imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            if not self.unlock_orientation: # Image orientation
                if self.axesOrientation[2] == 2:  # Sagittal
                    title = "Sagittal"
                    if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 1:  #OK
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        xLabel = "(-Y) A | PHASE | P (+Y)"
                        yLabel = "(-X) I | READOUT | S (+X)"
                        imageOrientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
                    else:
                        image = np.transpose(image, (0, 2, 1))
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        xLabel = "(-Y) A | READOUT | P (+Y)"
                        yLabel = "(-X) I | PHASE | S (+X)"
                        imageOrientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
                elif self.axesOrientation[2] == 1: # Coronal
                    title = "Coronal"
                    if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 2: #OK
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        image = np.flip(image, axis=0)
                        xLabel = "(+Z) R | PHASE | L (-Z)"
                        yLabel = "(-X) I | READOUT | S (+X)"
                        imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                    else:
                        image = np.transpose(image, (0, 2, 1))
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        image = np.flip(image, axis=0)
                        xLabel = "(+Z) R | READOUT | L (-Z)"
                        yLabel = "(-X) I | PHASE | S (+X)"
                        imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                elif self.axesOrientation[2] == 0:  # Transversal
                    title = "Transversal"
                    if self.axesOrientation[0] == 1 and self.axesOrientation[1] == 2:
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        xLabel = "(+Z) R | PHASE | L (-Z)"
                        yLabel = "(+Y) P | READOUT | A (-Y)"
                        imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                    else:  #OK
                        image = np.transpose(image, (0, 2, 1))
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        xLabel = "(+Z) R | READOUT | L (-Z)"
                        yLabel = "(+Y) P | PHASE | A (-Y)"
                        imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            else:
                xLabel = "%s axis" % axesStr[1]
                yLabel = "%s axis" % axesStr[0]
                title = "Image"

            result1 = {}
            result1['widget'] = 'image'
            result1['data'] = image
            result1['xLabel'] = xLabel
            result1['yLabel'] = yLabel
            result1['title'] = title
            result1['row'] = 0
            result1['col'] = 0

            result2 = {}
            result2['widget'] = 'image'
            if self.parFourierFraction==1:
                result2['data'] = np.log10(np.abs(self.mapVals['kSpace3D']))
            else:
                result2['data'] = np.abs(self.mapVals['kSpace3D'])
            result2['xLabel'] = "k%s"%axesStr[1]
            result2['yLabel'] = "k%s"%axesStr[0]
            result2['title'] = "k-Space"
            result2['row'] = 0
            result2['col'] = 1

            # DICOM TAGS
            # Image
            imageDICOM = np.transpose(image, (0, 2, 1))
            # If it is a 3d image
            if len(imageDICOM.shape) > 2:
                # Obtener dimensiones
                slices, rows, columns = imageDICOM.shape
                self.meta_data["Columns"] = columns
                self.meta_data["Rows"] = rows
                self.meta_data["NumberOfSlices"] = slices
                self.meta_data["NumberOfFrames"] = slices
            # if it is a 2d image
            else:
                # Obtener dimensiones
                rows, columns = imageDICOM.shape
                self.meta_data["Columns"] = columns
                self.meta_data["Rows"] = rows
                self.meta_data["NumberOfSlices"] = 1
                self.meta_data["NumberOfFrames"] = 1
            imgAbs = np.abs(imageDICOM)
            imgFullAbs = np.abs(imageDICOM) * (2 ** 15 - 1) / np.amax(np.abs(imageDICOM))
            x2 = np.amax(np.abs(imageDICOM))
            imgFullInt = np.int16(np.abs(imgFullAbs))
            imgFullInt = np.reshape(imgFullInt, (slices, rows, columns))
            arr = np.zeros((slices, rows, columns), dtype=np.int16)
            arr = imgFullInt
            self.meta_data["PixelData"] = arr.tobytes()
            self.meta_data["WindowWidth"] = 26373
            self.meta_data["WindowCenter"] = 13194
            self.meta_data["ImageOrientationPatient"] = imageOrientation_dicom
            resolution = self.mapVals['resolution'] * 1e3
            self.meta_data["PixelSpacing"] = [resolution[0], resolution[1]]
            self.meta_data["SliceThickness"] = resolution[2]
            # Sequence parameters
            self.meta_data["RepetitionTime"] = self.mapVals['repetitionTime']
            self.meta_data["EchoTime"] = self.mapVals['echoSpacing']
            self.meta_data["EchoTrainLength"] = self.mapVals['etl']

            # Add results into the output attribute (result1 must be the image to save in dicom)
            self.output = [result1, result2]

        # Reset rotation angle and dfov to zero
        self.mapVals['angle'] = self.angle
        self.mapVals['dfov'] = np.array(self.mapVals['dfov'])
        self.mapVals['dfov'][self.axesOrientation] = self.dfov.reshape(-1)
        self.mapVals['dfov'] = list(self.mapVals['dfov'])

        # Save results
        self.saveRawData()
        self.save_ismrmrd()
        

        self.mapVals['angle'] = 0.0
        self.mapVals['dfov'] = [0.0, 0.0, 0.0]
        try:
            self.sequence_list['RARE'].mapVals['angle'] = 0.0
            self.sequence_list['RARE'].mapVals['dfov'] = [0.0, 0.0, 0.0]
        except:
            pass
        hw.dfov = [0.0, 0.0, 0.0]

        if self.mode == 'Standalone':
            self.plotResults()

        return self.output

    def myPhantom(self):
        # Reorganize the fov
        n_pixels = self.nPoints[0]*self.nPoints[1]*self.nPoints[2]

        # Get x, y and z vectors in real (x, y, z) and relative (rd, ph, sl) coordinates
        rd = np.linspace(-self.fov[0] / 2, self.fov[0] / 2, self.nPoints[0])
        ph = np.linspace(-self.fov[1] / 2, self.fov[1] / 2, self.nPoints[1])
        if self.nPoints[2]==1:
            sl = sl = np.linspace(-0, 0, 1)
            p = np.array([0.01, 0.01, 0.0])
            p = p[self.axesOrientation]
        else:
            sl = np.linspace(-self.fov[2] / 2, self.fov[2] / 2, self.nPoints[2])
            p = np.array([0.01, 0.01, 0.01])
        ph, sl, rd = np.meshgrid(ph, sl, rd)
        rd = np.reshape(rd, (1, -1))
        ph = np.reshape(ph, (1, -1))
        sl = np.reshape(sl, (1, -1))
        pos_rela = np.concatenate((rd, ph, sl), axis=0)
        pos_real = pos_rela[self.axesOrientation, :]

        # Generate the phantom
        image = np.zeros((1, n_pixels))
        image = np.concatenate((pos_real, image), axis=0)
        r = 0.01
        for ii in range(n_pixels):
            d = np.sqrt((pos_real[0,ii] - p[0])**2 + (pos_real[1,ii] - p[1])**2 + (pos_real[2,ii] - p[2])**2)
            if d <= r:
                image[3, ii] = 1
        image_3d = np.reshape(image[3, :], self.nPoints[-1::-1])
        
        # Generate k-space
        kspace_3d = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(image_3d)))
        
        kspace = np.reshape(kspace_3d, (1, -1))
        
        return kspace

    def dummyAnalysis(self):
        # Get position vector
        fov = self.fov[0]
        n = self.nPoints[0]
        res = fov / n
        rd_vec = np.linspace(-fov / 2, fov / 2, n)

        # Get dummy data
        dummy_pulses = self.mapVals['dummyData'] * 1
        dummy_pulses = np.reshape(sig.decimate(np.reshape(dummy_pulses, -1),
                                               hw.oversamplingFactor,
                                               ftype='fir',
                                               zero_phase=True),
                                  (self.etl, -1))
        dummy1 = dummy_pulses[0, 10:-10]
        dummy2 = dummy_pulses[1, 10:-10]

        # Calculate 1d projections from odd and even echoes
        proj1 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dummy1)))
        proj2 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dummy2)))
        proj1 = proj1 / np.max(np.abs(proj1))
        proj2 = proj2 / np.max(np.abs(proj2))
        proj1[np.abs(proj1) < 0.1] = 0
        proj2[np.abs(proj2) < 0.1] = 0

        # Maks the results
        rd_1 = rd_vec[np.abs(proj1) != 0]
        proj1 = proj1[np.abs(proj1) != 0]
        rd_2 = rd_vec[np.abs(proj2) != 0]
        proj2 = proj2[np.abs(proj2) != 0]

        # Get phase
        phase1 = np.unwrap(np.angle(proj1))
        phase2 = np.unwrap(np.angle(proj2))

        # Do linear regression
        res1 = linregress(rd_1, phase1)
        res2 = linregress(rd_2, phase2)

        # Print info
        print('Info from dummy pulses:')
        print('Phase difference at iso-center: %0.1f º' % ((res2.intercept - res1.intercept) * 180 / np.pi))
        print('Phase slope difference %0.3f rads/m' % (res2.slope - res1.slope))
        
        
    def save_ismrmrd(self):
        """
        Save the current instance's data in ISMRMRD format.

        This method saves the raw data, header information, and reconstructed images to an HDF5 file
        using the ISMRMRD (Image Storage and Reconstruction format for MR Data) format.

        Steps performed:
        1. Generate a timestamp-based filename and directory path for the output file.
        2. Initialize the ISMRMRD dataset with the generated path.
        3. Populate the header and write the XML header to the dataset. Informations can be added.
        4. Reshape the raw data matrix and iterate over scans, slices, and phases to write each acquisition. WARNING : RARE sequence follows ind order to fill the k-space.
        5. Set acquisition flags and properties.
        6. Append the acquisition data to the dataset.
        7. Reshape and save the reconstructed images.
        8. Close the dataset.

        Attribute:
        - self.data_full_mat (numpy.array): Full matrix of raw data to be reshaped and saved.

        Returns:
        None. It creates an HDF5 file with the ISMRMRD format.
        """
        
        directory_rmd = self.directory_rmd
        name = datetime.datetime.now()
        name_string = name.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3]
        self.mapVals['name_string'] = name_string
        if hasattr(self, 'raw_data_name'):
            file_name = "%s.%s" % (self.raw_data_name, name_string)
        else:
            self.raw_data_name = self.mapVals['seqName']
            file_name = "%s.%s" % (self.mapVals['seqName'], name_string)
            
        path= "%s/%s.h5" % (directory_rmd, file_name)
        
        dset = ismrmrd.Dataset(path, f'/dataset', True) # Create the dataset
        
        nScans = self.mapVals['nScans']
        nPoints = np.array(self.mapVals['nPoints'])
        etl = self.mapVals['etl']
        nRD = self.nPoints[0]
        nPH = self.nPoints[1]
        nSL = ((self.nPoints[2] // 2) + self.mapVals['partialAcquisition']) * self.axesEnable[2] + (1 - self.axesEnable[2])
        ind = self.getIndex(self.etl, nPH, self.sweepMode)
        nRep = (nPH//etl)*nSL
        bw = self.mapVals['bw']
        
        axesOrientation = self.axesOrientation
        axesOrientation_list = axesOrientation.tolist()

        read_dir = [0, 0, 0]
        phase_dir = [0, 0, 0]
        slice_dir = [0, 0, 0]

        read_dir[axesOrientation_list.index(0)] = 1
        phase_dir[axesOrientation_list.index(1)] = 1
        slice_dir[axesOrientation_list.index(2)] = 1
        
        # Experimental Conditions field
        exp = ismrmrd.xsd.experimentalConditionsType() 
        magneticFieldStrength = hw.larmorFreq*1e6/hw.gammaB
        exp.H1resonanceFrequency_Hz = hw.larmorFreq

        self.header.experimentalConditions = exp 

        # Acquisition System Information field
        sys = ismrmrd.xsd.acquisitionSystemInformationType() 
        sys.receiverChannels = 1 
        self.header.acquisitionSystemInformation = sys


        # Encoding field can be filled if needed
        encoding = ismrmrd.xsd.encodingType()  
        encoding.trajectory = ismrmrd.xsd.trajectoryType.CARTESIAN
        #encoding.trajectory =ismrmrd.xsd.trajectoryType[data.processing.trajectory.upper()]

        # # encoded and recon spaces 
        # efov = ismrmrd.xsd.fieldOfViewMm()  
        # efov.x =  
        # efov.y =
        # efov.z = 

        # rfov = ismrmrd.xsd.fieldOfViewMm() 
        # rfov.x = 
        # rfov.y = 
        # rfov.z = 

        # ematrix = ismrmrd.xsd.matrixSizeType()
        # rmatrix = ismrmrd.xsd.matrixSizeType()

        # ematrix.x = 
        # ematrix.y = 
        # ematrix.z = 
        # rmatrix.x = 
        # rmatrix.y = 
        # rmatrix.z = 

        # espace = ismrmrd.xsd.encodingSpaceType() 
        # espace.matrixSize = ematrix
        # espace.fieldOfView_mm = efov
        # rspace = ismrmrd.xsd.encodingSpaceType()
        # rspace.matrixSize = rmatrix
        # rspace.fieldOfView_mm = rfov
        
        # encoding.encodedSpace = espace
        # encoding.reconSpace = rspace

        # Encoding limits field can be filled if needed
        
        # limits = ismrmrd.xsd.encodingLimitsType()
        # limits1 = ismrmrd.xsd.limitType()
        # limits1.minimum = 
        # limits1.center = 
        # limits1.maximum = 
        # limits.kspace_encoding_step_1 = limits1

        # limits_rep = ismrmrd.xsd.limitType()
        # limits_rep.minimum = 
        # limits_rep.center = 
        # limits_rep.maximum = 
        # limits.repetition = limits_rep

        # limits_rest = ismrmrd.xsd.limitType()
        # limits_rest.minimum = 
        # limits_rest.center = 
        # limits_rest.maximum =
        # limits.kspace_encoding_step_0 = 
        # limits.slice = 
        # limits.average =
        # limits.contrast =
        # limits.kspaceEncodingStep2 = 
        # limits.phase =  
        # limits.segment = 
        # limits.set =

        # encoding.encodingLimits = limits
        # self.header.encoding.append(encoding)
        
        dset.write_xml_header(self.header.toXML()) # Write the header to the dataset
                
        
        
        new_data = np.zeros((nPH * nSL * nScans, nRD + 2*self.addRdPoints))
        new_data = np.reshape(self.dataFullmat, (nScans, nSL, nPH, nRD+ 2*self.addRdPoints))
        
        counter=0  
        for scan in range(nScans):
            for slice_idx in range(nSL):
                for phase_idx in range(nPH):
                    
                    line = new_data[scan, slice_idx, phase_idx, :]
                    line2d = np.reshape(line, (1, nRD+2*self.addRdPoints))
                    acq = ismrmrd.Acquisition.from_array(line2d, None)
                    
                    index_in_repetition = phase_idx % etl
                    current_repetition = (phase_idx // etl) + (slice_idx * (nPH // etl))
                    
                    acq.clearAllFlags()
                    
                    if index_in_repetition == 0: 
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_CONTRAST)
                    elif index_in_repetition == etl - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_CONTRAST)
                    
                    if ind[phase_idx]== 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_PHASE)
                    elif ind[phase_idx] == nPH - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_PHASE)
                    
                    if slice_idx == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                    elif slice_idx == nSL - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                        
                    if int(current_repetition) == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_REPETITION)
                    elif int(current_repetition) == nRep - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)
                        
                    if scan == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_AVERAGE)
                    elif scan == nScans-1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_AVERAGE)
                    
                    
                    counter += 1 
                    
                    # +1 to start at 1 instead of 0
                    acq.idx.repetition = int(current_repetition + 1)
                    acq.idx.kspace_encode_step_1 = ind[phase_idx]+1 # phase
                    acq.idx.slice = slice_idx + 1
                    acq.idx.contrast = index_in_repetition + 1
                    acq.idx.average = scan + 1 # scan
                    
                    acq.scan_counter = counter
                    acq.discard_pre = self.addRdPoints
                    acq.discard_post = self.addRdPoints
                    acq.sample_time_us = 1/bw
                    self.dfov = np.array(self.dfov)
                    acq.position = (ctypes.c_float * 3)(*self.dfov.flatten())

                    
                    acq.read_dir = (ctypes.c_float * 3)(*read_dir)
                    acq.phase_dir = (ctypes.c_float * 3)(*phase_dir)
                    acq.slice_dir = (ctypes.c_float * 3)(*slice_dir)
                    
                    dset.append_acquisition(acq) # Append the acquisition to the dataset
                        
                        
        image=self.mapVals['image3D']
        image_reshaped = np.reshape(image, (self.nPoints[::-1]))
        
        for slice_idx in range (nSL): ## image3d does not have scan dimension
            
            image_slice = image_reshaped[slice_idx, :, :]
            img = ismrmrd.Image.from_array(image_slice)
            img.transpose = False
            img.field_of_view = (ctypes.c_float * 3)(*(self.fov)*10) # mm
           
            img.position = (ctypes.c_float * 3)(*self.dfov)
            
            # img.data_type= 8 ## COMPLEX FLOAT
            img.image_type = 5 ## COMPLEX
            
            
            
            img.read_dir = (ctypes.c_float * 3)(*read_dir)
            img.phase_dir = (ctypes.c_float * 3)(*phase_dir)
            img.slice_dir = (ctypes.c_float * 3)(*slice_dir)
            
            dset.append_image(f"image_raw", img) # Append the image to the dataset
                
        
        dset.close()    


if __name__ == '__main__': 

        
    seq = RARE()
    seq.sequenceAtributes()

    # A
    seq.sequenceRun(plotSeq=False, demo=True)
    # seq.sequencePlot(standalone=True)

    # # B
    # seq.sequenceRun(demo=True, plotSeq=False)
    seq.sequenceAnalysis(mode='Standalone')
