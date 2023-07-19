"""
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia, Spain
@date: 19 tue Apr 2022
@email: josalggui@i3m.upv.es
"""

import os
import sys

# *****************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char == '\\' or char == '/') and path[ii + 1:ii + 14] == 'PhysioMRI_GUI':
        sys.path.append(path[0:ii + 1] + 'PhysioMRI_GUI')
        sys.path.append(path[0:ii + 1] + 'marcos_client')
    ii += 1
# ******************************************************************************
import experiment as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw
import configs.units as units


class ShimmingSweep(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(ShimmingSweep, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='ShimmingSweepInfo', val='Shimming')
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, units=units.kHz, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90.0, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180.0, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, units=units.us, field='RF')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=10., units=units.ms, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., units=units.ms, field='SEQ')
        self.addParameter(key='nPoints', string='nPoints', val=60, field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, units=units.ms, field='SEQ')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=0, field='SEQ')
        self.addParameter(key='shimming0', string='Shimming (*1e4)', val=[-12.5, -12.5, 7.5], units=units.sh, field='OTH')
        self.addParameter(key='nShimming', string='n Shimming steps', val=10, field='OTH')
        self.addParameter(key='dShimming', string='Shiming step', val=[2.5, 2.5, 2.5], field='OTH')

    def sequenceInfo(self):
        print(" ")
        print("Shimming")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence sweep the shimming in the three axis")

    def sequenceTime(self):
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        nShimming = self.mapVals['nShimming']
        return (repetitionTime * nShimming * 3 / 60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        self.plot_seq = plotSeq
        self.demo = demo

        # Calculate the rf amplitudes
        self.rfExAmp = self.rfExFA / (self.rfExTime * hw.b1Efficiency)
        self.rfReAmp = self.rfReFA / (self.rfReTime * hw.b1Efficiency)

        # Shimming vectors
        dsx = self.nShimming * self.dShimming[0]
        dsy = self.nShimming * self.dShimming[1]
        dsz = self.nShimming * self.dShimming[2]
        sxVector = np.reshape(
            np.linspace(self.shimming0[0] - dsx / 2, self.shimming0[0] + dsx / 2, num=self.nShimming, endpoint=False),
            (self.nShimming, 1))
        syVector = np.reshape(
            np.linspace(self.shimming0[1] - dsy / 2, self.shimming0[1] + dsy / 2, num=self.nShimming, endpoint=False),
            (self.nShimming, 1))
        szVector = np.reshape(
            np.linspace(self.shimming0[2] - dsz / 2, self.shimming0[2] + dsz / 2, num=self.nShimming, endpoint=False),
            (self.nShimming, 1))
        self.mapVals['sxVector'] = sxVector
        self.mapVals['syVector'] = syVector
        self.mapVals['szVector'] = szVector

        # Set time parameters to us
        self.repetitionTime *= 1e6
        self.rfExTime *= 1e6
        self.echoTime *= 1e6
        self.rfReTime *= 1e6
        self.acqTime *= 1e6

        # Perform shimming
        self.mapVals['data'] = np.array([])
        self.shimming(axis='x')
        self.shimming(axis='y')
        self.shimming(axis='z')



    def sequenceAnalysis(self, obj=''):
        # Get data
        data = np.reshape(self.mapVals['data'], (3, self.nShimming, -1))

        # Get FFT
        dataFFT = np.zeros((3, self.nShimming), dtype=complex)
        for ii in range(3):
            for jj in range(self.nShimming):
                dataFFT[ii, jj] = np.max(np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data[ii, jj, :])))))
        self.mapVals['amplitudeVSshimming'] = dataFFT

        # Get max signal for each excitation
        sxVector = np.squeeze(self.mapVals['sxVector'])
        syVector = np.squeeze(self.mapVals['syVector'])
        szVector = np.squeeze(self.mapVals['szVector'])

        # Get the shimming values
        sx = sxVector[np.argmax(dataFFT[0, :])]
        sy = syVector[np.argmax(dataFFT[1, :])]
        sz = szVector[np.argmax(dataFFT[2, :])]
        print("Shimming X = %0.1f" % (sx * 1e4))
        print("Shimming Y = %0.1f" % (sy * 1e4))
        print("Shimming Z = %0.1f" % (sz * 1e4))

        # Shimming plot
        result1 = {'widget': 'curve',
                   'xData': sxVector * 1e4,
                   'yData': [np.abs(dataFFT[0, :]), np.abs(dataFFT[1, :]), np.abs(dataFFT[2, :])],
                   'xLabel': 'Shimming',
                   'yLabel': 'Spectrum amplitude',
                   'title': 'Shimming',
                   'legend': ['X', 'Y', 'Z'],
                   'row': 0,
                   'col': 0}

        # Update the shimming in hw_config
        if obj != "standalone":
            for seqName in self.sequenceList:
                self.sequenceList[seqName].mapVals['shimming'] = [np.round(sx * 1e4, decimals=1),
                                                                  np.round(sy * 1e4, decimals=1),
                                                                  np.round(sz * 1e4, decimals=1)]

        self.saveRawData()
        shimming = [np.round(sx * 1e4, decimals=1), np.round(sy * 1e4, decimals=1), np.round(sz * 1e4, decimals=1)]
        self.mapVals['shimming0'] = shimming

        self.out = [result1]
        return self.out

    def createSequence(self):
        self.iniSequence(20, [0.0, 0.0, 0.0])

        for repeIndex in range((3 * self.nShimming) + self.dummyPulses):
            # Set time for repetition
            t0 = 40 + repeIndex * self.repetitionTime

            # Set shimming
            self.setGradient(t0, self.shimmingMatrix[repeIndex, 0], 0)
            self.setGradient(t0, self.shimmingMatrix[repeIndex, 1], 1)
            self.setGradient(t0, self.shimmingMatrix[repeIndex, 2], 2)

            # Initialize time
            tEx = t0 + 20e3

            # Excitation pulse
            t0 = tEx - hw.blkTime - self.rfExTime / 2
            self.rfRecPulse(t0, self.rfExTime, self.rfExAmp, 0)

            # Refocusing pulse
            t0 = tEx + self.echoTime / 2 - self.rfReTime / 2 - hw.blkTime
            self.rfRecPulse(t0, self.rfReTime, self.rfReAmp, np.pi / 2)

            # Acquisition window
            if repeIndex >= self.dummyPulses:
                t0 = tEx + self.echoTime - self.acqTime / 2
                self.rxGate(t0, self.acqTime)

        # End sequence
        self.endSequence((3 * self.nShimming + self.dummyPulses) * self.repetitionTime)

    def shimming(self, axis='x'):
        # Create shimming matrix
        sxVector = self.mapVals['sxVector']
        syVector = self.mapVals['syVector']
        szVector = self.mapVals['szVector']
        if axis=='x':
            syStatic = np.reshape(np.ones(self.nShimming) * self.shimming0[1], (self.nShimming, 1))
            szStatic = np.reshape(np.ones(self.nShimming) * self.shimming0[2], (self.nShimming, 1))
            self.shimmingMatrix = np.concatenate((sxVector, syStatic, szStatic), axis=1)
        elif axis=='y':
            sxStatic = np.reshape(np.ones(self.nShimming) * self.shimming0[0], (self.nShimming, 1))
            szStatic = np.reshape(np.ones(self.nShimming) * self.shimming0[2], (self.nShimming, 1))
            self.shimmingMatrix = np.concatenate((sxStatic, syVector, szStatic), axis=1)
        elif axis=='z':
            sxStatic = np.reshape(np.ones(self.nShimming) * self.shimming0[0], (self.nShimming, 1))
            syStatic = np.reshape(np.ones(self.nShimming) * self.shimming0[1], (self.nShimming, 1))
            self.shimmingMatrix = np.concatenate((sxStatic, syStatic, szVector), axis=1)
        s0 = np.zeros((self.dummyPulses, 3))
        self.shimmingMatrix = np.concatenate((s0, self.shimmingMatrix), axis=0)

        # Create experiment
        bw = self.nPoints / self.acqTime * hw.oversamplingFactor  # MHz
        samplingPeriod = 1 / bw
        self.expt = ex.Experiment(lo_freq=hw.larmorFreq + self.freqOffset,
                                  rx_t=samplingPeriod,
                                  init_gpa=False,
                                  gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                  )
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1 / samplingPeriod / hw.oversamplingFactor  # MHz
        self.mapVals['bw'] = bw * 1e6  # Hz
        self.acqTime = self.nPoints / bw  # us
        self.createSequence()
        if self.floDict2Exp():
            print("\nSequence waveforms loaded successfully")
            pass
        else:
            print("\nERROR: sequence waveforms out of hardware bounds")
            return False

        # Run experiment and get best shimming for current axis
        if not self.plot_seq:
            rxd, msgs = self.expt.run()
            self.expt.__del__()
            print(msgs)
            data = sig.decimate(rxd['rx0'] * hw.adcFactor, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['data'] = np.concatenate((self.mapVals['data'], data), axis=0)
            data = np.reshape(self.mapVals['data'], (3, self.nShimming, -1))
            dataFFT = np.zeros(self.nShimming)
            for ii in range(self.nShimming):
                dataFFT[ii] = np.max(np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data[ii, :])))))
            if axis=='x':
                self.shimming0[0] = sxVector[np.argmax(dataFFT)]
            elif axis=='y':
                self.shimming0[1] = syVector[np.argmax(dataFFT)]
            elif axis=='z':
                self.shimming0[2] = szVector[np.argmax(dataFFT)]
        return True

if __name__ == '__main__':
    seq = ShimmingSweep()
    seq.sequenceRun()
    seq.sequenceAnalysis(obj='Standalone')
