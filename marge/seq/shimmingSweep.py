"""
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia, Spain
@date: 19 tue Apr 2022
@email: josalggui@i3m.upv.es
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
import marge.marcos.marcos_client.experiment
import numpy as np
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import marge.configs.hw_config as hw
import marge.configs.units as units


class ShimmingSweep(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(ShimmingSweep, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='ShimmingSweepInfo', val='Shimming')
        self.addParameter(key='toMaRGE', val=True)
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
        self.addParameter(key='shimming0', string='Shimming', val=[-12.5, -12.5, 7.5], units=units.sh, field='OTH')
        self.addParameter(key='nShimming', string='n Shimming steps', val=10, field='OTH')
        self.addParameter(key='dShimming', string='Shiming step', val=[2.5, 2.5, 2.5], units=units.sh, field='OTH')

    def sequenceInfo(self):
        
        print("Shimming")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence sweep the shimming in the three axis\n")

    def sequenceTime(self):
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        nShimming = self.mapVals['nShimming']
        return (repetitionTime * nShimming * 3 / 60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        self.plot_seq = plotSeq
        self.demo = demo

        # Calculate the rf amplitudes
        self.rfExAmp = self.rfExFA * np.pi / 180 / (self.rfExTime*1e6 * hw.b1Efficiency)
        self.rfReAmp = self.rfReFA * np.pi / 180 / (self.rfReTime*1e6 * hw.b1Efficiency)

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
        if self.shimming(axis='x'):
            pass
        else:
            return False
        if self.shimming(axis='y'):
            pass
        else:
            return False
        if self.shimming(axis='z'):
            pass
        else:
            return False

        return True



    def sequenceAnalysis(self, mode=None):
        self.mode = mode

        # Get data
        data = np.reshape(self.mapVals['data'], (3, self.nShimming, -1))

        def getFHWM(s=None):
            bw = self.mapVals['bw']*1e-3
            f_vector = np.linspace(-bw/2, bw/2, self.nPoints)
            target = np.max(s) / 2
            p0 = np.argmax(s)
            f0 = f_vector[p0]
            s1 = np.abs(s[0:p0]-target)
            f1 = f_vector[np.argmin(s1)]
            s2 = np.abs(s[p0::]-target)
            f2 = f_vector[np.argmin(s2)+p0]
            return f2-f1

        # Get FFT
        dataFFT = np.zeros((3, self.nShimming))
        dataFWHM = np.zeros((3, self.nShimming))
        for ii in range(3):
            for jj in range(self.nShimming):
                spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data[ii, jj, :]))))
                dataFFT[ii, jj] = np.max(spectrum)
                dataFWHM[ii, jj] = getFHWM(spectrum)
        self.mapVals['amplitudeVSshimming'] = dataFFT

        # Get max signal for each excitation
        sxVector = np.squeeze(self.mapVals['sxVector'])
        syVector = np.squeeze(self.mapVals['syVector'])
        szVector = np.squeeze(self.mapVals['szVector'])

        # Get the shimming values
        sx = sxVector[np.argmax(dataFFT[0, :])]
        sy = syVector[np.argmax(dataFFT[1, :])]
        sz = szVector[np.argmax(dataFFT[2, :])]
        fwhm = dataFWHM[2, np.argmax(dataFFT[2, :])]
        print("Shimming X = %0.1f" % (sx / units.sh))
        print("Shimming Y = %0.1f" % (sy / units.sh))
        print("Shimming Z = %0.1f" % (sz / units.sh))
        print("FHWM = %0.0f Hz" % (fwhm*1e3))
        print("Homogeneity = %0.0f ppm" % (fwhm*1e3/hw.larmorFreq))
        print("Shimming loaded into the sequences.")

        # Shimming plot
        result1 = {'widget': 'curve',
                   'xData': [sxVector / units.sh, syVector / units.sh, szVector / units.sh],
                   'yData': [np.abs(dataFFT[0, :]), np.abs(dataFFT[1, :]), np.abs(dataFFT[2, :])],
                   'xLabel': 'Shimming',
                   'yLabel': 'a.u.',
                   'title': 'Spectrum amplitude',
                   'legend': ['X', 'Y', 'Z'],
                   'row': 0,
                   'col': 0}

        result2 = {'widget': 'curve',
                   'xData': [sxVector / units.sh, syVector / units.sh, szVector / units.sh],
                   'yData': [dataFWHM[0, :], dataFWHM[1, :], dataFWHM[2, :]],
                   'xLabel': 'Shimming',
                   'yLabel': 'FHWM (kHz)',
                   'title': 'FWHM',
                   'legend': ['X', 'Y', 'Z'],
                   'row': 2,
                   'col': 0}

        # Update the shimming in hw_config
        if mode != "standalone":
            for seqName in self.sequence_list:
                self.sequence_list[seqName].mapVals['shimming'] = [np.round(sx / units.sh, decimals=1),
                                                                  np.round(sy / units.sh, decimals=1),
                                                                  np.round(sz / units.sh, decimals=1)]
        shimming = [np.round(sx / units.sh, decimals=1),
                    np.round(sy / units.sh, decimals=1),
                    np.round(sz / units.sh, decimals=1)]
        self.mapVals['shimming0'] = shimming

        self.output = [result1, result2]

        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()

        return self.output

    def createSequence(self):
        self.iniSequence(20, [0.0, 0.0, 0.0])

        for repeIndex in range(self.nShimming + self.dummyPulses):
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
        self.endSequence((self.nShimming + self.dummyPulses) * self.repetitionTime)

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
        if not self.demo:
            self.expt = ex.Experiment(lo_freq=hw.larmorFreq + self.freqOffset,
                                      rx_t=samplingPeriod,
                                      init_gpa=False,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                      )
            samplingPeriod = self.expt.get_rx_ts()[0]
        self.mapVals['samplingPeriod'] = samplingPeriod
        bw = 1 / samplingPeriod / hw.oversamplingFactor  # MHz
        self.mapVals['bw'] = bw * 1e6  # Hz
        self.acqTime = self.nPoints / bw  # us

        # Create sequence and load it to red pitaya
        self.createSequence()
        if self.floDict2Exp(demo=self.demo):
            print("Sequence waveforms loaded successfully")
            pass
        else:
            print("ERROR: sequence waveforms out of hardware bounds")
            return False

        # Run experiment and get best shimming for current axis
        if not self.plot_seq:
            if not self.demo:
                rxd, msgs = self.expt.run()
                print(msgs)
                self.expt.__del__()
            else:
                rxd = {'rx0': np.random.randn(self.nPoints * self.nShimming * hw.oversamplingFactor) +
                              1j * np.random.randn(self.nPoints * self.nShimming * hw.oversamplingFactor)}
            data = sig.decimate(rxd['rx0'] * hw.adcFactor, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['data'] = np.concatenate((self.mapVals['data'], data), axis=0)
            data = np.reshape(data, (self.nShimming, -1))
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
    seq.sequenceAtributes()
    seq.sequenceRun(demo=True)
    seq.sequenceAnalysis(mode='Standalone')
