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


class ShimmingSweep(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(ShimmingSweep, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='ShimmingSweepInfo', val='Shimming')
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90.0, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180.0, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, field='RF')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=10., field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ')
        self.addParameter(key='nPoints', string='nPoints', val=60, field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='SEQ')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=0, field='SEQ')
        self.addParameter(key='shimming0', string='Shimming (*1e4)', val=[-12.5, -12.5, 7.5], field='OTH')
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
        init_gpa = False  # Starts the gpa
        demo = False

        # # Create the inputs automatically. For some reason it only works if there is a few code later...
        # for key in self.mapKeys:
        #     if type(self.mapVals[key])==list:
        #         locals()[key] = np.array(self.mapVals[key])
        #     else:
        #         locals()[key] = self.mapVals[key]

        # I do not understand why I cannot create the input parameters automatically
        seqName = self.mapVals['seqName']
        freqOffset = self.mapVals['freqOffset'] * 1e-3  # MHz
        rfExFA = self.mapVals['rfExFA'] / 180 * np.pi  # rads
        rfExTime = self.mapVals['rfExTime']  # us
        rfReFA = self.mapVals['rfReFA'] / 180 * np.pi  # rads
        rfReTime = self.mapVals['rfReTime']  # us
        echoTime = self.mapVals['echoTime'] * 1e3  # us
        repetitionTime = self.mapVals['repetitionTime'] * 1e3  # us
        nPoints = self.mapVals['nPoints']
        acqTime = self.mapVals['acqTime'] * 1e3  # us
        shimming0 = np.array(self.mapVals['shimming0']) * 1e-4
        nShimming = self.mapVals['nShimming']
        dShimming = np.array(self.mapVals['dShimming']) * 1e-4
        dummyPulses = self.mapVals['dummyPulses']

        # Calculate the rf amplitudes
        rfExAmp = rfExFA / (rfExTime * hw.b1Efficiency)
        rfReAmp = rfReFA / (rfReTime * hw.b1Efficiency)

        # Shimming vectors
        dsx = nShimming * dShimming[0]
        dsy = nShimming * dShimming[1]
        dsz = nShimming * dShimming[2]
        sxVector = np.reshape(
            np.linspace(shimming0[0] - dsx / 2, shimming0[0] + dsx / 2, num=nShimming, endpoint=False), (nShimming, 1))
        syVector = np.reshape(
            np.linspace(shimming0[1] - dsy / 2, shimming0[1] + dsy / 2, num=nShimming, endpoint=False), (nShimming, 1))
        szVector = np.reshape(
            np.linspace(shimming0[2] - dsz / 2, shimming0[2] + dsz / 2, num=nShimming, endpoint=False), (nShimming, 1))
        sxStatic = np.reshape(np.ones(nShimming) * shimming0[0], (nShimming, 1))
        syStatic = np.reshape(np.ones(nShimming) * shimming0[1], (nShimming, 1))
        szStatic = np.reshape(np.ones(nShimming) * shimming0[2], (nShimming, 1))
        sx = np.concatenate((sxVector, syStatic, szStatic), axis=1)
        sy = np.concatenate((sxStatic, syVector, szStatic), axis=1)
        sz = np.concatenate((sxStatic, syStatic, szVector), axis=1)
        s0 = np.zeros((dummyPulses,3))
        shimmingMatrix = np.concatenate((s0 ,sx, sy, sz), axis=0)
        # shimmingMatrix = np.concatenate((s0, shimmingMatrix), axis=0)
        self.mapVals['sxVector'] = sxVector
        self.mapVals['syVector'] = syVector
        self.mapVals['szVector'] = szVector

        #  SEQUENCE  ############################################################################################
        def createSequence():
            self.iniSequence(20, [0.0, 0.0, 0.0])

            for repeIndex in range((3 * nShimming) + dummyPulses):
                # Set time for repetition
                t0 = 40 + repeIndex * repetitionTime

                # Set shimming
                self.setGradient(t0, shimmingMatrix[repeIndex, 0], 0)
                self.setGradient(t0, shimmingMatrix[repeIndex, 1], 1)
                self.setGradient(t0, shimmingMatrix[repeIndex, 2], 2)

                # Initialize time
                tEx = t0 + 20e3

                # Excitation pulse
                t0 = tEx - hw.blkTime - rfExTime / 2
                self.rfRecPulse(t0, rfExTime, rfExAmp, 0)

                # Refocusing pulse
                t0 = tEx + echoTime / 2 - rfReTime / 2 - hw.blkTime
                self.rfRecPulse(t0, rfReTime, rfReAmp, np.pi / 2)

                # Acquisition window
                if repeIndex >= self.dummyPulses:
                    t0 = tEx + echoTime - acqTime / 2
                    self.rxGate(t0, acqTime)

            # End sequence
            self.endSequence((3 * nShimming + dummyPulses) * repetitionTime)

        # Create experiment
        bw = nPoints / acqTime * hw.oversamplingFactor  # MHz
        samplingPeriod = 1 / bw
        self.expt = ex.Experiment(lo_freq=hw.larmorFreq + freqOffset,
                                  rx_t=samplingPeriod,
                                  init_gpa=init_gpa,
                                  gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                  )
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1 / samplingPeriod / hw.oversamplingFactor  # MHz
        self.mapVals['bw'] = bw * 1e6  # Hz
        acqTime = nPoints / bw  # us
        createSequence()
        if self.floDict2Exp():
            print("\nSequence waveforms loaded successfully")
            pass
        else:
            print("\nERROR: sequence waveforms out of hardware bounds")
            return False

        if not plotSeq:
            rxd, msgs = self.expt.run()
            print(msgs)
            data = sig.decimate(rxd['rx0'] * hw.adcFactor, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['data'] = data
        self.expt.__del__()

        return True

    def sequenceAnalysis(self, obj=''):
        # Get data
        nShimming = self.mapVals['nShimming']
        nPoints = self.mapVals['nPoints']
        data = np.reshape(self.mapVals['data'], (3, nShimming, -1))

        # Get FFT
        dataFFT = np.zeros((3, nShimming), dtype=complex)
        for ii in range(3):
            for jj in range(nShimming):
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


if __name__ == '__main__':
    seq = ShimmingSweep()
    seq.sequenceRun()
    seq.sequenceAnalysis(obj='Standalone')
