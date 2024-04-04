"""
@author: J.M. Algarín, february 03th 2022
MRILAB @ I3M
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
import controller.experiment_gui as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import configs.hw_config as hw
import configs.units as units

class Noise(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(Noise, self).__init__()
        # Input the parameters
        self.rxChannel = None
        self.nPoints = None
        self.bw = None
        self.freqOffset = None
        self.addParameter(key='seqName', string='NoiseInfo', val='Noise')
        self.addParameter(key='freqOffset', string='RF frequency offset (kHz)', val=0.0, units=units.kHz, field='RF')
        self.addParameter(key='nPoints', string='Number of points', val=2500, field='RF')
        self.addParameter(key='bw', string='Acquisition bandwidth (kHz)', val=50.0, units=units.kHz, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')

    def sequenceInfo(self):
        print(" ")
        print("Noise")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Get a noise measurement")
        print(" ")

    def sequenceTime(self):
        return(0)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False
        self.demo = demo

        # Fix units to MHz and us
        self.freqOffset *= 1e-6 # MHz
        self.bw *= 1e-6 # MHz

        if self.demo:
            dataR = np.random.randn((self.nPoints + 2 * hw.addRdPoints) * hw.oversamplingFactor)
            dataC = np.random.randn((self.nPoints + 2 * hw.addRdPoints) * hw.oversamplingFactor)
            data = dataR+1j*dataC
            data = self.decimate(dataOver=data, nRdLines=1, option='Normal')
            acqTime = self.nPoints/self.bw
            tVector = np.linspace(0, acqTime, num=self.nPoints) * 1e-3  # ms
            spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
            fVector = np.linspace(-self.bw / 2, self.bw / 2, num=self.nPoints) * 1e3  # kHz
            self.dataTime = [tVector, data]
            self.dataSpec = [fVector, spectrum]
        else:
            samplingPeriod = 1 / self.bw
            self.expt = ex.Experiment(lo_freq=hw.larmorFreq + self.freqOffset,
                                      rx_t=samplingPeriod,
                                      init_gpa=init_gpa,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                      print_infos=False)
            samplingPeriod = self.expt.getSamplingRate()
            self.bw = 1/samplingPeriod
            acqTime = self.nPoints/self.bw

            # SEQUENCE
            self.iniSequence(20, np.array((0, 0, 0)))
            t0 = 30 + hw.addRdPoints*hw.oversamplingFactor/self.bw
            self.rxGateSync(t0, acqTime, channel=self.rxChannel)
            t0 = t0 + acqTime + hw.addRdPoints*hw.oversamplingFactor/self.bw
            self.endSequence(t0+20)
            if self.floDict2Exp():
                print("\nSequence waveforms loaded successfully")
                pass
            else:
                print("\nERROR: sequence waveforms out of hardware bounds")
                return False

            if plotSeq == 0:
                rxd, msgs = self.expt.run()
                data = self.decimate(rxd['rx%i' % self.rxChannel], 1, option='Normal')
                self.mapVals['data'] = data
                tVector = np.linspace(0, acqTime, num=self.nPoints) * 1e-3  # ms
                spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
                fVector = np.linspace(-self.bw / 2, self.bw / 2, num=self.nPoints) * 1e3  # kHz
                self.dataTime = [tVector, data]
                self.dataSpec = [fVector, spectrum]
            self.expt.__del__()

        return True

    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        noiserms = np.std(self.dataTime[1])
        self.mapVals['RMS noise'] = noiserms
        self.mapVals['sampledPoint'] = noiserms # for sweep method
        noiserms = noiserms*1e3
        print('\nrms noise: %0.5f uV' % noiserms)
        bw = self.mapVals['bw']*1e3 # Hz
        johnson = np.sqrt(2 * 50 * hw.temperature * bw * 1.38e-23) * 10 ** (hw.lnaGain / 20) * 1e6  # uV
        print('Expected by Johnson: %0.5f uV' % johnson)

        # Plot signal versus time
        result1 = {'widget': 'curve',
                   'xData': self.dataTime[0],
                   'yData': [np.abs(self.dataTime[1]), np.real(self.dataTime[1]), np.imag(self.dataTime[1])],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Noise vs time',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 0,
                   'col': 0}

        # Plot spectrum
        result2 = {'widget': 'curve',
                   'xData': self.dataSpec[0],
                   'yData': [np.abs(self.dataSpec[1])],
                   'xLabel': 'Frequency (kHz)',
                   'yLabel': 'Mag FFT (a.u.)',
                   'title': 'Noise spectrum',
                   'legend': [''],
                   'row': 1,
                   'col': 0}

        self.output = [result1, result2]

        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()

        # ####################
        # dataOver = self.mapVals['dataOver']
        # plt.plot(np.real(dataOver))
        # plt.plot(np.imag(dataOver))
        # plt.show()

        return self.output


if __name__=='__main__':
    seq = Noise()
    seq.sequenceAtributes()
    seq.sequenceRun(demo=False)
    seq.sequenceAnalysis(mode='Standalone')

