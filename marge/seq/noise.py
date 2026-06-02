"""
@author: J.M. Algarín, february 03th 2022
MRILAB @ I3M
"""


import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
import marge.controller.experiment_gui as ex
import numpy as np
from marge.seq.mriBlankSeq import MRIBLANKSEQ
import marge.configs.hw_config as hw
import marge.configs.units as units

class Noise(MRIBLANKSEQ):
    def __init__(self):
        super().__init__()

        # Input the parameters
        self.repetitionTime = None
        self.rxChannel = None
        self.nPoints = None
        self.bw = None
        self.freqOffset = None
        self.addParameter(key='seqName', string='NoiseInfo', val='Noise')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='freqOffset', string='RF frequency offset (kHz)', val=0.0, units=units.kHz, field='RF')
        self.addParameter(key='nPoints', string='Number of points', val=2500, field='RF')
        self.addParameter(key='bw', string='Acquisition bandwidth (kHz)', val=50.0, units=units.kHz, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=500.0, field='RF', units=units.ms)
        self.addParameter(key='sleepTime', string='Sleep Time (s)', val=0.0, field='OTH')

    def sequenceInfo(self):
        print("Noise")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Get a noise measurement\n")

    def sequenceTime(self):
        return(0)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False
        self.demo = demo

        # Fix units to MHz and us
        self.freqOffset *= 1e-6  # MHz
        self.bw *= 1e-6  # MHz
        self.repetitionTime *= 1e6  # us

        self.mapVals['larmorFreq'] = hw.larmorFreq

        if self.demo:
            dataR = np.random.randn((self.nPoints + 2 * hw.addRdPoints) * hw.oversamplingFactor)
            dataC = np.random.randn((self.nPoints + 2 * hw.addRdPoints) * hw.oversamplingFactor)
            data = dataR+1j*dataC
            data = self.decimate(data_over=data, n_adc=1, option='Normal')
            self.mapVals['data'] = data
            acqTime = self.nPoints/self.bw
            tVector = np.linspace(0, acqTime, num=self.nPoints) * 1e-3  # ms
            spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
            fVector = np.linspace(-self.bw / 2, self.bw / 2, num=self.nPoints) * 1e3  # kHz
            self.dataTime = [tVector, data]
            self.dataSpec = [fVector, spectrum]
            time.sleep(self.repetitionTime*1e-6)
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
            self.ttlOffRecPulse(t0, acqTime)
            self.rxGateSync(t0, acqTime, channel=self.rxChannel)
            t0 = t0 + acqTime + hw.addRdPoints*hw.oversamplingFactor/self.bw
            if t0 < self.repetitionTime:
                self.endSequence(self.repetitionTime)
            else:
                self.endSequence(t0+20)

            # Load sequence to red pitaya
            if self.floDict2Exp():
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("ERROR: sequence waveforms out of hardware bounds")
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


if __name__=='__main__':
    seq = Noise()
    seq.sequenceAtributes()
    seq.sequenceRun(demo=True)
    seq.sequenceAnalysis(mode='Standalone')

