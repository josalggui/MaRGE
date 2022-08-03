"""
@author: T. Guallart Naval, february 03th 2022
MRILAB @ I3M
"""

import os
import sys
#*****************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************
import time
import experiment as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw
from plotview.spectrumplot import SpectrumPlot

class Noise(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(Noise, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='NoiseInfo', val='Noise')
        self.addParameter(key='larmorFreq', string='Central frequency (MHz)', val=3.00, field='RF')
        self.addParameter(key='nPoints', string='Number of points', val=2500, field='RF')
        self.addParameter(key='bw', string='Acquision bandwidth (kHz)', val=50.0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')

    def sequenceInfo(self):
        print(" ")
        print("Noise")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Get a noise measurement")

    def sequenceTime(self):
        return(0)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0):
        init_gpa = False
        demo = False

        # Create inputs parameters
        seqName = self.mapVals['seqName']
        larmorFreq = self.mapVals['larmorFreq'] # MHz
        nPoints = self.mapVals['nPoints']
        bw = self.mapVals['bw']*1e-3 # MHz
        rxChannel = self.mapVals['rxChannel']

        if demo:
            data = np.random.randn(nPoints*hw.oversamplingFactor)
            acqTime = nPoints/bw
        else:
            bw = bw * hw.oversamplingFactor
            samplingPeriod = 1 / bw
            self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
            samplingPeriod = self.expt.get_rx_ts()[0]
            bw = 1/samplingPeriod/hw.oversamplingFactor
            acqTime = nPoints/bw

            # SEQUENCE
            # Rx gate
            t0 = 20
            self.iniSequence(20, np.array((0, 0, 0)))
            self.rxGate(t0, acqTime, rxChannel=rxChannel)
            self.endSequence(2*acqTime)

        if plotSeq == 0:
            print('Running...')
            t0 = time.time()
            rxd, msgs = self.expt.run()
            t1 = time.time()
            print('Run time = %f s' %(t1-t0))
            # print(msgs)
            self.expt.__del__()
            data = sig.decimate(rxd['rx%i'%rxChannel]*13.788, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['data'] = data
            print('End')
            tVector = np.linspace(0, acqTime, num=nPoints) * 1e-3  # ms
            spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
            fVector = np.linspace(-bw / 2, bw / 2, num=nPoints) * 1e3  # kHz
            self.dataTime = [tVector, data]
            self.dataSpec = [fVector, spectrum]
        elif plotSeq == 1:
            self.expt.__del__()


    def sequenceAnalysis(self, obj=''):
        noise = np.abs(self.dataTime[1])
        noiserms = np.mean(noise)
        self.mapVals['RMS noise'] = noiserms
        self.mapVals['sampledPoint'] = noiserms # for sweep method
        self.saveRawData()

        # Plot signal versus time
        timePlot = SpectrumPlot(self.dataTime[0], [np.abs(self.dataTime[1]), np.real(self.dataTime[1]), np.imag(self.dataTime[1])],
                                ['abs', 'real', 'imag'],
                                'Time (ms)', 'Signal amplitude (mV)',
                                'Signal vs time, rms noise: %1.3f mV' %noiserms)

        # Plot spectrum
        freqPlot = SpectrumPlot(self.dataSpec[0], [np.abs(self.dataSpec[1])], [''],
                                'Frequency (kHz)', 'Mag FFT (a.u.)',
                                'Signal spectrum')

        return([timePlot, freqPlot])

if __name__=='__main__':
    seq = Noise()
    seq.sequenceRun()