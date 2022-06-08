"""
@author: T. Guallart Naval, february 03th 2022
MRILAB @ I3M
"""

import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw
from plotview.spectrumplot import SpectrumPlot

class Noise(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(Noise, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='Noise', val='Noise')
        self.addParameter(key='larmorFreq', string='Central frequency (MHz)', val=3.00, field='OTH')
        self.addParameter(key='nPoints', string='Number of points', val=2500, field='OTH')
        self.addParameter(key='bw', string='Acquision bandwidth (kHz)', val=50, field='OTH')

    def sequenceRun(self, plotSeq):
        init_gpa = False

        # Create inputs parameters
        seqName = self.mapVals['seqName']
        larmorFreq = self.mapVals['larmorFreq'] # MHz
        nPoints = self.mapVals['nPoints']
        bw = self.mapVals['bw']*1e-3 # MHz

        # Create rawData
        rawData = {}
        rawData['seqName'] = seqName
        rawData['larmorFreq'] = larmorFreq*1e6
        rawData['nPoints'] = nPoints
        rawData['bandwidth'] = bw*1e6

        # INIT EXPERIMENT
        bw = bw*hw.oversamplingFactor
        samplingPeriod = 1/bw
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1/samplingPeriod/hw.oversamplingFactor
        acqTime = nPoints/bw

        # SEQUENCE
        # Rx gate
        t0 = 20
        self.rxGate(t0, acqTime)

        if plotSeq == 0:
            print('Running...')
            rxd, msgs = self.expt.run()
            self.expt.__del__()
            print('End')
            data = sig.decimate(rxd['rx0']*13.788, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            rawData['data'] = data
            name = self.saveRawData(rawData)
        elif plotSeq == 1:
            expt.plot_sequence()
            plt.show()
            expt.__del__()

        tVector = np.linspace(0, acqTime, num=nPoints)*1e-3 # ms
        spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
        fVector = np.linspace(-bw/2, bw/2, num=nPoints)*1e3 # kHz
        self.dataTime = [tVector, data]
        self.dataSpec = [fVector, spectrum]

        return msgs

    def sequenceAnalysisGUI(self, obj):

        # Signal versus time
        timePlot = SpectrumPlot(self.dataTime[0],
                                np.abs(self.dataTime[1]),
                                [], [],
                                'Time (ms)', 'Signal amplitude (mV)',
                                "%s" % (self.mapVals['seqName']))

        # Spectrum
        freqPlot = SpectrumPlot(self.dataSpec[0],
                                np.abs(self.dataSpec[1]),
                                [], [],
                                'Frequency (kHz)', 'Mag FFT (a.u.)',
                                "%s" % (self.mapVals['seqName']))

        # Update figures
        obj.parent.plotview_layout.addWidget(timePlot)
        obj.parent.plotview_layout.addWidget(freqPlot)