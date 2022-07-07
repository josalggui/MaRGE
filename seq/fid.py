"""
@author: José Miguel Algarín Guisado
@modifield: T. Guallart Naval, february 28th 2022
MRILAB @ I3M
"""

import experiment as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw
from plotview.spectrumplot import SpectrumPlot

class FID(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(FID, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='FIDinfo', val='FID')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
        self.addParameter(key='deadTime', string='RF dead time (us)', val=400, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='RF')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4, field='RF')
        self.addParameter(key='nPoints', string='Number of points', val=100, field='RF')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='RF')

    def sequenceInfo(self):
        print(" ")
        print("FID")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs a single FID")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        return(repetitionTime*nScans/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0):
        init_gpa = False  # Starts the gpa

        # Create input parameters
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq'] # MHz
        rfExAmp = self.mapVals['rfExAmp']
        rfExTime = self.mapVals['rfExTime'] # us
        deadTime = self.mapVals['deadTime'] # us
        repetitionTime = self.mapVals['repetitionTime']*1e3 # us
        acqTime = self.mapVals['acqTime']*1e3 # us
        nPoints = self.mapVals['nPoints']
        shimming = np.array(self.mapVals['shimming'])*1e-4

        def createSequence():
            # Initialize time
            t0 = 20
            tEx = t0 + hw.blkTime + rfExTime / 2

            # Shimming
            self.iniSequence(t0, shimming)

            # Excitation pulse
            t0 = tEx - hw.blkTime - rfExTime / 2
            self.rfRecPulse(t0, rfExTime, rfExAmp, 0)

            # Rx gate
            t0 = tEx + rfExTime / 2 + deadTime
            self.rxGate(t0, acqTime)

            self.endSequence(repetitionTime)


        # Initialize the experiment
        bw = nPoints / acqTime * hw.oversamplingFactor  # MHz
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1 / samplingPeriod / hw.oversamplingFactor  # MHz
        acqTime = nPoints / bw  # us
        self.mapVals['acqTime'] = acqTime*1e-3 # ms
        self.mapVals['bw'] = bw # MHz
        createSequence()

        overData = []
        if plotSeq == 0:
            # Run the experiment and get data
            for ii in range(nScans):
                rxd, msgs = self.expt.run()
                overData = np.concatenate((overData, rxd['rx0']*13.788), axis=0)
            print(msgs)
            dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['overData'] = overData
            self.mapVals['dataFull'] = dataFull
            data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
            self.mapVals['data'] = data
            self.expt.__del__()

            # Save data to sweep plot (single point)
            self.mapVals['sampledPoint'] = data[0]

    def sequenceAnalysis(self, obj=''):
        signal = self.mapVals['data']
        signal = np.reshape(signal, (-1))
        acqTime = self.mapVals['acqTime'] # ms
        bw = self.mapVals['bw']*1e3 # kHz
        nPoints = self.mapVals['nPoints']
        larmorFreq = self.mapVals['larmorFreq']
        deadTime = self.mapVals['deadTime']*1e-3 # ms
        rfExTime = self.mapVals['rfExTime']*1e-3 # ms

        tVector = np.linspace(rfExTime+deadTime, rfExTime+deadTime+acqTime, nPoints)
        fVector = np.linspace(-bw/2, bw/2, nPoints)
        spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal)))
        spectrum = np.reshape(spectrum, (-1))

        self.saveRawData()

        # Add time signal to the layout
        signalPlot = SpectrumPlot(tVector, [np.abs(signal), np.real(signal), np.imag(signal)], ['abs', 'real', 'imag'],
                                  'Time (ms)', 'Signal amplitude (mV)', 'Signal vs time')

        # Add frequency spectrum to the layout
        spectrumPlot = SpectrumPlot(fVector, [np.abs(spectrum)], [''], 'Frequency (kHz)', 'Spectrum amplitude (a.u.)',
                                    'Spectrum')

        return([signalPlot, spectrumPlot])