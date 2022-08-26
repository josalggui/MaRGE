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


class Larmor(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(Larmor, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='LarmorInfo', val='Larmor')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ')
        self.addParameter(key='bw', string='Bandwidth (kHz)', val=50, field='RF')
        self.addParameter(key='dF', string='Frequency resolution (Hz)', val=100, field='RF')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')

    def sequenceInfo(self):
        print(" ")
        print("Larmor")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs a single spin echo to find larmor")
        print(" ")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        return(repetitionTime*nScans/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0):
        init_gpa = False  # Starts the gpa
        demo = False

        # Create the inputs automatically. For some reason it only works if there is a few code later...
        # for key in self.mapKeys:
        #     if type(self.mapVals[key])==list:
        #         locals()[key] = np.array(self.mapVals[key])
        #     else:
        #         locals()[key] = self.mapVals[key]

        # I do not understand why I cannot create the input parameters automatically
        seqName = self.mapVals['seqName']
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq'] # MHz
        rfExAmp = self.mapVals['rfExAmp']
        rfExTime = self.mapVals['rfExTime'] # us
        rfReAmp = self.mapVals['rfReAmp']
        rfReTime = self.mapVals['rfReTime'] # us
        repetitionTime = self.mapVals['repetitionTime']*1e3 # us
        bw = self.mapVals['bw']*1e-3 # MHz
        dF = self.mapVals['dF']*1e-6 # MHz
        shimming = np.array(self.mapVals['shimming'])*1e-4

        # Calculate acqTime and echoTime
        nPoints = int(bw/dF)
        acqTime = 1/dF # us
        echoTime = 2*acqTime # us
        self.mapVals['nPoints'] = nPoints
        self.mapVals['acqTime'] = acqTime*1e-6
        self.mapVals['echoTime'] = echoTime*1e-6

        def createSequence():
            # Initialize time
            t0 = 20
            tEx = 20e3

            # Shimming
            self.iniSequence(t0, shimming)

            # Excitation pulse
            t0 = tEx - hw.blkTime - rfExTime / 2
            self.rfRecPulse(t0, rfExTime, rfExAmp, 0)

             # Refocusing pulse
            t0 = tEx + echoTime / 2 - hw.blkTime - rfReTime / 2
            self.rfRecPulse(t0, rfReTime, rfReAmp, np.pi / 2)

            # Rx gate
            t0 = tEx + echoTime - acqTime / 2
            self.rxGate(t0, acqTime)

            self.endSequence(repetitionTime)


        # Initialize the experiment
        bw = nPoints / acqTime * hw.oversamplingFactor  # MHz
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq,
                                  rx_t=samplingPeriod,
                                  init_gpa=init_gpa,
                                  gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                  print_infos=False)
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1 / samplingPeriod / hw.oversamplingFactor  # MHz
        acqTime = nPoints / bw  # us
        self.mapVals['bw'] = bw*1e3 # kHz
        createSequence()

        dataFull = []
        if plotSeq == 1:
            self.expt.__del__()
        elif plotSeq == 0:
            # Run the experiment and get data
            for ii in range(nScans):
                rxd, msgs = self.expt.run()
                rxd['rx0'] = np.real(rxd['rx0'])-1j*np.imag(rxd['rx0'])
                dataFull = np.concatenate((dataFull, rxd['rx0']*13.788), axis=0)
            dataFull = sig.decimate(dataFull, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['dataFull'] = dataFull
            data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
            self.mapVals['data'] = data
            self.expt.__del__()

            # Process data to be plotted
            self.results = [data]
            self.mapVals['sampledPoint'] = data[int(nPoints/2)]
        return 0

    def sequenceAnalysis(self, obj=''):
        signal = self.results
        signal = np.reshape(signal, (-1))
        acqTime = self.mapVals['acqTime']*1e3 # ms
        bw = self.mapVals['bw'] # kHz
        nPoints = self.mapVals['nPoints']
        larmorFreq = self.mapVals['larmorFreq']

        tVector = np.linspace(-acqTime/2, acqTime/2, nPoints)
        fVector = np.linspace(-bw/2, bw/2, nPoints)
        spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal)))
        spectrum = np.reshape(spectrum, (-1))

        idf = np.argmax(np.abs(spectrum))
        fCentral = fVector[idf]*1e-3
        print('Larmor frequency: %1.5f MHz' % (larmorFreq + fCentral))
        self.mapVals['larmorFreqCal'] = larmorFreq + fCentral
        self.mapVals['signalVStime'] = [tVector, signal]
        self.mapVals['spectrum'] = [fVector, spectrum]

        self.saveRawData()

        # Add time signal to the layout
        signalPlotWidget = SpectrumPlot(xData=tVector,
                                        yData=[np.abs(signal), np.real(signal), np.imag(signal)],
                                        legend=['abs', 'real', 'imag'],
                                        xLabel='Time (ms)',
                                        yLabel='Signal amplitude (mV)',
                                        title='')

        # Add frequency spectrum to the layout
        spectrumPlotWidget = SpectrumPlot(xData=fVector,
                                          yData=[np.abs(spectrum)],
                                          legend=[''],
                                          xLabel='Frequency (kHz)',
                                          yLabel='Spectrum amplitude (a.u.)',
                                          title='Larmor frequency: %1.5f MHz' % (larmorFreq + fCentral))

        # create self.out to run in iterative mode
        self.out = [[signalPlotWidget, spectrumPlotWidget]]

        return (self.out)