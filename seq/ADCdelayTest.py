import experiment as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw
from plotview.spectrumplot import SpectrumPlot

class ADCdelayTest(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(ADCdelayTest, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='ADCdelayTest', val='ADCdelayTest')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=8.31, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.1, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ')
        self.addParameter(key='nPoints', string='Number of points', val=1000, field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=1.0, field='IM')
        self.addParameter(key='addRdPoints', string='Add Rd Points', val=0, field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')

    def sequenceInfo(self):
        print(" ")
        print("ADCdelayTest")

    def sequenceRun(self, plotSeq=0):
        init_gpa = False  # Starts the gpa

        # Create input parameters
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq'] # MHz
        rfExAmp = self.mapVals['rfExAmp']
        repetitionTime = self.mapVals['repetitionTime']*1e3 # us
        acqTime = self.mapVals['acqTime']*1e3 # us
        nPoints = self.mapVals['nPoints']
        txChannel = self.mapVals['txChannel']
        rxChannel = self.mapVals['rxChannel']
        addRdPoints = self.mapVals['addRdPoints']

        # Miscellaneus

        def createSequence():
            tRx = 20
            self.rxGate(tRx, acqTimeReal, rxChannel=rxChannel)
            self.rfRecPulse(tRx+0/bwReal, 300, rfExAmp, 0, txChannel=txChannel)
            self.endSequence(repetitionTime*nScans)


        # Initialize the experiment
        bw = nPoints / acqTime  # MHz
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriodReal = self.expt.get_rx_ts()[0]
        bwReal = 1 / samplingPeriodReal  # MHz
        acqTimeReal = nPoints / bwReal  # us
        self.mapVals['bw'] = bwReal
        createSequence()

        if plotSeq == 0:
            # Run the experiment and get data
            rxd, msgs = self.expt.run()
            rxd['rx%i' % rxChannel] = np.real(rxd['rx%i'%rxChannel])-1j*np.imag(rxd['rx%i'%rxChannel])
            overData = rxd['rx%i'%rxChannel]*13.788
            # dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            dataFull = overData
            self.mapVals['overData'] = overData
            data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
            self.mapVals['data'] = data
            self.expt.__del__()

            # Save data to sweep plot (single point)
            self.mapVals['sampledPoint'] = data[0]

    def sequenceAnalysis(self, obj=''):
        addRdPoints = self.mapVals['addRdPoints']
        signal = self.mapVals['data'][addRdPoints::]
        signal = np.reshape(signal, (-1))
        acqTime = self.mapVals['acqTime'] # ms
        bw = self.mapVals['bw']*1e3 # kHz
        nPoints = self.mapVals['nPoints']


        tVector = np.linspace(0, acqTime, nPoints)*1e3 #us
        nVector = np.linspace(1, nPoints, nPoints)
        fVector = np.linspace(-bw/2, bw/2, nPoints)
        spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal))))
        spectrum = np.reshape(spectrum, -1)

        # Get max and FHWM
        spectrum = np.abs(spectrum)
        maxValue = np.max(spectrum)
        maxIndex = np.argmax(spectrum)
        spectrumA = np.abs(spectrum[0:maxIndex]-maxValue)
        spectrumB = np.abs(spectrum[maxIndex:nPoints]-maxValue)
        indexA = np.argmin(spectrumA)
        indexB = np.argmin(spectrumB)+maxIndex
        freqA = fVector[indexA]
        freqB = fVector[indexB]

        self.saveRawData()

        # Add time signal to the layout
        signalPlotWidget = SpectrumPlot(xData=nVector,
                                        yData=[np.abs(signal)],
                                        legend=['abs'],
                                        xLabel='Points',
                                        yLabel='Signal amplitude (mV)',
                                        title='Signal vs Npoints, BWacq=%0.1f kHz' % bw)
        signalPlotWidget.plotitem.curves[0].setSymbol('x')
        # Add frequency spectrum to the layout
        spectrumPlotWidget = SpectrumPlot(xData=tVector,
                                        yData=[np.abs(signal)],
                                        legend=['abs'],
                                        xLabel='Time (us)',
                                        yLabel='Signal amplitude (mV)',
                                        title='Signal vs time, BWacq=%0.1f kHz' % bw)
        # spectrumPlotWidget.plotitem.setLogMode(y=True)


        return([signalPlotWidget, spectrumPlotWidget])