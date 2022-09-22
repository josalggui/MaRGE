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
        self.addParameter(key='bw', string='Acq BW (kHz)', val=100.0, field='IM')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')

    def sequenceInfo(self):
        print(" ")
        print("ADCdelayTest")
        print("Connect Tx to Rx.")

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
        repetitionTime = self.mapVals['repetitionTime']*1e3 # us
        bw = self.mapVals['bw']*1e-3 # MHz
        nPoints = self.mapVals['nPoints']
        txChannel = self.mapVals['txChannel']
        rxChannel = self.mapVals['rxChannel']

        # Miscellaneus

        def createSequence():
            tRx = 20
            self.rxGate(tRx, acqTimeReal, rxChannel=rxChannel)
            self.rfRecPulse(tRx, 100 / bwReal, rfExAmp, 0, txChannel=txChannel)
            self.endSequence(repetitionTime*nScans)


        # Initialize the experiment
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriodReal = self.expt.get_rx_ts()[0]
        bwReal = 1 / samplingPeriodReal  # MHz
        acqTimeReal = nPoints / bwReal  # us
        self.mapVals['acqTime'] = acqTimeReal
        self.mapVals['bw'] = bwReal*1e3 # kHz
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

            signal = self.mapVals['data']
            signal = np.reshape(signal, (-1))
            # Look for that first index with signal larger than 210 uV (criterion of be close to flat top of the RF pulse
            found = 0
            ii = 0
            while found == 0 and ii<nPoints:
                if abs(signal[ii]) > 210:
                    found = 1
                    self.mapVals['sampledPoint'] = ii
                ii += 1

    def sequenceAnalysis(self, obj=''):
        signal = self.mapVals['data']
        signal = np.reshape(signal, (-1))
        acqTime = self.mapVals['acqTime']*1e-3 # ms
        bw = self.mapVals['bw']*1e3 # kHz
        nPoints = self.mapVals['nPoints']
        tVector = np.linspace(0, acqTime, nPoints)*1e3 #us
        nVector = np.linspace(1, nPoints, nPoints)

        self.saveRawData()

        # Add time signal to the layout
        signalVsPointWidget = SpectrumPlot(xData=nVector,
                                        yData=[np.abs(signal)],
                                        legend=['abs'],
                                        xLabel='Points',
                                        yLabel='Signal amplitude (mV)',
                                        title='Signal vs Npoints, BWacq=%0.1f kHz' % bw)
        signalVsPointWidget.plotitem.curves[0].setSymbol('x')
        # Add frequency spectrum to the layout
        signalVsTimeWidget = SpectrumPlot(xData=tVector,
                                        yData=[np.abs(signal)],
                                        legend=['abs'],
                                        xLabel='Time (us)',
                                        yLabel='Signal amplitude (mV)',
                                        title='Signal vs time, BWacq=%0.1f kHz' % bw)


        return([signalVsPointWidget, signalVsTimeWidget])