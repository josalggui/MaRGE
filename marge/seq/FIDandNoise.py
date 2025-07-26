"""
@author: José Miguel Algarín Guisado
@modifield: T. Guallart Naval, february 28th 2022
MRILAB @ I3M
"""

import marge.marcos.marcos_client.experiment
import numpy as np
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import marge.configs.hw_config as hw
import pyqtgraph as pg

class FIDandNoise(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(FIDandNoise, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='FIDandNoiseinfo', val='FIDandNoise')
        self.addParameter(key='toMaRGE', val=False)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
        self.addParameter(key='deadTime', string='RF dead time (us)', val=400.0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='SEQ')
        self.addParameter(key='nPoints', string='Number of points', val=100, field='IM')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='shimmingTime', string='Shimming time (ms)', val=1, field='OTH')

    def sequenceInfo(self):
        
        print("FIDandNoise")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs a single FID\n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        return(repetitionTime*nScans/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
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
        txChannel = self.mapVals['txChannel']
        rxChannel = self.mapVals['rxChannel']
        shimmingTime = self.mapVals['shimmingTime']*1e3 # us

        # Miscellaneus
        addRdPoints = 0
        self.mapVals['addRdPoints'] = addRdPoints

        def createSequence():
            # Shimming
            self.iniSequence(20, shimming)
            self.rxGate(30, acqTime + 2 * addRdPoints / bw, rxChannel=rxChannel)


            for scan in range(1, nScans+1):
                tEx = shimmingTime + repetitionTime*scan + hw.blkTime + rfExTime / 2

                # Excitation pulse
                t0 = tEx - hw.blkTime - rfExTime / 2
                self.rfRecPulse(t0, rfExTime, rfExAmp, 0, txChannel=txChannel)

                # Rx gate
                t0 = tEx + rfExTime / 2 + deadTime - addRdPoints / bw
                self.rxGate(t0, acqTime + 2 * addRdPoints / bw, rxChannel=rxChannel)



            self.endSequence(repetitionTime*(nScans+1))


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
        if self.floDict2Exp():
            print("Sequence waveforms loaded successfully")
            pass
        else:
            print("ERROR: sequence waveforms out of hardware bounds")
            return False

        if plotSeq == 0:
            # Run the experiment and get data
            rxd, msgs = self.expt.run()
            rxd['rx%i' % rxChannel] = np.real(rxd['rx%i'%rxChannel])-1j*np.imag(rxd['rx%i'%rxChannel])
            overData = rxd['rx%i'%rxChannel]*hw.adcFactor
            dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['overData'] = overData
            self.mapVals['dataFull'] = dataFull

            noisetemp = dataFull[0:nPoints]
            fidtemp = dataFull[nPoints:]

            data = np.average(np.reshape(fidtemp, (nScans, -1)), axis=0)
            self.mapVals['data'] = data
            self.expt.__del__()

            # Save data to sweep plot (single point)
            self.mapVals['sampledPoint'] = data[0]


            # Noise
            tVector = np.linspace(0, acqTime, nPoints) * 1e-3  # ms
            spectrumnoise = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(noisetemp)))
            fVector = np.linspace(-bw / 2, bw / 2, nPoints) * 1e3  # kHz
            self.dataTime = [tVector, noisetemp]
            self.dataSpec = [fVector, spectrumnoise]

        return True

    def sequenceAnalysis(self, obj=''):
        addRdPoints = self.mapVals['addRdPoints']
        nPoints = self.mapVals['nPoints']
        signal = self.mapVals['data'][addRdPoints:nPoints+addRdPoints]
        signal = np.reshape(signal, (-1))
        acqTime = self.mapVals['acqTime'] # ms
        bw = self.mapVals['bw']*1e3 # kHz
        nPoints = self.mapVals['nPoints']
        deadTime = self.mapVals['deadTime']*1e-3 # ms
        rfExTime = self.mapVals['rfExTime']*1e-3 # ms

        tVector = np.linspace(rfExTime/2+deadTime+0.5/bw, rfExTime/2+deadTime+acqTime-0.5/bw, nPoints)
        fVector = np.linspace(-bw/2, bw/2, nPoints)
        spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal))))
        spectrum = np.reshape(spectrum, -1)

        noise = np.abs(self.dataTime[1])
        noiserms = np.sqrt(2) * np.std(noise.real)*1e3 #uV
        johnson = np.sqrt(2 * 50 * hw.temperature * bw * 1.38e-23) * 10 ** (hw.lnaGain / 20) * 1e6  # uV
        self.mapVals['RMS noise'] = noiserms
        self.mapVals['sampledPoint'] = noiserms  # for sweep method
        print('rms noise: %0.5f uV' % noiserms)
        print('Expected by Johnson: %0.5f uV' % johnson)

        # Add time signal to the layout
        FIDsignalPlotWidget = SpectrumPlot(xData=tVector,
                                        yData=[np.abs(signal), np.real(signal), np.imag(signal)],
                                        legend=['abs', 'real', 'imag'],
                                        xLabel='Time (ms)',
                                        yLabel='Signal amplitude (mV)',
                                        title='Signal vs time')

        # Add frequency spectrum to the layout
        FIDspectrumPlotWidget = SpectrumPlot(xData=fVector,
                                          yData=[spectrum],
                                          legend=[''],
                                          xLabel='Frequency (kHz)',
                                          yLabel='Spectrum amplitude (a.u.)',
                                          title='Spectrum')


        NOISEtimePlotWidget = SpectrumPlot(xData=self.dataTime[0],
                                      yData=[np.abs(self.dataTime[1]), np.real(self.dataTime[1]),
                                             np.imag(self.dataTime[1])],
                                      legend=['abs', 'real', 'imag'],
                                      xLabel='Time (ms)',
                                      yLabel='Signal amplitude (mV)',
                                      title='Noise vs time, rms noise: %1.3f mV' % noiserms)

        # Plot spectrum
        NOISEfreqPlotWidget = SpectrumPlot(xData=self.dataSpec[0],
                                      yData=[np.abs(self.dataSpec[1])],
                                      legend=[''],
                                      xLabel='Frequency (kHz)',
                                      yLabel='Mag FFT (a.u.)',
                                      title='Noise spectrum')



        # create self.out to run in iterative mode
        self.output = [FIDsignalPlotWidget, NOISEfreqPlotWidget]

        self.saveRawData()

        return self.output



        # ****************************************************
        # addRdPoints = self.mapVals['addRdPoints']
        # nPoints = self.mapVals['nPoints']
        # signal = self.mapVals['data'][addRdPoints:nPoints + addRdPoints]
        # signal = np.reshape(signal, (-1))
        # acqTime = self.mapVals['acqTime']  # ms
        # bw = self.mapVals['bw'] * 1e3  # kHz
        # nPoints = self.mapVals['nPoints']
        # deadTime = self.mapVals['deadTime'] * 1e-3  # ms
        # rfExTime = self.mapVals['rfExTime'] * 1e-3  # ms
        # tVector = np.linspace(rfExTime / 2 + deadTime + 0.5 / bw, rfExTime / 2 + deadTime + acqTime - 0.5 / bw, nPoints)
        # fVector = np.linspace(-bw / 2, bw / 2, nPoints)
        # spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal))))
        # spectrum = np.reshape(spectrum, -1)
        #
        # # Get max and FHWM
        # spectrum = np.abs(spectrum)
        # maxValue = np.max(spectrum)
        # maxIndex = np.argmax(spectrum)
        # spectrumA = np.abs(spectrum[0:maxIndex] - maxValue)
        # spectrumB = np.abs(spectrum[maxIndex:nPoints] - maxValue)
        # indexA = np.argmin(spectrumA)
        # indexB = np.argmin(spectrumB) + maxIndex
        # freqA = fVector[indexA]
        # freqB = fVector[indexB]
        # noise = np.abs(self.dataTime[1])
        # noiserms = np.sqrt(2) * np.std(noise.real) *1000
        # self.mapVals['RMS noise'] = noiserms
        # self.mapVals['sampledPoint'] = noiserms  # for sweep method
        # spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal))))
        # fitedLarmor = self.mapVals['larmorFreq'] + fVector[np.argmax(np.abs(spectrum))] * 1e-3
        #
        # johnson = np.sqrt(2 * 50 * 293 * bw * 1e3 * 1.38e-23) * 10 ** (50 / 20) * 1e6  # uV
        # print('Larmor frequency: %1.5f MHz' % fitedLarmor)
        # print('Expected by Johnson: %0.5f uV' % johnson)
        # print('rms noise: %0.5f uV' % noiserms)
        # self.saveRawData()
        #
        # window = pg.LayoutWidget()
        #
        # FIDsignalPlotWidget = SpectrumPlot(xData=tVector,
        #                                    yData=[np.abs(signal), np.real(signal), np.imag(signal)],
        #                                    legend=['abs', 'real', 'imag'],
        #                                    xLabel='Time (ms)',
        #                                    yLabel='Signal amplitude (mV)',
        #                                    title='Signal vs time')
        # window.addWidget(FIDsignalPlotWidget, col=0, row=0)
        #
        # FIDspectrumPlotWidget = SpectrumPlot(xData=fVector,
        #                                      yData=[spectrum],
        #                                      legend=[''],
        #                                      xLabel='Frequency (kHz)',
        #                                      yLabel='Spectrum amplitude (a.u.)',
        #                                      title='Spectrum')
        # window.addWidget(FIDspectrumPlotWidget, col=1, row=0)
        #
        # NOISEtimePlotWidget = SpectrumPlot(xData=self.dataTime[0],
        #                                    yData=[np.abs(self.dataTime[1]), np.real(self.dataTime[1]),
        #                                           np.imag(self.dataTime[1])],
        #                                    legend=['abs', 'real', 'imag'],
        #                                    xLabel='Time (ms)',
        #                                    yLabel='Signal amplitude (mV)',
        #                                    title='Noise vs time, rms noise: %1.3f mV' % noiserms)
        # window.addWidget(NOISEtimePlotWidget, col=0, row=1)
        #
        # NOISEfreqPlotWidget = SpectrumPlot(xData=self.dataSpec[0],
        #                                    yData=[np.abs(self.dataSpec[1])],
        #                                    legend=[''],
        #                                    xLabel='Frequency (kHz)',
        #                                    yLabel='Mag FFT (a.u.)',
        #                                    title='Noise spectrum')
        # window.addWidget(NOISEfreqPlotWidget, col=1, row=1)
        #
        # self.out = [window]
        # return (self.out)