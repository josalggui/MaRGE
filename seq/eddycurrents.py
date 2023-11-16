# import experiment as ex
# import numpy as np
# import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
# import scipy.signal as sig
# import configs.hw_config as hw
# from scipy.optimize import curve_fit
# from plotview.spectrumplot import SpectrumPlot
#
# class EDDYCURRENTS(blankSeq.MRIBLANKSEQ):
#     def __init__(self):
#         super(EDDYCURRENTS, self).__init__()
#         # Input the parameters
#         self.addParameter(key='seqName', string='EDDYCURRENTSinfo', val='EDDYCURRENTS')
#         self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
#         self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=8.365, field='RF')
#         self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=1.0, field='RF')
#         self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
#         self.addParameter(key='rfExPhase', string='RF phase', val=60.0, field='RF')
#         self.addParameter(key='deadTime', string='RF dead time (us)', val=80.0, field='RF')
#         self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
#         self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
#
#         self.addParameter(key='sampleLength', string='Sample length (cm)', val=4.0, field='SEQ')
#         self.addParameter(key='nReadout', string='Number of points', val=600, field='SEQ')
#         self.addParameter(key='tAdq', string='Acquisition time (ms)', val=3.0, field='SEQ')
#         self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000, field='SEQ')
#         self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0, 0, 0], field='SEQ')
#         self.addParameter(key='gAxis', string='G axis', val=2, field='SEQ')
#         self.addParameter(key='gSteps', string='G steps', val=20, field='SEQ')
#         self.addParameter(key='gRiseTime', string='G rise time (us)', val=150, field='SEQ')
#         self.addParameter(key='gAmp', string='G amplitude (mT/m)', val=5.0, field='SEQ')
#         self.addParameter(key='gDuration', string='G duration (ms)', val=1, field='SEQ')
#         self.addParameter(key='tDelay', string='Delay G-RF (ms)', val=1, field='SEQ')
#         self.addParameter(key='addRdPoints', string='addRdPoints', val=10, field='SEQ')
#
#     def sequenceInfo(self):
#         print(" ")
#         print("EDDYCURRENTS")
#         print("Author: Dr. J.M. Algarín")
#         print("Contact: josalggui@i3m.upv.es")
#         print("mriLab @ i3M, CSIC, Spain")
#
#     def sequenceTime(self):
#         return(4*self.mapVals['repetitionTime']*1e-3/60)  # minutes, scanTime
#
#     def sequenceRun(self, plotSeq=0):
#         init_gpa = False  # Starts the gpa
#
#         # Create input parameters
#         nScans = self.mapVals['nScans']
#         larmorFreq = self.mapVals['larmorFreq'] # MHz
#         rfExAmp = self.mapVals['rfExAmp']
#         rfExTime = self.mapVals['rfExTime'] # us
#         rfExPhase = self.mapVals['rfExPhase']
#         deadTime = self.mapVals['deadTime'] # us
#         nReadout = self.mapVals['nReadout']
#         tAdq = self.mapVals['tAdq'] * 1e3  # us
#         TR = self.mapVals['repetitionTime'] * 1e3  # us
#         shimming = np.array(self.mapVals['shimming']) * 1e-4
#         gAxis = self.mapVals['gAxis']
#         gSteps = self.mapVals['gSteps']
#         gRiseTime = self.mapVals['gRiseTime'] # us
#         gAmp = self.mapVals['gAmp']*1e-3 #T/m
#         gDuration = self.mapVals['gDuration']*1e3  # us
#         delayGtoRF = self.mapVals['tDelay']*1e3  # us
#         addRdPoints = self.mapVals['addRdPoints']
#         txChannel = self.mapVals['txChannel']
#         rxChannel = self.mapVals['rxChannel']
#
#
#         # CONSTANTES
#         tini = 20
#         gAmpMax = 40
#         shimming = np.array(shimming) * 1e-4
#
#         # CONDICIONES PARAMETROS INICIALES.
#         if gAmp > gAmpMax:
#             gAmp = gAmpMax
#
#         rfExPhase = rfExPhase * np.pi / 180
#         rfExAmp = rfExAmp * np.exp(1j * rfExPhase)
#
#         def createSequence():
#             # Shimming
#             self.iniSequence(tini, shimming)
#
#             for scan in range(nScans):
#                 # Gradient pulse 0
#                 tGup0 = tini + TR
#                 self.gradTrap(tGup0, gRiseTime, gDuration, 0, gSteps, gAxis, shimming)
#                 tRF0 = tGup0 + gDuration + 2*gRiseTime + delayGtoRF - hw.blkTime
#                 self.rfRecPulse(tRF0, rfExTime, rfExAmp, rfExPhase, txChannel=txChannel)
#                 # Rx gate
#                 tRx0 = tRF0 + hw.blkTime + rfExTime + deadTime - addRdPoints / BWreal
#                 self.rxGate(tRx0, acqTimeReal + addRdPoints / BWreal, rxChannel=rxChannel)
#                 # # **********************************************
#
#                 # Gradient pulse +
#                 tGup1 = tini + 2*TR
#                 self.gradTrap(tGup1, gRiseTime, gDuration, +gAmp, gSteps, gAxis, shimming)
#                 # RF pulse
#                 tRF1 = tGup1 + gDuration + 2 * gRiseTime + delayGtoRF - hw.blkTime
#                 self.rfRecPulse(tRF1, rfExTime, rfExAmp, rfExPhase, txChannel=txChannel)
#                 # Rx gate
#                 tRx1 = tRF1 + hw.blkTime + rfExTime + deadTime - addRdPoints / BWreal
#                 self.rxGate(tRx1, acqTimeReal + addRdPoints / BWreal, rxChannel=rxChannel)
#                 # **********************************************
#
#                 # Gradient pulse -
#                 tGup2 = tini + 3*TR
#                 self.gradTrap(tGup2, gRiseTime, gDuration, -gAmp, gSteps, gAxis, shimming)
#                 # RF pulse
#                 tRF2 = tGup2 + gDuration + 2 * gRiseTime + delayGtoRF - hw.blkTime
#                 self.rfRecPulse(tRF2, rfExTime, rfExAmp, rfExPhase, txChannel=txChannel)
#                 # Rx gate
#                 tRx2 = tRF2 + hw.blkTime + rfExTime + deadTime - addRdPoints / BWreal
#                 self.rxGate(tRx2, acqTimeReal + addRdPoints / BWreal, rxChannel=rxChannel)
#
#             self.endSequence(tini + 4*TR)
#
#         # Initialize the experiment
#         BWoriginal = nReadout / tAdq # MHz
#         self.mapVals['BWoriginal'] = BWoriginal
#         samplingPeriodOriginal = 1 / BWoriginal # us
#         BWov = BWoriginal * hw.oversamplingFactor  # MHz
#         samplingPeriodov = 1 / BWov
#         self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriodov, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
#         samplingPeriodReal = self.expt.get_rx_ts()[0]
#         BWreal = 1 / samplingPeriodReal / hw.oversamplingFactor  # MHz
#         acqTimeReal = nReadout / BWreal  # us
#         createSequence()
#
#         if plotSeq == 0:
#             # Run the experiment and get data
#             nScans = self.mapVals['nScans']
#             rxd, msgs = self.expt.run()
#             rxd['rx%i' % rxChannel] = np.real(rxd['rx%i'%rxChannel])-1j*np.imag(rxd['rx%i'%rxChannel])
#             overData = rxd['rx%i'%rxChannel]*13.788
#             dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
#             self.mapVals['overData'] = overData
#             self.mapVals['dataFull'] = dataFull
#             data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
#             self.mapVals['data'] = data
#             signal = np.reshape(data, (3, nReadout+self.mapVals['addRdPoints']))
#             signalFiltered = np.delete(signal, np.s_[0:addRdPoints], axis=1)
#             self.mapVals['signals'] = signalFiltered
#             self.expt.__del__()
#
#     def sequenceAnalysis(self, obj=''):
#         gammabar=hw.gammaB
#         gamma = 2*np.pi*gammabar
#         sampleLength=self.mapVals['sampleLength']*1e-3
#         BW = self.mapVals['BWoriginal']*1e3
#         FreqVector = np.linspace(-BW/2, BW/2, self.mapVals['nReadout'])
#         ReadoutVector = np.linspace(0, self.mapVals['tAdq'], self.mapVals['nReadout'])
#         signals = self.mapVals['signals']
#         signal0 = signals[0, :]
#         signalup = signals[1, :]
#         signaldown = signals[2, :]
#         spectrum0 = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal0))))
#         spectrumup = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signalup))))
#         spectrumdown = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signaldown))))
#
#         signalupNorm=abs(signalup)*abs(signal0[0])/abs(signalup[0])
#         signaldownNorm = abs(signaldown) * abs(signal0[0]) / abs(signaldown[0])
#
#         ratio_plus= abs(signalupNorm)/abs(signal0)
#         ratio_minus = abs(signaldownNorm)/abs(signal0)
#
#         def objective(x, a):
#             return (1 - a * x ** 2)
#
#         # curve fit a+
#         popt, _ = curve_fit(objective, ReadoutVector, ratio_plus)
#         # summarize the parameter values
#         a_plus = popt
#
#         # curve fit a-
#         popt, _ = curve_fit(objective, ReadoutVector, ratio_minus)
#         # summarize the parameter values
#         a_minus = popt
#
#         ResidualGradient = np.sqrt(12*(a_plus+a_minus)/(gamma*sampleLength*gamma*sampleLength))
#
#         self.saveRawData()
#
#         # Add time signal to the layout
#         signalPlotWidget = SpectrumPlot(xData=ReadoutVector,
#                                         yData=[np.abs(signal0), np.abs(signalup), np.abs(signaldown)],
#                                         legend=['G=0', 'G-', 'G+'],
#                                         xLabel='Time (ms)',
#                                         yLabel='Signal amplitude (mV)',
#                                         title = 'Signal vs time, Residual Gradient: %1.3f mT/m' % ResidualGradient*1e3)
#
#         # Add frequency spectrum to the layout
#         spectrumPlotWidget = SpectrumPlot(xData=FreqVector,
#                                         yData=[np.abs(spectrum0), np.abs(spectrumup), np.abs(spectrumdown)],
#                                         legend=['G=0', 'G-', 'G+'],
#                                         xLabel='Frequency (kHz)',
#                                         yLabel='Signal amplitude (mV)',
#                                         title='Signal vs freq')
#         return([signalPlotWidget, spectrumPlotWidget])


# **********************************************************************************************************************
# # METHOD TO ASSESS MINIMUM DELAY TO EDDY CURRENTS ATTENUATION FOR A GRADIENT ESTABLISHED IN A ZTE EMBODIMENT
import experiment as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw

class EDDYCURRENTS(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(EDDYCURRENTS, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='EDDYCURRENTSinfo', val='EDDYCURRENTS')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=8.365, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=1.0, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
        self.addParameter(key='rfExPhase', string='RF phase', val=60.0, field='RF')
        self.addParameter(key='deadTime', string='RF dead time (us)', val=80.0, field='RF')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='nReadout', string='Number of points', val=600, field='SEQ')
        self.addParameter(key='tAdq', string='Acquisition time (ms)', val=3.0, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0, 0, 0], field='SEQ')
        self.addParameter(key='gAxis', string='G axis', val=2, field='SEQ')
        self.addParameter(key='gSteps', string='G steps', val=20, field='SEQ')
        self.addParameter(key='gRiseTime', string='G rise time (us)', val=150, field='SEQ')
        self.addParameter(key='gAmp', string='G amplitude (mT/m)', val=5.0, field='SEQ')
        self.addParameter(key='tDelayInit', string='Delay G-RF initial (ms)', val=1, field='SEQ')
        self.addParameter(key='tDelayMiddle', string='Delay G-RF middle (ms)', val=1, field='SEQ')
        self.addParameter(key='tDelayFinal', string='Delay G-RF final (ms)', val=1, field='SEQ')
        self.addParameter(key='addRdPoints', string='addRdPoints', val=10, field='SEQ')

    def sequenceInfo(self):
        print(" ")
        print("EDDYCURRENTS")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")

    def sequenceTime(self):
        return(4*self.mapVals['repetitionTime']*1e-3/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False  # Starts the gpa

        # Create input parameters
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq'] # MHz
        rfExAmp = self.mapVals['rfExAmp']
        rfExTime = self.mapVals['rfExTime'] # us
        rfExPhase = self.mapVals['rfExPhase']
        deadTime = self.mapVals['deadTime'] # us
        nReadout = self.mapVals['nReadout']
        tAdq = self.mapVals['tAdq'] * 1e3  # us
        TR = self.mapVals['repetitionTime'] * 1e3  # us
        gAxis = self.mapVals['gAxis']
        gSteps = self.mapVals['gSteps']
        gRiseTime = self.mapVals['gRiseTime'] # us
        gAmp = self.mapVals['gAmp']*1e-3 #T/m
        delayGtoRF_0 = self.mapVals['tDelayInit']*1e3  # us
        delayGtoRF_M = self.mapVals['tDelayMiddle'] * 1e3  # us
        delayGtoRF_F = self.mapVals['tDelayFinal'] * 1e3  # us
        addRdPoints = self.mapVals['addRdPoints']
        txChannel = self.mapVals['txChannel']
        rxChannel = self.mapVals['rxChannel']
        shimming = self.mapVals['shimming']

        # CONSTANTES
        tini = 20
        gAmpMax = 40
        shimming = np.array(shimming) * 1e-4

        # CONDICIONES PARAMETROS INICIALES.
        if gAmp > gAmpMax:
            gAmp = gAmpMax

        rfExPhase = rfExPhase * np.pi / 180
        rfExAmp = rfExAmp * np.exp(1j * rfExPhase)

        # CONDICIONES PARAMETROS INICIALES.

        rfExPhase = rfExPhase * np.pi / 180
        rfExAmp = rfExAmp * np.exp(1j * rfExPhase)

        def createSequence():

            if TR < delayGtoRF_F + rfExTime + deadTime + acqTimeReal:
                print("TR so short")

            for scan in range(nScans):
                # Gradient pulse
                tGup0 = tini
                self.gradTrap(tGup0, gRiseTime, TR, gAmp, gSteps, gAxis, shimming)

                # Tx gate 1
                tRF1 = tGup0 + gRiseTime + delayGtoRF_0 - hw.blkTime
                self.rfRecPulse(tRF1, rfExTime, rfExAmp, rfExPhase, txChannel=txChannel)
                # Rx gate 1
                tRx1 = tRF1 + hw.blkTime + rfExTime + deadTime - addRdPoints / BWreal
                self.rxGate(tRx1, acqTimeReal + addRdPoints / BWreal, rxChannel=rxChannel)

                # Tx gate 2
                tRF2 = tGup0 + gRiseTime + delayGtoRF_M - hw.blkTime
                self.rfRecPulse(tRF2, rfExTime, rfExAmp, rfExPhase, txChannel=txChannel)
                # Rx gate 2
                tRx2 = tRF2 + hw.blkTime + rfExTime + deadTime - addRdPoints / BWreal
                self.rxGate(tRx2, acqTimeReal + addRdPoints / BWreal, rxChannel=rxChannel)

                # Tx gate 3
                tRF3 = tGup0 + gRiseTime + delayGtoRF_F - hw.blkTime
                self.rfRecPulse(tRF3, rfExTime, rfExAmp, rfExPhase, txChannel=txChannel)
                # Rx gate 3
                tRx3 = tRF3 + hw.blkTime + rfExTime + deadTime - addRdPoints / BWreal
                self.rxGate(tRx3, acqTimeReal + addRdPoints / BWreal, rxChannel=rxChannel)

            self.endSequence(tini + (TR + 2*gRiseTime))

        # Initialize the experiment
        BWoriginal = nReadout / tAdq # MHz
        self.mapVals['BWoriginal'] = BWoriginal
        samplingPeriodOriginal = 1 / BWoriginal # us
        BWov = BWoriginal * hw.oversamplingFactor  # MHz
        samplingPeriodov = 1 / BWov
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriodov, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriodReal = self.expt.get_rx_ts()[0]
        BWreal = 1 / samplingPeriodReal / hw.oversamplingFactor  # MHz
        acqTimeReal = nReadout / BWreal  # us
        createSequence()
        if self.floDict2Exp():
            print("\nSequence waveforms loaded successfully")
            pass
        else:
            print("\nERROR: sequence waveforms out of hardware bounds")
            return False

        if plotSeq == 0:
            # Run the experiment and get data
            nScans = self.mapVals['nScans']
            rxd, msgs = self.expt.run()
            rxd['rx%i' % rxChannel] = np.real(rxd['rx%i'%rxChannel])-1j*np.imag(rxd['rx%i'%rxChannel])
            overData = rxd['rx%i'%rxChannel]*hw.adcFactor
            dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['overData'] = overData
            self.mapVals['dataFull'] = dataFull
            data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
            self.mapVals['data'] = data
            signal = np.reshape(data, (3, nReadout+self.mapVals['addRdPoints']))
            signalFiltered = np.delete(signal, np.s_[0:addRdPoints], axis=1)
            self.mapVals['signals'] = signalFiltered
            self.expt.__del__()

        return True

    def sequenceAnalysis(self, obj=''):
        BW = self.mapVals['BWoriginal']*1e3
        FreqVector = np.linspace(-BW/2, BW/2, self.mapVals['nReadout'])
        ReadoutVector = np.linspace(0, self.mapVals['tAdq'], self.mapVals['nReadout'])
        signals = self.mapVals['signals']
        signal0 = signals[0, :]
        signalup = signals[1, :]
        signaldown = signals[2, :]
        spectrum0 = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal0))))
        spectrumup = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signalup))))
        spectrumdown = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signaldown))))


        delayGtoRF_0 = self.mapVals['tDelayInit']
        delayGtoRF_M = self.mapVals['tDelayMiddle']
        delayGtoRF_F =self.mapVals['tDelayFinal']

        # Add time signal to the layout
        result1 = {'widget': 'curve',
                   'xData': ReadoutVector,
                   'yData': [np.abs(signal0), np.abs(signalup), np.abs(signaldown)],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Signal vs time, Residual Gradient: %1.3f mT/m' % ResidualGradient*1e3,
                   'legend': ['G=0', 'G-', 'G+'],
                   'row': 0,
                   'col': 0}

        # Add frequency spectrum to the layout
        result2 = {'widget': 'curve',
                   'xData': FreqVector,
                   'yData': [np.abs(spectrum0), np.abs(spectrumup), np.abs(spectrumdown)],
                   'xLabel': 'Frequency (kHz)',
                   'yLabel': 'Signal amplitude (a.u.)',
                   'title': 'Signal vs freq',
                   'legend': ['G=0', 'G-', 'G+'],
                   'row': 1,
                   'col': 0}

        self.output = [result1, result2]

        self.saveRawData()

        return self.output
