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

class EDDYCURRENTS(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(EDDYCURRENTS, self).__init__()
        # Input the parameters
        self.output = None
        self.larmorFreq = None
        self.nReadouts = None
        self.nScans = None
        self.gAmplitude = None
        self.acqTime = None
        self.deadTime = None
        self.rfExAmp = None
        self.rfExTime = None
        self.gAxis = None
        self.gSteps = None
        self.gDuration = None
        self.gRiseTime = None
        self.repetitionTime = None
        self.nDelays = None
        self.tDelayMax = None
        self.tDelayMin = None
        self.shimming = None
        self.addParameter(key='seqName', string='EDDYCURRENTSinfo', val='EDDYCURRENTS')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.0, units=units.MHz, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, units=units.us, field='RF')
        self.addParameter(key='deadTime', string='RF dead time (us)', val=400.0, units=units.us, field='RF')
        self.addParameter(key='nReadouts', string='Number of points', val=600, field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=3.0, units=units.ms, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=20, units=units.ms, field='SEQ')
        self.addParameter(key='shimming', string='Shimming', val=[0, 0, 0], units=units.sh, field='SEQ')
        self.addParameter(key='gAxis', string='G axis', val='x', tip="'x', 'y' or 'z'", field='SEQ')
        self.addParameter(key='gSteps', string='G steps', val=0, field='SEQ',
                          tip="0 if you want to use the default value in the hw_config.py file")
        self.addParameter(key='gRiseTime', string='G rise time (us)', val=0, units=units.us, field='SEQ',
                          tip="0 if you want to use the default value in the hw_config.py file")
        self.addParameter(key='gAmplitude', string='G amplitude (mT/m)', val=5.0, units=units.mTm, field='SEQ')
        self.addParameter(key='gDuration', string='G duration (ms)', val=10, units=units.ms, field='SEQ')
        self.addParameter(key='tDelayMin', string='Min delay G2RF (ms)', val=1, units=units.ms, field='SEQ')
        self.addParameter(key='tDelayMax', string='Max delay G2RF (ms)', val=10, units=units.ms, field='SEQ')
        self.addParameter(key='nDelays', string='Number of delays', val=2, field='SEQ')

    def sequenceInfo(self):
        print("EDDYCURRENTS")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Sequence to characterize the eddy currents.\n")

    def sequenceTime(self):
        n_delays = self.mapVals['nDelays']
        n_scans = self.mapVals['nScans']
        tr = self.mapVals['repetitionTime'] # ms
        return tr*1e-3*n_scans*n_delays*3/60  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        init_gpa = False
        self.demo = demo

        # Define delay array
        delays = np.linspace(self.tDelayMin, self.tDelayMax, self.nDelays)*1e6 #us

        # Fix gSteps and gRiseTime
        if self.gSteps == 0:
            self.gSteps = hw.grad_steps
        if self.gRiseTime == 0:
            self.gRiseTime = hw.grad_rise_time*1e-6

        # Fix axis
        if self.gAxis == 'x':
            self.gAxis = 0
        elif self.gAxis == 'y':
            self.gAxis = 1
        elif self.gAxis == 'z':
            self.gAxis = 2

        def createSequence():
            rd_points = 0
            t_ini = 20

            # Shimming
            self.iniSequence(t_ini, self.shimming)

            for delay in delays:
                # Gradient pulse 0
                t0 = t_ini
                t0 += 20
                self.gradTrap(t0, self.gRiseTime, self.gDuration, 0, self.gSteps, self.gAxis, self.shimming)
                # Tx RF pulse
                t0 = t0 + self.gDuration + 2*self.gRiseTime + delay - hw.blkTime
                self.rfRecPulse(t0, rfTime=self.rfExTime, rfAmplitude=self.rfExAmp, rfPhase=0)
                # Rx gate
                t0 = t0 + hw.blkTime + self.rfExTime + self.deadTime
                self.rxGateSync(t0, self.acqTime)
                rd_points += self.nReadouts + 2 * hw.addRdPoints
                # **********************************************

                # Gradient pulse +
                t0 = t_ini + self.repetitionTime
                t0 += 20
                self.gradTrap(t0, self.gRiseTime, self.gDuration, self.gAmplitude, self.gSteps, self.gAxis, self.shimming)
                # Tx RF pulse
                t0 = t0 + self.gDuration + 2 * self.gRiseTime + delay - hw.blkTime
                self.rfRecPulse(t0, rfTime=self.rfExTime, rfAmplitude=self.rfExAmp, rfPhase=0)
                # Rx gate
                t0 = t0 + hw.blkTime + self.rfExTime + self.deadTime
                self.rxGateSync(t0, self.acqTime)
                rd_points += self.nReadouts + 2 * hw.addRdPoints
                # **********************************************

                # Gradient pulse -
                t0 = t_ini + 2 * self.repetitionTime
                t0 += 20
                self.gradTrap(t0, self.gRiseTime, self.gDuration, -self.gAmplitude, self.gSteps, self.gAxis, self.shimming)
                # Tx RF pulse
                t0 = t0 + self.gDuration + 2 * self.gRiseTime + delay - hw.blkTime
                self.rfRecPulse(t0, rfTime=self.rfExTime, rfAmplitude=self.rfExAmp, rfPhase=0)
                # Rx gate
                t0 = t0 + hw.blkTime + self.rfExTime + self.deadTime
                self.rxGateSync(t0, self.acqTime)
                rd_points += self.nReadouts + 2 * hw.addRdPoints
                # **********************************************

                # Update t_ini
                t_ini += 3 * self.repetitionTime

            self.endSequence(self.repetitionTime*3*self.nDelays*self.nScans)

            return rd_points

        # Time parameters to us
        self.gRiseTime *= 1e6
        self.gDuration *= 1e6
        self.gRiseTime *= 1e6
        self.rfExTime *= 1e6
        self.acqTime *= 1e6
        self.deadTime *= 1e6
        self.repetitionTime *= 1e6

        # Initialize the experiment
        bw = self.nReadouts / self.acqTime  # MHz
        sampling_period = 1 / bw  # us
        self.mapVals['samplingPeriod'] = sampling_period
        if not self.demo:
            self.expt = ex.Experiment(lo_freq=self.larmorFreq * 1e-6,
                                      rx_t=sampling_period,
                                      init_gpa=False,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                      )
            sampling_period = self.expt.getSamplingRate()
        bw = 1 / sampling_period  # MHz
        self.acqTime = self.nReadouts / bw  # us
        self.mapVals['bw'] = bw * 1e6

        # Create the sequence and add instructions to the experiment
        points_to_measure = createSequence()
        if not self.demo:
            if self.floDict2Exp():
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("ERROR: sequence waveforms out of hardware bounds")
                return False

        # Run the experiment
        data_over = []  # To save oversampled data
        if not plotSeq:
            for scan in range(self.nScans):
                print("Scan %i running..." % (scan + 1))
                if not self.demo:
                    points_measured = 0
                    while points_measured != (points_to_measure * hw.oversamplingFactor):
                        rxd, msgs = self.expt.run()
                        points_measured = np.size(rxd['rx0'])
                        print("Acquired points = %i" % points_measured)
                        print("Expected points = %i" % (points_to_measure * hw.oversamplingFactor))
                else:
                    rxd = {'rx0': np.random.randn(points_to_measure * hw.oversamplingFactor) +
                                  1j * np.random.randn(
                        points_to_measure * hw.oversamplingFactor)}
                    points_measured = np.size(rxd['rx0'])
                    print("Acquired points = %i" % points_measured)
                    print("Expected points = %i" % (points_to_measure * hw.oversamplingFactor))
                data_over = np.concatenate((data_over, rxd['rx0']), axis=0)
                print("Scan %i ready!" % (scan + 1))

            self.mapVals['data_oversampled'] = data_over
        elif plotSeq and standalone:
            self.sequencePlot(standalone=standalone)
            return True

        # Close the experiment
        if not self.demo:
            self.expt.__del__()

        # Process data to be plotted
        if not plotSeq:
            data_full = self.decimate(data_over, 3*self.nScans*self.nDelays, option='Normal')
            self.mapVals['data_full'] = data_full
            data = np.average(np.reshape(data_full, (self.nScans, self.nDelays, 3, -1)), axis=0)
            self.mapVals['data'] = data

            # Get fft
            spectrums = np.zeros_like(data, dtype=complex)
            for delay_idx in range(self.nDelays):
                for rx_idx in range(3):
                    data_prov = data[delay_idx, rx_idx, :]
                    spectrums[delay_idx, rx_idx, :] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data_prov)))
            self.mapVals['spectrums'] = spectrums

            # Data to sweep sequence
            self.mapVals['sampledPoint'] = data[0, 0, 0]

        return True


    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        # Load data
        spectrums = self.mapVals['spectrums']
        
        # Get spectrum amplitudes
        gradZero = np.squeeze(spectrums[:, 0, :])
        gradPositive = np.squeeze(spectrums[:, 1, :])
        gradNegative = np.squeeze(spectrums[:, 2, :])
        gZero = np.max(np.abs(gradZero), axis=1)
        gPositive = np.max(np.abs(gradPositive), axis=1)
        gNegative = np.max(np.abs(gradNegative), axis=1)
        t_vector = np.linspace(self.tDelayMin, self.tDelayMax, self.nDelays)*1e3 # ms
        
        
        # Spectrum maps
        result1 = {'widget': 'image',
                   'data': np.abs(np.expand_dims(np.transpose(spectrums, axes=(1, 0, 2))[0, :, :], axis=0)),
                   'xLabel': 'Frequency (a.u.)',
                   'yLabel': 'Delay (a.u.)',
                   'title': 'No gradient',
                   'row': 0,
                   'col': 0}

        # Plot 2
        result2 = {'widget': 'image',
                   'data': np.abs(np.expand_dims(np.transpose(spectrums, axes=(1, 0, 2))[1, :, :], axis=0)),
                   'xLabel': 'Frequency (a.u.)',
                   'yLabel': 'Delay (a.u.)',
                   'title': 'Positive gradient',
                   'row': 0,
                   'col': 1}

        # Plot 3
        result3 = {'widget': 'image',
                   'data': np.abs(np.expand_dims(np.transpose(spectrums, axes=(1, 0, 2))[2, :, :], axis=0)),
                   'xLabel': 'Frequency (a.u.)',
                   'yLabel': 'Delay (a.u.)',
                   'title': 'Negative gradient',
                   'row': 1,
                   'col': 0}
        
        # Plot 4
        result4 = {'widget': 'curve',
                'xData': t_vector,
                'yData': [gZero, gPositive, gNegative],
                'xLabel': 'Delay G2RF Time (ms)',
                'yLabel': 'Spectrum amplitude (a.u.)',
                'title': 'Echo',
                'legend': ['No gradient', 'Positive gradient', 'Negative Gradient'],
                'row': 1,
                'col': 1
                }
        
        self.output = [result4]

        # save data once self.output is created
        self.saveRawData()

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        return self.output
        


# **********************************************************************************************************************
# METHOD TO ASSESS MINIMUM DELAY TO EDDY CURRENTS ATTENUATION FOR A GRADIENT ESTABLISHED IN A ZTE EMBODIMENT
# import experiment as ex
# import numpy as np
# import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
# import scipy.signal as sig
# import configs.hw_config as hw
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
#         self.addParameter(key='nReadout', string='Number of points', val=600, field='SEQ')
#         self.addParameter(key='tAdq', string='Acquisition time (ms)', val=3.0, field='SEQ')
#         self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000, field='SEQ')
#         self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0, 0, 0], field='SEQ')
#         self.addParameter(key='gAxis', string='G axis', val=2, field='SEQ')
#         self.addParameter(key='gSteps', string='G steps', val=20, field='SEQ')
#         self.addParameter(key='gRiseTime', string='G rise time (us)', val=150, field='SEQ')
#         self.addParameter(key='gAmp', string='G amplitude (mT/m)', val=5.0, field='SEQ')
#         self.addParameter(key='tDelayInit', string='Delay G-RF initial (ms)', val=1, field='SEQ')
#         self.addParameter(key='tDelayMiddle', string='Delay G-RF middle (ms)', val=1, field='SEQ')
#         self.addParameter(key='tDelayFinal', string='Delay G-RF final (ms)', val=1, field='SEQ')
#         self.addParameter(key='addRdPoints', string='addRdPoints', val=10, field='SEQ')
#
#     def sequenceInfo(self):
#         
#         print("EDDYCURRENTS")
#         print("Author: Dr. J.M. Algarín")
#         print("Contact: josalggui@i3m.upv.es")
#         print("mriLab @ i3M, CSIC, Spain")
#
#     def sequenceTime(self):
#         return(4*self.mapVals['repetitionTime']*1e-3/60)  # minutes, scanTime
#
#     def sequenceRun(self, plotSeq=0, demo=False):
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
#         gAxis = self.mapVals['gAxis']
#         gSteps = self.mapVals['gSteps']
#         gRiseTime = self.mapVals['gRiseTime'] # us
#         gAmp = self.mapVals['gAmp']*1e-3 #T/m
#         delayGtoRF_0 = self.mapVals['tDelayInit']*1e3  # us
#         delayGtoRF_M = self.mapVals['tDelayMiddle'] * 1e3  # us
#         delayGtoRF_F = self.mapVals['tDelayFinal'] * 1e3  # us
#         addRdPoints = self.mapVals['addRdPoints']
#         txChannel = self.mapVals['txChannel']
#         rxChannel = self.mapVals['rxChannel']
#         shimming = self.mapVals['shimming']
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
#         # CONDICIONES PARAMETROS INICIALES.
#
#         rfExPhase = rfExPhase * np.pi / 180
#         rfExAmp = rfExAmp * np.exp(1j * rfExPhase)
#
#         def createSequence():
#
#             if TR < delayGtoRF_F + rfExTime + deadTime + acqTimeReal:
#                 print("TR so short")
#
#             for scan in range(nScans):
#                 # Gradient pulse
#                 tGup0 = tini
#                 self.gradTrap(tGup0, gRiseTime, TR, gAmp, gSteps, gAxis, shimming)
#
#                 # Tx gate 1
#                 tRF1 = tGup0 + gRiseTime + delayGtoRF_0 - hw.blkTime
#                 self.rfRecPulse(tRF1, rfExTime, rfExAmp, rfExPhase, txChannel=txChannel)
#                 # Rx gate 1
#                 tRx1 = tRF1 + hw.blkTime + rfExTime + deadTime - addRdPoints / BWreal
#                 self.rxGate(tRx1, acqTimeReal + addRdPoints / BWreal, rxChannel=rxChannel)
#
#                 # Tx gate 2
#                 tRF2 = tGup0 + gRiseTime + delayGtoRF_M - hw.blkTime
#                 self.rfRecPulse(tRF2, rfExTime, rfExAmp, rfExPhase, txChannel=txChannel)
#                 # Rx gate 2
#                 tRx2 = tRF2 + hw.blkTime + rfExTime + deadTime - addRdPoints / BWreal
#                 self.rxGate(tRx2, acqTimeReal + addRdPoints / BWreal, rxChannel=rxChannel)
#
#                 # Tx gate 3
#                 tRF3 = tGup0 + gRiseTime + delayGtoRF_F - hw.blkTime
#                 self.rfRecPulse(tRF3, rfExTime, rfExAmp, rfExPhase, txChannel=txChannel)
#                 # Rx gate 3
#                 tRx3 = tRF3 + hw.blkTime + rfExTime + deadTime - addRdPoints / BWreal
#                 self.rxGate(tRx3, acqTimeReal + addRdPoints / BWreal, rxChannel=rxChannel)
#
#             self.endSequence(tini + (TR + 2*gRiseTime))
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
#         if self.floDict2Exp():
#             print("Sequence waveforms loaded successfully")
#             pass
#         else:
#             print("ERROR: sequence waveforms out of hardware bounds")
#             return False
#
#         if plotSeq == 0:
#             # Run the experiment and get data
#             nScans = self.mapVals['nScans']
#             rxd, msgs = self.expt.run()
#             rxd['rx%i' % rxChannel] = np.real(rxd['rx%i'%rxChannel])-1j*np.imag(rxd['rx%i'%rxChannel])
#             overData = rxd['rx%i'%rxChannel]*hw.adcFactor
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
#         return True
#
#     def sequenceAnalysis(self, obj=''):
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
#
#         delayGtoRF_0 = self.mapVals['tDelayInit']
#         delayGtoRF_M = self.mapVals['tDelayMiddle']
#         delayGtoRF_F =self.mapVals['tDelayFinal']
#
#         # Add time signal to the layout
#         result1 = {'widget': 'curve',
#                    'xData': ReadoutVector,
#                    'yData': [np.abs(signal0), np.abs(signalup), np.abs(signaldown)],
#                    'xLabel': 'Time (ms)',
#                    'yLabel': 'Signal amplitude (mV)',
#                    'title': 'Signal vs time, Residual Gradient: %1.3f mT/m' % ResidualGradient*1e3,
#                    'legend': ['G=0', 'G-', 'G+'],
#                    'row': 0,
#                    'col': 0}
#
#         # Add frequency spectrum to the layout
#         result2 = {'widget': 'curve',
#                    'xData': FreqVector,
#                    'yData': [np.abs(spectrum0), np.abs(spectrumup), np.abs(spectrumdown)],
#                    'xLabel': 'Frequency (kHz)',
#                    'yLabel': 'Signal amplitude (a.u.)',
#                    'title': 'Signal vs freq',
#                    'legend': ['G=0', 'G-', 'G+'],
#                    'row': 1,
#                    'col': 0}
#
#         self.output = [result1, result2]
#
#         self.saveRawData()
#
#         return self.output


if __name__ == '__main__':
    seq = EDDYCURRENTS()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    # seq.sequencePlot(standalone=True)
    seq.sequenceAnalysis(mode='Standalone')