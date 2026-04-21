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
from scipy.signal import firwin, freqz, lfilter
from scipy.optimize import curve_fit

class EDDYCURRENTS2(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(EDDYCURRENTS2, self).__init__()
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
        self.addParameter(key='seqName', string='EDDYCURRENTS2info', val='EDDYCURRENTS2')
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
        self.addParameter(key='DownsampleFilter', string='Downsample Filter', val=5, field='SEQ')
        self.addParameter(key='preemphasisOn', string='Bool Preemphasis', val=0, field='OTH')
        self.addParameter(key='Npointswaveform', string='Samples preemphasis', val=50, field='OTH')
        self.addParameter(key='AmpEddys', string='Amplitude Eddys (uT)', val=[4.0, 1.0], field='OTH')
        self.addParameter(key='TauEddys', string='Tau Eddys (us)', val=[1000, 200], field='OTH')
        self.addParameter(key='sample_pos', string='Sample Position', val='center', tip='center, +x, -x, +y, -y, +z, -z', field='OTH')


    def sequenceInfo(self):
        print("EDDYCURRENTS2")
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

        sample_pos = ['center', '+x', '-x', '+y', '-y', '+z', '-z']
        if not self.sample_pos in sample_pos:
            print('ERROR: unexpected sample position.')
            return False

        # Define delay array
        delays = np.linspace(self.tDelayMin, self.tDelayMax, self.nDelays)*1e6 #us

        # Fix gSteps and gRiseTime
        if self.gSteps == 0:
            self.gSteps = hw.grad_steps
        if self.gRiseTime == 0:
            self.gRiseTime = hw.grad_rise_time

        preemphasisOn = self.mapVals['preemphasisOn']
        if preemphasisOn == 1:
            Npointswaveform = self.mapVals['Npointswaveform']
            AmpEddys = self.mapVals['AmpEddys']
            TauEddys = self.mapVals['TauEddys']


        if self.gAxis == 'x':
            self.gAxis = 0
            currentX = self.mapVals['gAmplitude'] * hw.effX
            print('Current Gx: %0.5f A' % currentX)
        elif self.gAxis == 'y':
            self.gAxis = 1
            currentY = self.mapVals['gAmplitude'] * hw.effY
            print('Current Gy: %0.5f A' % currentY)
        elif self.gAxis == 'z':
            self.gAxis = 2
            currentZ = self.mapVals['gAmplitude'] * hw.effZ
            print('Current Gz: %0.5f A' % currentZ)

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
                if preemphasisOn == 0:
                    self.gradTrap(t0, self.gRiseTime, self.gDuration, self.gAmplitude, self.gSteps, self.gAxis, self.shimming)

                if preemphasisOn==1:
                    self.gradTrapPreemphasis(t0, self.gRiseTime, self.gDuration, self.gAmplitude, Npointswaveform, self.gAxis, self.shimming, AmpEddys, TauEddys)

                self.ttl(t0 + self.gDuration + 2 * self.gRiseTime, 10, channel=1)

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
                if preemphasisOn == 0:
                    self.gradTrap(t0, self.gRiseTime, self.gDuration, -self.gAmplitude, self.gSteps, self.gAxis, self.shimming)
                if preemphasisOn == 1:
                    self.gradTrapPreemphasis(t0, self.gRiseTime, self.gDuration, -self.gAmplitude, Npointswaveform, self.gAxis, self.shimming, AmpEddys, TauEddys)

                self.ttl(t0+ self.gDuration + 2 * self.gRiseTime, 10, channel=1)

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

            self.endSequence(self.repetitionTime*3*self.nDelays)

            return rd_points

        # Time parameters to us
        self.gRiseTime *= 1e6
        self.gDuration *= 1e6
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
        overData = []
        if not plotSeq:
            for scan in range(self.nScans):
                print("Scan %i running..." % (scan + 1))
                if not self.demo:
                    rxd, msgs = self.expt.run()
                    rxd['rx0'] = rxd['rx0']  # mV
                    print(scan, "/", self.nScans, "Sequence finished")
                    # Get data
                    overData = np.concatenate((overData, rxd['rx0']), axis=0)

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
            data_full = self.decimate(data_over, 3*self.nScans*self.nDelays, option='PETRA')
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
        fidsmap = (self.mapVals['data'])
        complexfidsmap = (self.mapVals['data'])

        # Get spectrum amplitudes
        gradZero = np.squeeze(spectrums[:, 0, :])
        gradPositive = np.squeeze(spectrums[:, 1, :])
        gradNegative = np.squeeze(spectrums[:, 2, :])
        gZero = np.max(np.abs(gradZero), axis=1)
        gPositive = np.max(np.abs(gradPositive), axis=1)
        gNegative = np.max(np.abs(gradNegative), axis=1)
        t_vector = np.linspace(self.tDelayMin, self.tDelayMax, self.nDelays) * 1e3  # ms
        tFID_vector = np.linspace(0, self.acqTime, self.mapVals['nReadouts']) * 1e3  # ms
        self.mapVals['tFID_vector'] = tFID_vector

        ratioFID = np.zeros((3, self.mapVals['nDelays']))
        Be = np.zeros((1, self.mapVals['nDelays'], self.mapVals['nReadouts']))
        for ii in range(self.mapVals['nDelays']):
            fidsdelayii = np.squeeze(fidsmap[ii, :, :])
            complexfidsdelayii = np.squeeze(complexfidsmap[ii, :, :])
            ratioFID[0, ii] = np.std(fidsdelayii[0, :] - fidsdelayii[0, :])
            ratioFID[1, ii] = np.std(fidsdelayii[1, :] - fidsdelayii[0, :])
            ratioFID[2, ii] = np.std(fidsdelayii[2, :] - fidsdelayii[0, :])
            Be[0, ii, :] = (1 / (2 * 2 * np.pi * hw.gammaB)) * np.gradient(
                (np.angle(complexfidsdelayii[1, :]) - np.angle(complexfidsdelayii[2, :])),
                tFID_vector * 1e-3) * 1e6  # uT
        self.mapVals['ratioFID'] = ratioFID



        # Plot 5
        result5 = {'widget': 'curve',
                   'xData': tFID_vector*1e-6,
                   'yData': [np.squeeze(np.abs(complexfidsmap[0, 0, :])), np.squeeze(np.abs(complexfidsmap[0, 1, :])),
                             np.squeeze(np.abs(complexfidsmap[0, 2, :]))],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'FID strength (a.u.)',
                   'title': 'Abs value FIDs for minimum delay',
                   'legend': ['No gradient', 'Positive gradient', 'Negative Gradient'],
                   'row': 0,
                   'col': 0
                   }

        # Plot 9
        result9 = {'widget': 'curve',
                   'xData': tFID_vector*1e-6,
                   'yData': [np.squeeze(np.unwrap(np.angle(complexfidsmap[0, 0, :]))),
                             np.squeeze(np.unwrap(np.angle(complexfidsmap[0, 1, :]))),
                             np.squeeze(np.unwrap(np.angle(complexfidsmap[0, 2, :]))),
                             np.squeeze(np.unwrap(np.angle(complexfidsmap[0, 1, :]))) - np.squeeze(np.unwrap(np.angle(complexfidsmap[0, 2, :])))],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'FID strength (a.u.)',
                   'title': 'Phase of FIDs for minimum delay',
                   'legend': ['No gradient', 'G+', 'G-','G+ - G-'],
                   'row': 1,
                   'col': 0
                   }

        # Plot 6
        result6 = {'widget': 'curve',
                   'xData': tFID_vector*1e-6,
                   'yData': [np.squeeze(np.abs(complexfidsmap[self.nDelays - 1, 0, :])),
                             np.squeeze(np.abs(complexfidsmap[self.nDelays - 1, 1, :])),
                             np.squeeze(np.abs(complexfidsmap[self.nDelays - 1, 2, :]))],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'FID strength (a.u.)',
                   'title': 'Abs value FIDs for maximum delay',
                   'legend': ['No gradient', 'Positive gradient', 'Negative Gradient'],
                   'row': 0,
                   'col': 1
                   }


        # --- Tiempo original ---
        timeFID = np.linspace(self.deadTime * 1e-6,
                              self.acqTime * 1e-3,
                              self.mapVals['nReadouts']) * 1e-3

        timeFIDPlot = np.linspace(self.deadTime * 1e-6,
                                  self.acqTime * 1e-3,
                                  self.mapVals['nReadouts'] - 1) * 1e-3

        self.mapVals['timeFID'] = timeFID

        # --- Obtención de FID ---
        FIDdelayCte = np.squeeze(complexfidsmap[0, :, :])
        FIDplus = np.squeeze(np.unwrap(np.angle(FIDdelayCte[1, :])))
        FIDminus = np.squeeze(np.unwrap(np.angle(FIDdelayCte[2, :])))


        # datos en crudo
        difference = (FIDplus - FIDminus) * (1 / (4 * np.pi * 42.577e6))
        BeddyRaw = np.diff(difference) / np.diff(timeFID)
        BeddyFitted = np.polyval(
            np.polyder(np.polyfit(timeFID, difference, 5)),
            timeFID
        )
        BeddyFittedCutted = BeddyFitted[:-1]

        self.mapVals['BeddyRaw'] = BeddyRaw
        self.mapVals['BeddyFitted'] = BeddyFitted


        # Plot 7
        result7 = {'widget': 'curve',
                   'xData': tFID_vector*1e-6,
                   'yData': [np.angle(FIDdelayCte[0, :]),np.angle(FIDdelayCte[1, :]),np.angle(FIDdelayCte[2, :])],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'FID strength (a.u.)',
                   'title': 'Phase FIDs for minimum delay',
                   'legend': ['No gradient', 'Positive gradient', 'Negative Gradient'],
                   'row': 1,
                   'col': 1
                   }

        result8 = {
            'widget': 'curve',
            'xData': timeFIDPlot*1e3,
            'yData': [np.real(BeddyRaw), np.real(BeddyFittedCutted)],
            'xLabel': 'Time (ms)',
            'yLabel': 'B_eddy(t)',
            'title': 'B_eddy(t) (minimum delay) - RAW',
            'legend': ['Raw estimation', 'Fitted estimation'],
            'row': 0,
            'col': 2
        }

        # con filtro
        factor = self.mapVals['DownsampleFilter']
        phaseDiffRaw = FIDplus - FIDminus
        N = len(phaseDiffRaw)
        Ns = N // factor
        phaseDiffFiltered = phaseDiffRaw[:Ns * factor].reshape(Ns, factor).mean(axis=1)
        timeFIDFiltered = timeFID[:Ns * factor].reshape(Ns, factor).mean(axis=1)
        differenceFiltered = phaseDiffFiltered * (1 / (4 * np.pi * 42.577e6))
        BeddyFiltered = np.diff(differenceFiltered) / np.diff(timeFIDFiltered)
        polyFilt = np.polyfit(timeFIDFiltered, differenceFiltered, 5)
        BeddyFittedFiltered = np.polyval(np.polyder(polyFilt), timeFIDFiltered)
        BeddyFittedFilteredCutted = BeddyFittedFiltered[:-1]
        self.mapVals['BeddyFiltered'] = BeddyFiltered
        self.mapVals['BeddyFilteredFit'] = BeddyFittedFiltered


        result4 = {
            'widget': 'curve',
            'xData': timeFIDFiltered[:-1]*1e3,
            'yData': [np.real(BeddyFiltered),
                      np.real(BeddyFittedFilteredCutted)],
            'xLabel': 'Time (ms)',
            'yLabel': 'B_eddy(t) filtered',
            'title': 'B_eddy(t) (minimum delay) - FILTERED',
            'legend': ['Filtered estimation', 'Filtered fitted estimation'],
            'row': 1,
            'col': 2
        }

        self.output = [result5, result6, result7, result4, result8, result9]

        # save data once self.output is created
        self.saveRawData()

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        return self.output



if __name__ == '__main__':
    seq = EDDYCURRENTS2()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    # seq.sequencePlot(standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
