"""
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia, Spain
@date: 13 mon June 2022
@email: josalggui@i3m.upv.es
"""

import controller.experiment_gui as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw
import configs.units as units


class TSEPRE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(TSEPRE, self).__init__()
        # Input the parameters
        self.rfExPhase = None
        self.demo = None
        self.axis = None
        self.rdPreemphasis = None
        self.rdDephTime = None
        self.fov = None
        self.rdGradTime = None
        self.nScans = None
        self.expt = None
        self.nPoints = None
        self.repetitionTime = None
        self.acqTime = None
        self.echoSpacing = None
        self.etl = None
        self.rfReTime = None
        self.rfExTime = None
        self.rfReFA = None
        self.rfExFA = None
        self.shimming = None
        self.addParameter(key='seqName', string='TSE_prescan_info', val='TSE_prescan')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=35.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=70.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, field='SEQ', units=units.ms)
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=500., field='SEQ', units=units.ms)
        self.addParameter(key='nPoints', string='nPoints', val=60, field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='SEQ', units=units.ms)
        self.addParameter(key='fov', string='FOV (cm)', val=15.0, units=units.cm, field='IM')
        self.addParameter(key='axis', string='Axis', val=0, field='IM', tip="0=x, 1=y, 2=z")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=5.0, units=units.ms, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='etl', string='Echo train length', val=2, field='SEQ')
        self.addParameter(key='shimming', string='Shimming', val=[0, 0, 0], field='OTH', units=units.sh)
        self.addParameter(key='rfExPhase', string='RF excitation phase (º)', val=0.0, field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')

    def sequenceInfo(self):
        print(" ")
        print("Turbo Spin Echo for eddy currents calibration")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        return (repetitionTime * nScans / 60)  # minutes, scanTime

    def sequenceRun(self, plotSeq, demo=False):
        init_gpa = False  # Starts the gpa
        self.demo = demo

        # Get rf pulse amplitude
        rf_ex_amp = self.rfExFA / (self.rfExTime * 1e6 * hw.b1Efficiency) * np.pi / 180
        rf_re_amp = self.rfReFA / (self.rfReTime * 1e6 * hw.b1Efficiency) * np.pi / 180
        self.mapVals['rfExAmp'] = rf_ex_amp
        self.mapVals['rfReAmp'] = rf_re_amp

        # Readout gradient time
        if self.rdGradTime < self.acqTime:
            self.rdGradTime = self.acqTime
            self.mapVals['rdGradTime'] = self.rdGradTime * 1

        # Max gradient amplitude
        rd_grad_amplitude = self.nPoints / (hw.gammaB * self.fov * self.acqTime)
        self.mapVals['rd_grad_amplitude'] = rd_grad_amplitude

        # Readout dephasing amplitude
        rd_deph_amplitude = 0.5 * rd_grad_amplitude * (hw.grad_rise_time + self.rdGradTime) / (
                    hw.grad_rise_time + self.rdDephTime)
        self.mapVals['rd_deph_amplitude'] = rd_deph_amplitude

        def createSequence():
            acq_points = 0

            # Set shimming
            self.iniSequence(20, self.shimming)

            # Initialize time
            t_ex = 20e3

            # Excitation pulse
            t0 = t_ex - hw.blkTime - self.rfExTime / 2
            self.rfRecPulse(t0, self.rfExTime, rf_ex_amp, self.rfExPhase * np.pi / 180)

            # Dephasing readout
            t0 = t_ex + self.rfExTime / 2 - hw.gradDelay
            self.gradTrap(t0, grad_rise_time, self.rdDephTime, rd_deph_amplitude * self.rdPreemphasis, hw.grad_steps,
                          self.axis, self.shimming)

            for echo_index in range(self.etl):
                t_echo = t_ex + (echo_index + 1) * self.echoSpacing

                # Refocusing pulse
                t0 = t_echo - self.echoSpacing / 2 - self.rfReTime / 2 - hw.blkTime
                self.rfRecPulse(t0, self.rfReTime, rf_re_amp, np.pi / 2)

                # Readout gradient
                t0 = t_echo - self.rdGradTime / 2 - grad_rise_time - hw.gradDelay
                self.gradTrap(t0, grad_rise_time, self.rdGradTime, rd_grad_amplitude, hw.grad_steps, self.axis,
                              self.shimming)

                # Rx gating
                t0 = t_echo - self.acqTime / 2
                self.rxGateSync(t0, self.acqTime)
                acq_points += self.nPoints + 2 * hw.addRdPoints

            # End sequence
            self.endSequence(self.repetitionTime)

            return acq_points

        # Time variables in us
        self.rfExTime *= 1e6
        self.rfReTime *= 1e6
        self.repetitionTime *= 1e6
        self.echoSpacing *= 1e6
        self.acqTime *= 1e6
        self.rdDephTime *= 1e6
        self.rdGradTime *= 1e6
        grad_rise_time = hw.grad_rise_time * 1e6

        # Bandwidth and sampling rate
        bw = self.nPoints / self.acqTime # MHz
        sampling_period = 1 / bw  # us

        # Create sequence
        if not self.demo:
            self.expt = ex.Experiment(lo_freq=hw.larmorFreq, rx_t=sampling_period, init_gpa=init_gpa,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
            sampling_period = self.expt.getSamplingRate()  # us
            bw = 1 / sampling_period  # MHz
        self.acqTime = self.nPoints / bw  # us
        self.mapVals['samplingPeriod'] = sampling_period * 1e-6  # s
        self.mapVals['bw'] = bw * 1e6  # Hz
        aa = createSequence()

        # Save instructions into MaRCoS if not a demo
        if not self.demo:
            if self.floDict2Exp():
                print("\nSequence waveforms loaded successfully")
                pass
            else:
                print("\nERROR: sequence waveforms out of hardware bounds")
                return False

        # Run scans
        data_ov = []
        for scan in range(self.nScans):
            if not plotSeq:
                if not self.demo:
                    rxd, msgs = self.expt.run()
                    print(msgs)
                    rxd['rx0'] = rxd['rx0'] * hw.adcFactor
                else:
                    rxd = {}
                    rxd['rx0'] = np.random.randn(aa * hw.oversamplingFactor) + 1j * np.random.randn(
                        aa * hw.oversamplingFactor)
                data_ov = np.concatenate((data_ov, rxd['rx0']), axis=0)

        if not self.demo:
            self.expt.__del__()

        if not plotSeq:
            self.mapVals['data_ov'] = data_ov
            data_full = self.decimate(data_ov, self.etl*self.nScans)
            self.mapVals['data_full'] = data_full
            data = np.average(np.reshape(data_full, (self.nScans, -1)), axis=0)
            self.mapVals['data'] = data
            data = np.reshape(data, (self.etl, -1))
            self.mapVals['sampledPoint'] = data[0, int(self.nPoints / 2)]  # To be used by sweep class

        return True

    def sequenceAnalysis(self, obj=''):
        # Get images
        signal = np.reshape(self.mapVals['data'], (self.etl, -1))
        img1 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal[0, :])))
        img2 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal[1, :])))
        x_vector = np.linspace(-self.fov/2, self.fov/2, self.nPoints)

        # Plot image magnitude
        result1 = {'widget': 'curve',
                   'xData': x_vector,
                   'yData': [np.abs(img1), np.abs(img2)],
                   'xLabel': 'FOV (cm)',
                   'yLabel': 'Amplitude (a.u.)',
                   'title': 'Image amplitude',
                   'legend': ['First echo', 'Second echo'],
                   'row': 0,
                   'col': 0}

        # Plot image phase
        result2 = {'widget': 'curve',
                   'xData': x_vector,
                   'yData': [np.unwrap(np.angle(img1)), np.unwrap(np.angle(img2))],
                   'xLabel': 'FOV (cm)',
                   'yLabel': 'Image phase (rads)',
                   'title': 'Image phase',
                   'legend': ['First echo', 'Second echo'],
                   'row': 0,
                   'col': 1}

        # create self.out to run in iterative mode
        self.output = [result1, result2]

        self.saveRawData()

        return self.output
