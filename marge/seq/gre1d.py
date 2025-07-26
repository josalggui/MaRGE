"""
@author: José Miguel Algarín Guisado
@modifield: T. Guallart Naval, february 28th 2022
MRILAB @ I3M
"""

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
import marge.controller.experiment_gui as ex
import numpy as np
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import marge.configs.hw_config as hw
import marge.configs.units as units


class GRE1D(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(GRE1D, self).__init__()
        # Input the parameters
        self.n_rd = None
        self.nScans = None
        self.expt = None
        self.repetitionTime = None
        self.rxTime = None
        self.axis = None
        self.rdPreemphasis = None
        self.rfReTime = None
        self.echoTime = None
        self.shimming = None
        self.rdDephTime = None
        self.fov_1d = None
        self.nPoints = None
        self.rdGradTime = None
        self.acqTime = None
        self.rfExTime = None
        self.rfExFA = None
        self.demo = None
        self.addParameter(key='seqName', string='GRE1DInfo', val='GRE1D')
        self.addParameter(key='toMaRGE', val=False)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90.0, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF', units=units.us)
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ', units=units.ms)
        self.addParameter(key='echoTime', string='Echo time (ms)', val=10.0, field='SEQ', units=units.ms)
        self.addParameter(key='nPoints', string='nPoints', val=60, field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=1.0, field='SEQ', units=units.ms)
        self.addParameter(key='rxTime', string='Acquisition window (ms)', val=1.0, field='SEQ', units=units.ms)
        self.addParameter(key='fov_1d', string='FOV (cm)', val=15.0, units=units.cm, field='IM')
        self.addParameter(key='axis', string='Axis', val=0, field='IM', tip="0=x, 1=y, 2=z")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=2.0, units=units.ms, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='shimming', string='Shimming', val=[-0.0, -0.0, 0.0], field='OTH', units=units.sh)
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')

    def sequenceInfo(self):
        
        print("GRE 1D")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs a 1d gradient echo\n")
        

    def sequenceTime(self):
        n_scans = self.mapVals['nScans']
        repetition_time = self.mapVals['repetitionTime'] * 1e-3
        return repetition_time * n_scans / 60  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        # Miscellaneous
        init_gpa = False  # Starts the gpa
        self.demo = demo

        # Calculate the excitation amplitude
        rf_ex_amp = self.rfExFA * np.pi / 180 / (self.rfExTime * 1e6 * hw.b1Efficiency)

        # Readout gradient time
        if self.rdGradTime < self.acqTime:
            self.rdGradTime = self.acqTime
            self.mapVals['rdGradTime'] = self.rdGradTime * 1

        # Max gradient amplitude
        rd_grad_amplitude = self.nPoints / (hw.gammaB * self.fov_1d * self.acqTime)
        self.mapVals['rd_grad_amplitude'] = rd_grad_amplitude

        # Readout dephasing amplitude
        rd_deph_amplitude = - 0.5 * rd_grad_amplitude * (hw.grad_rise_time + self.rdGradTime) / (
                hw.grad_rise_time + self.rdDephTime)
        self.mapVals['rd_deph_amplitude'] = rd_deph_amplitude

        def createSequence():
            rd_points = 0

            # Initialize time
            t0 = 20
            t_ex = 20e3

            # Shimming
            self.iniSequence(t0, self.shimming)

            # Excitation pulse
            t0 = t_ex - hw.blkTime - self.rfExTime / 2
            self.rfRecPulse(t0, self.rfExTime, rf_ex_amp, 0)

            # Dephasing readout
            t0 = t_ex + self.rfExTime / 2 - hw.gradDelay
            self.gradTrap(t0, grad_rise_time, self.rdDephTime, rd_deph_amplitude * self.rdPreemphasis, hw.grad_steps,
                          self.axis, self.shimming)

            # Readout gradient
            t0 = t_ex + self.echoTime - self.rdGradTime / 2 - grad_rise_time - hw.gradDelay
            self.gradTrap(t0, grad_rise_time, self.rdGradTime, rd_grad_amplitude, hw.grad_steps, self.axis,
                          self.shimming)

            # Rx gate
            t0 = t_ex + self.echoTime - self.rxTime / 2
            self.rxGateSync(t0, self.rxTime)
            rd_points += self.n_rd

            self.endSequence(t_ex + self.repetitionTime)

            return rd_points

        # Time parameters to us
        self.rfExTime *= 1e6
        self.repetitionTime *= 1e6
        self.echoTime *= 1e6
        self.acqTime *= 1e6
        self.rxTime *= 1e6
        self.rdDephTime *= 1e6
        self.rdGradTime *= 1e6
        grad_rise_time = hw.grad_rise_time * 1e6

        # Bandwidth
        bw = self.nPoints / self.acqTime  # MHz
        sampling_period = 1 / bw  # us

        # Acquisition windows
        if self.rxTime < self.acqTime:
            self.rxTime = self.acqTime
        self.n_rd = int(self.rxTime * bw)
        self.rxTime = self.n_rd / bw  # us

        # Initialize the experiment
        if not self.demo:
            self.expt = ex.Experiment(lo_freq=hw.larmorFreq, rx_t=sampling_period, init_gpa=init_gpa,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
            sampling_period = self.expt.getSamplingRate()  # us
            bw = 1 / sampling_period  # MHz
        self.acqTime = self.nPoints / bw  # us
        self.rxTime = self.n_rd / bw  # us
        self.mapVals['samplingPeriod'] = sampling_period * 1e-6  # s
        self.mapVals['bw'] = bw * 1e6  # Hz
        aa = createSequence()

        # Save instructions into MaRCoS if not a demo
        if not self.demo:
            if self.floDict2Exp():
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("ERROR: sequence waveforms out of hardware bounds")
                return False

        # Run scans
        data_ov = []
        for scan in range(self.nScans):
            if not plotSeq:
                if not self.demo:
                    rxd, msgs = self.expt.run()
                    print(msgs)
                else:
                    rxd = {}
                    rxd['rx0'] = np.random.randn((aa + 2 * hw.addRdPoints) * hw.oversamplingFactor) + \
                                 1j * np.random.randn((aa + 2 * hw.addRdPoints) * hw.oversamplingFactor)
                data_ov = np.concatenate((data_ov, rxd['rx0']), axis=0)

        # Delete experiment
        if not self.demo:
            self.expt.__del__()

        # Process data
        if not plotSeq:
            self.mapVals['data_ov'] = data_ov
            data_full = self.decimate(data_ov, self.nScans)
            self.mapVals['data_full'] = data_full
            data = np.average(np.reshape(data_full, (self.nScans, -1)), axis=0)
            self.mapVals['data'] = data
            self.mapVals['sampledPoint'] = data[int(self.n_rd / 2)]  # To be used by sweep class

        return True

    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        # Get images
        s0 = self.mapVals['data']
        s1 = s0[int(self.n_rd / 2 - self.nPoints / 2):int(self.n_rd / 2 + self.nPoints / 2)]
        img = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(s1)))
        t_vector = np.linspace(-self.rxTime / 2, self.rxTime / 2, self.n_rd) * 1e-3  # ms
        x_vector = np.linspace(-self.fov_1d / 2, self.fov_1d / 2, self.nPoints) * 1e2  # cm

        # Plot image magnitude
        result1 = {'widget': 'curve',
                   'xData': t_vector,
                   'yData': [np.abs(s0), np.real(s0), np.imag(s0)],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Amplitude (mV)',
                   'title': 'Signal vs time',
                   'legend': ['Magnitude', 'Real part', 'Imaginary part'],
                   'row': 0,
                   'col': 0}

        # Plot image phase
        result2 = {'widget': 'curve',
                   'xData': x_vector,
                   'yData': [np.abs(img)],
                   'xLabel': 'FOV (cm)',
                   'yLabel': 'Amplitude (a.u.)',
                   'title': 'Image',
                   'legend': [''],
                   'row': 1,
                   'col': 0}

        # create self.out to run in iterative mode
        self.output = [result1, result2]

        # Save results
        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()

        return self.output


if __name__ == '__main__':
    seq = GRE1D()
    seq.sequenceAtributes()
    seq.sequenceRun(demo=True)
    seq.sequenceAnalysis(mode='Standalone')
