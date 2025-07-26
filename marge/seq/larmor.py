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
import marge.seq.mriBlankSeq as blankSeq
import marge.configs.hw_config as hw
import marge.configs.units as units


class Larmor(blankSeq.MRIBLANKSEQ):
    """
    Runs the Larmor sequence, which uses a spin echo technique to estimate the Larmor frequency.
    """
    def __init__(self):
        super(Larmor, self).__init__()
        # Input the parameters
        self.dF = None
        self.bw = None
        self.demo = None
        self.expt = None
        self.larmorFreq = None
        self.repetitionTime = None
        self.rfReTime = None
        self.rfReFA = None
        self.rfExFA = None
        self.rfExTime = None
        self.nScans = None
        self.addParameter(key='seqName', string='LarmorInfo', val='Larmor')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.066, units=units.MHz, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90.0, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180.0, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF', units=units.us)
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, field='RF', units=units.us)
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ', units=units.ms)
        self.addParameter(key='bw', string='Bandwidth (kHz)', val=50, field='RF', units=units.kHz)
        self.addParameter(key='dF', string='Frequency resolution (Hz)', val=100, field='RF')
        self.addParameter(key='shimming', string='Shimming', val=[-12.5, -12.5, 7.5], field='OTH', units=units.sh)

    def sequenceInfo(self):
        
        print("Larmor")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs a single spin echo to find larmor\n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        return (repetitionTime * nScans / 60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        """
        This method sets up the sequence parameters, including the RF pulse amplitudes, acquisition times,
        and echo times. It initializes the experiment, runs the scan, and processes the acquired data. The
        Larmor frequency is determined by the spin echo response.

        Args:
            plotSeq (int, optional): Flag to indicate if sequence plotting is required. Defaults to 0 (no plotting).
            demo (bool, optional): Flag to indicate if the sequence should run in demo mode (using simulated data). Defaults to False.
            standalone (bool, optional): Flag to indicate if the sequence is run in standalone mode (with plotting). Defaults to False.

        Returns:
            bool: True if the sequence ran successfully, False otherwise.

        Notes:
            - The sequence assumes that hardware settings are configured correctly (e.g., B1 efficiency, bandwidth).
            - In demo mode, actual hardware is not used, and simulated data is generated for testing purposes.
        """
        init_gpa = False  # Starts the gpa
        self.demo = demo

        # Set the refocusing time in to twice the excitation time
        if self.rfReTime == 0:
            self.rfReTime = 2 * self.rfExTime

        # Calculate the excitation amplitude
        rf_ex_amp = self.rfExFA * np.pi / 180 / (self.rfExTime * 1e6 * hw.b1Efficiency)
        rf_re_amp = self.rfReFA * np.pi / 180 / (self.rfReTime * 1e6 * hw.b1Efficiency)

        # Calculate acq_time and echo_time
        n_points = int(self.bw / self.dF)
        acq_time = 1 / self.dF  # s
        echo_time = 2 * acq_time  # s
        self.mapVals['nPoints'] = n_points
        self.mapVals['acqTime'] = acq_time
        self.mapVals['echoTime'] = echo_time

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

            # Refocusing pulse
            t0 = t_ex + echo_time / 2 - hw.blkTime - self.rfReTime / 2
            self.rfRecPulse(t0, self.rfReTime, rf_re_amp, np.pi / 2)

            # Rx gate
            t0 = t_ex + echo_time - acq_time / 2
            self.rxGateSync(t0, acq_time)
            rd_points += n_points

            self.endSequence(t_ex + self.repetitionTime)

            return rd_points

        # Time parameters to us
        self.rfExTime *= 1e6
        self.rfReTime *= 1e6
        self.repetitionTime *= 1e6
        acq_time *= 1e6
        echo_time *= 1e6

        # Initialize the experiment
        self.bw = n_points / acq_time  # MHz
        sampling_period = 1 / self.bw  # us
        self.mapVals['samplingPeriod'] = sampling_period
        if not self.demo:
            self.expt = ex.Experiment(lo_freq=self.larmorFreq * 1e-6,
                                      rx_t=sampling_period,
                                      init_gpa=init_gpa,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                      )
            sampling_period = self.expt.getSamplingRate()
        self.bw = 1 / sampling_period  # MHz
        acq_time = n_points / self.bw  # us
        self.mapVals['bw_true'] = self.bw * 1e6

        # Create the sequence and add instructions to the experiment
        acq_points = createSequence()
        if self.floDict2Exp(demo=self.demo):
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
                    rxd, msgs = self.expt.run()
                else:
                    rxd = {'rx0': np.random.randn((acq_points + 2 * hw.addRdPoints) * hw.oversamplingFactor) +
                                  1j * np.random.randn((acq_points + 2 * hw.addRdPoints) * hw.oversamplingFactor)}
                data_over = np.concatenate((data_over, rxd['rx0']), axis=0)
                print("Acquired points = %i" % np.size([rxd['rx0']]))
                print("Expected points = %i" % ((acq_points + 2 * hw.addRdPoints) * hw.oversamplingFactor))
                print("Scan %i ready!" % (scan + 1))
        elif plotSeq and standalone:
            self.sequencePlot(standalone=standalone)

        # Close the experiment
        if not self.demo:
            self.expt.__del__()

        # Process data to be plotted
        if not plotSeq:
            data_full = self.decimate(data_over, self.nScans, option='Normal')
            self.mapVals['data_full'] = data_full
            data = np.average(np.reshape(data_full, (self.nScans, -1)), axis=0)
            self.mapVals['data'] = data

            # Data to sweep sequence
            self.mapVals['sampledPoint'] = data[int(n_points / 2)]

        return True

    def sequenceAnalysis(self, mode=None):
        """
        Analyzes the acquired data from the Larmor sequence to determine the Larmor frequency
        and compute the corresponding time-domain signal and frequency spectrum.

        This method processes the acquired data by generating time and frequency vectors,
        performing a Fourier transform to obtain the signal spectrum, and determining the
        central frequency. It updates the Larmor frequency and provides the results in both
        time and frequency domains. The data is then optionally plotted, and the results are
        saved for further analysis.

        Args:
            mode (str, optional): The mode of execution. If set to 'Standalone', the results are plotted
                                   in a standalone manner. Defaults to None.

        Returns:
            list: A list containing the time-domain signal and frequency spectrum for visualization.

        Notes:
            - The Larmor frequency is recalculated based on the signal's central frequency from the spectrum.
            - The time-domain signal and frequency spectrum are both included in the output layout for visualization.
            - The results are saved as raw data and can be accessed later.
            - If the mode is not 'Standalone', the Larmor frequency is updated in all sequences in the sequence list.
        """
        self.mode = mode
        # Load data
        signal = self.mapVals['data']
        acq_time = self.mapVals['acqTime'] * 1e3  # ms
        n_points = self.mapVals['nPoints']

        # Generate time and frequency vectors and calcualte the signal spectrum
        tVector = np.linspace(-acq_time / 2, acq_time / 2, n_points)
        fVector = np.linspace(-self.bw / 2, self.bw / 2, n_points) * 1e3  # kHz
        spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal)))

        # Get the central frequency
        idf = np.argmax(np.abs(spectrum))
        fCentral = fVector[idf] * 1e-3  # MHz
        hw.larmorFreq = self.mapVals['larmorFreq'] + fCentral
        print('Larmor frequency: %1.5f MHz' % hw.larmorFreq)
        self.mapVals['fCentral'] = fCentral
        self.mapVals['larmorFreq0'] = hw.larmorFreq
        self.mapVals['signalVStime'] = [tVector, signal]
        self.mapVals['spectrum'] = [fVector, spectrum]

        if mode != 'Standalone':
            for sequence in self.sequence_list.values():
                if 'larmorFreq' in sequence.mapVals:
                    sequence.mapVals['larmorFreq'] = hw.larmorFreq

        # Add time signal to the layout
        result1 = {'widget': 'curve',
                   'xData': tVector,
                   'yData': [np.abs(signal), np.real(signal), np.imag(signal)],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Echo',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 0,
                   'col': 0}

        # Add frequency spectrum to the layout
        result2 = {'widget': 'curve',
                   'xData': fVector,
                   'yData': [np.abs(spectrum)],
                   'xLabel': 'Frequency (kHz)',
                   'yLabel': 'Spectrum amplitude (a.u.)',
                   'title': 'Spectrum',
                   'legend': [''],
                   'row': 1,
                   'col': 0}

        # create self.out to run in iterative mode
        self.output = [result1, result2]

        # save data once self.output is created
        self.saveRawData()

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        self.mapVals['larmorFreq'] = hw.larmorFreq

        return self.output


if __name__ == '__main__':
    seq = Larmor()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
