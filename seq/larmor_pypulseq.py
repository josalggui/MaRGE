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
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import configs.hw_config as hw
import configs.units as units
import controller.controller_device as device
from marga_pulseq.interpreter import PSInterpreter
import pypulseq as pp


class LarmorPyPulseq(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(LarmorPyPulseq, self).__init__()
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
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='pypulseq', string='PyPulseq', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.066, units=units.MHz, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90.0, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180.0, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF', units=units.us)
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, field='RF', units=units.us)
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=100., field='SEQ', units=units.ms)
        self.addParameter(key='bw', string='Bandwidth (kHz)', val=50, field='RF', units=units.kHz)
        self.addParameter(key='dF', string='Frequency resolution (Hz)', val=100, field='RF')
        self.addParameter(key='shimming', string='Shimming', val=[-12.5, -12.5, 7.5], field='OTH', units=units.sh)

    def sequenceInfo(self):
        
        print("Larmor")
        print("Author: PhD. J.M. Algarín")
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
            demo (bool, optional): Flag to indicate if the sequence should run in demo mode (using simulated data).Defaults to False.
            standalone (bool, optional): Flag to indicate if the sequence is run in standalone mode (with plotting). Defaults to False.

        Returns:
            bool: True if the sequence ran successfully, False otherwise.

        Notes:
            - The sequence assumes that hardware settings are configured correctly (e.g., B1 efficiency, bandwidth).
            - In demo mode, actual hardware is not used, and simulated data is generated for testing purposes.
        """

        self.demo = demo
        self.plotSeq = plotSeq
        self.standalone = standalone

        '''
        Step 1: Define the interpreter for FloSeq/PSInterpreter.
        The interpreter is responsible for converting the high-level pulse sequence description into low-level
        instructions for the scanner hardware.
        '''

        flo_interpreter = PSInterpreter(
            tx_warmup=hw.blkTime,  # Transmit chain warm-up time (us)
            rf_center=hw.larmorFreq * 1e6,  # Larmor frequency (Hz)
            rf_amp_max=hw.b1Efficiency / (2 * np.pi) * 1e6,  # Maximum RF amplitude (Hz)
            gx_max=hw.gFactor[0] * hw.gammaB,  # Maximum gradient amplitude for X (Hz/m)
            gy_max=hw.gFactor[1] * hw.gammaB,  # Maximum gradient amplitude for Y (Hz/m)
            gz_max=hw.gFactor[2] * hw.gammaB,  # Maximum gradient amplitude for Z (Hz/m)
            grad_max=np.max(hw.gFactor) * hw.gammaB,  # Maximum gradient amplitude (Hz/m)
            grad_t=hw.grad_raster_time * 1e6,  # Gradient raster time (us)
        )

        '''
        Step 2: Define system properties using PyPulseq (pp.Opts).
        These properties define the hardware capabilities of the MRI scanner, such as maximum gradient strengths,
        slew rates, and dead times. They are typically set based on the hardware configuration file (`hw_config`).
        '''

        system = pp.Opts(
            rf_dead_time=hw.blkTime * 1e-6,  # Dead time between RF pulses (s)
            max_grad=np.max(hw.gFactor) * 1e3,  # Maximum gradient strength (mT/m)
            grad_unit='mT/m',  # Units of gradient strength
            max_slew=hw.max_slew_rate,  # Maximum gradient slew rate (mT/m/ms)
            slew_unit='mT/m/ms',  # Units of gradient slew rate
            grad_raster_time=hw.grad_raster_time,  # Gradient raster time (s)
            rise_time=hw.grad_rise_time,  # Gradient rise time (s)
            rf_raster_time=1e-6,
            block_duration_raster=1e-6
        )

        '''
        Step 3: Perform any calculations required for the sequence.
        In this step, students can implement the necessary calculations, such as timing calculations, RF amplitudes, and
        gradient strengths, before defining the sequence blocks.
        '''

        # Set the refocusing time in to twice the excitation time
        if self.rfReTime == 0:
            self.rfReTime = 2 * self.rfExTime

        # Calculate acq_time and echo_time
        n_points = int(self.bw / self.dF)
        acq_time = 1 / self.dF  # s
        echo_time = 2 * acq_time  # s
        sampling_period = 1 / self.bw * 1e6  # us
        self.mapVals['nPoints'] = n_points
        self.mapVals['acqTime'] = acq_time
        self.mapVals['echoTime'] = echo_time

        '''
        Step 4: Define the experiment to get the true bandwidth
        In this step, student needs to get the real bandwidth used in the experiment. To get this bandwidth, an
        experiment must be defined and the sampling period should be obtained using get_
        '''

        # Define device arguments
        dev_kwargs = {
            "lo_freq": hw.larmorFreq,  # MHz
            "rx_t": sampling_period,  # us
            "print_infos": True,
            "assert_errors": True,
            "halt_and_reset": False,
            "fix_cic_scale": True,
            "set_cic_shift": False,  # needs to be true for open-source cores
            "flush_old_rx": False,
            "init_gpa": False,
            "gpa_fhdo_offset_time": 1 / 0.2 / 3.1,
            "auto_leds": True
        }

        # Define master arguments
        master_kwargs = {
            'mimo_master': True,
            'trig_output_time': 1e5,
            'slave_trig_latency': 6.079
        }

        if not self.demo:
            dev = device.Device(ip_address=hw.rp_ip_list[0], port=hw.rp_port, **(master_kwargs | dev_kwargs))
            sampling_period = dev.get_sampling_period()  # us
            self.bw = 1 / sampling_period  # MHz
            print("Acquisition bandwidth fixed to: %0.3f kHz" % (self.bw * 1e3))
            dev.__del__()
        self.mapVals['bw_MHz'] = self.bw
        self.mapVals['sampling_period_us'] = sampling_period

        '''
        Step 5: Define sequence blocks.
        In this step, you will define the building blocks of the MRI sequence, including the RF pulses, gradient pulses,
        and ADC blocks.
        '''

        # Create excitation rf event
        delay_rf_ex = self.repetitionTime-acq_time/2-echo_time-self.rfExTime/2-hw.blkTime*1e-6
        event_rf_ex = pp.make_block_pulse(flip_angle=self.rfExFA * np.pi / 180,
                                          duration=self.rfExTime,
                                          delay=delay_rf_ex,
                                          system=system,
                                          use="excitation",)

        # Create refocusing rf event
        delay_rf_re = echo_time/2-self.rfExTime/2-self.rfReTime/2-hw.blkTime*1e-6
        event_rf_re = pp.make_block_pulse(flip_angle=self.rfReFA * np.pi / 180,
                                          duration=self.rfReTime,
                                          delay=delay_rf_re,
                                          system=system,
                                          use="refocusing")

        # Create ADC event
        delay_adc = echo_time/2-self.rfReTime/2-acq_time/2
        event_adc = pp.make_adc(num_samples=n_points*hw.oversamplingFactor,
                                duration=acq_time,  # s
                                delay=delay_adc,  # s
                                system=system)

        '''
        Step 6: Define your initializeBatch according to your sequence.
        In this step, you will create the initializeBatch method to create dummy pulses that will be initialized for
        each new batch.
        '''

        def initializeBatch():
            # Instantiate pypulseq sequence object
            batch = pp.Sequence(system)
            n_rd_points = 0
            n_adc = 0

            return batch, n_rd_points, n_adc

        '''
        Step 7: Define your createBatches method.
        In this step you will populate the batches adding the blocks previously defined in step 4, and accounting for
        number of acquired points to check if a new batch is required.
        '''

        def createBatches():
            """
            Create batches for the full pulse sequence.

            Instructions:
            - This function creates the complete pulse sequence by iterating through repetitions.
            - Each iteration adds new blocks to the sequence, including the RF pulse, ADC block, and repetition delay.
            - If a batch exceeds the maximum number of readout points, a new batch is started.

            Returns:
                waveforms (dict): Contains the waveforms for each batch.
                n_rd_points_dict (dict): Dictionary of readout points per batch.
                n_adc (int): Total number of ADC acquisitions across all batches
            """
            batches = {}  # Dictionary to save batches PyPulseq sequences
            waveforms = {}  # Dictionary to store generated waveforms per each batch
            n_rd_points_dict = {}  # Dictionary to track readout points for each batch
            n_rd_points = 0  # To account for number of acquired rd points
            seq_idx = 0  # Sequence batch index
            n_adc = 0  # To account for number of adc windows
            batch_num = "batch_0"  # Initial batch name

            # Update to the next batch
            seq_idx += 1
            n_rd_points_dict[batch_num] = n_rd_points  # Save readout points count
            n_rd_points = 0
            batch_num = f"batch_{seq_idx}"
            batches[batch_num], n_rd_points, n_adc_0 = initializeBatch()  # Initialize new batch
            n_adc += n_adc_0
            print(f"Creating {batch_num}.seq...")

            # Add sequence blocks
            # batches[batch_num].add_block(delay_first)
            batches[batch_num].add_block(event_rf_ex)
            batches[batch_num].add_block(event_rf_re)
            batches[batch_num].add_block(event_adc)
            n_rd_points += n_points
            n_adc += 1

            # After final repetition, save and interpret the last batch
            batches[batch_num].write(batch_num + ".seq")
            waveforms[batch_num], param_dict = flo_interpreter.interpret(batch_num + ".seq")
            print(f"{batch_num}.seq ready!")
            print(f"{len(batches)} batches created. Sequence ready!")

            # Update the number of acquired ponits in the last batch
            n_rd_points_dict.pop('batch_0')
            n_rd_points_dict[batch_num] = n_rd_points

            return waveforms, n_rd_points_dict, n_adc

        '''
        Step 8: Run the batches
        This step will handle the different batches, run it and get the resulting data. This should not be modified.
        Oversampled data will be available in self.mapVals['data_over']
        Decimated data will be available in self.mapVals['data_decimated']
        The decimated data is shifted to account for CIC delay, so data is synchronized with real-time signal
        '''

        waveforms, n_readouts, n_adc = createBatches()
        return self.runBatches(waveforms=waveforms,
                               n_readouts=n_readouts,
                               n_adc=n_adc,
                               frequency=hw.larmorFreq,  # MHz
                               bandwidth=self.bw,  # MHz
                               decimate='Normal',
                               )

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
        signal = self.mapVals['data_decimated'][0]
        acq_time = self.mapVals['acqTime'] * 1e3  # ms
        n_points = self.mapVals['nPoints']

        # Generate time and frequency vectors and calculate the signal spectrum
        tVector = np.linspace(-acq_time / 2, acq_time / 2, n_points)
        fVector = np.linspace(-self.bw / 2, self.bw / 2, n_points) * 1e-3  # kHz
        spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal)))

        # Get the central frequency
        idf = np.argmax(np.abs(spectrum))
        fCentral = fVector[idf] * 1e-3  # MHz
        hw.larmorFreq = self.mapVals['larmorFreq'] + fCentral
        print('Larmor frequency: %1.5f MHz' % hw.larmorFreq)
        self.mapVals['larmorFreq'] = hw.larmorFreq
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

        return self.output


if __name__ == '__main__':
    seq = LarmorPyPulseq()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
