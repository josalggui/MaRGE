"""
@author: J.M. Algarín, february 03th 2022
MRILAB @ I3M
"""

import os
import sys
import time

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
import controller.controller_device as device
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import configs.hw_config as hw
import configs.units as units
from marga_pulseq.interpreter import PSInterpreter
import pypulseq as pp


class Noise(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(Noise, self).__init__()
        # Input the parameters
        self.repetitionTime = None
        self.rxChannel = None
        self.nPoints = None
        self.bw = None
        self.freqOffset = None
        self.addParameter(key='seqName', string='NoiseInfo', val='Noise')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='pypulseq', string='PyPulseq', val=True)
        self.addParameter(key='freqOffset', string='RF frequency offset (kHz)', val=0.0, units=units.kHz, field='RF')
        self.addParameter(key='nPoints', string='Number of points', val=2500, field='RF')
        self.addParameter(key='bw', string='Acquisition bandwidth (kHz)', val=50.0, units=units.kHz, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=500.0, field='RF', units=units.ms)

    def sequenceInfo(self):
        print("Noise")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Get a noise measurement\n")

    def sequenceTime(self):
        return 0  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
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

        # Fix units to MHz and us
        self.freqOffset *= 1e-6  # MHz
        self.bw *= 1e-6  # MHz
        acq_time = self.nPoints / self.bw * 1e-6  # s
        sampling_period = 1 / self.bw  # us
        self.mapVals['larmorFreq'] = hw.larmorFreq

        '''
        Step 4: Define the experiment to get the true bandwidth
        In this step, student needs to get the real bandwidth used in the experiment. To get this bandwidth, an
        experiment must be defined and the sampling period should be obtained using get_
        '''

        if not self.demo:
            # Define device arguments
            dev_kwargs = {
                "lo_freq": hw.larmorFreq + self.freqOffset,  # MHz
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

            dev = device.Device(ip_address=hw.rp_ip_list[0], port=hw.rp_port[0], **(master_kwargs | dev_kwargs))
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

        # Create ADC event
        delay_adc = self.repetitionTime - acq_time
        event_adc = pp.make_adc(num_samples=self.nPoints * hw.oversamplingFactor,
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
            batches[batch_num].add_block(event_adc)
            n_rd_points += self.nPoints
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
        self.mode = mode
        data = self.mapVals['data_decimated'][0]
        tVector = np.linspace(0, self.nPoints / self.bw, num=self.nPoints) * 1e-3  # ms
        noiserms = np.std(data)
        self.mapVals['RMS noise'] = noiserms
        self.mapVals['sampledPoint'] = noiserms  # for sweep method
        noiserms = noiserms * 1e3
        print('rms noise: %0.5f uV' % noiserms)
        bw = self.bw * 1e6  # Hz
        johnson = np.sqrt(2 * 50 * hw.temperature * bw * 1.38e-23) * 10 ** (hw.lnaGain / 20) * 1e6  # uV
        print('Expected by Johnson: %0.5f uV' % johnson)

        spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
        fVector = np.linspace(-self.bw / 2, self.bw / 2, num=self.nPoints) * 1e3  # kHz

        # Plot signal versus time
        result1 = {'widget': 'curve',
                   'xData': tVector,
                   'yData': [np.abs(data), np.real(data), np.imag(data)],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Noise vs time',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 0,
                   'col': 0}

        # Plot spectrum
        result2 = {'widget': 'curve',
                   'xData': fVector,
                   'yData': [np.abs(spectrum)],
                   'xLabel': 'Frequency (kHz)',
                   'yLabel': 'Mag FFT (a.u.)',
                   'title': 'Noise spectrum',
                   'legend': [''],
                   'row': 1,
                   'col': 0}

        self.output = [result1, result2]

        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()

        return self.output


if __name__ == '__main__':
    seq = Noise()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=False, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
