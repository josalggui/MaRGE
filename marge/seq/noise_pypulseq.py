"""
@author: J.M. Algarín, february 03th 2022
MRILAB @ I3M
"""

import marge.controller.experiment_gui as ex
import marge.controller.controller_device as device
import numpy as np
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import marge.configs.hw_config as hw
import marge.configs.units as units
from marga_pulseq.interpreter import PSInterpreter
import pypulseq as pp


class Noise(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(Noise, self).__init__()
        # Input the parameters
        self.decimation_factor = None
        self.oversampling_factor = None
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
        self.addParameter(key='oversampling_factor', string='Oversampling factor', val=6, field='OTH',
                          tip='Oversampling factor applied during readout')
        self.addParameter(key='decimation_factor', string='Decimation factor', val=3, field='OTH',
                          tip='Decimation applied to acquired data')
        self.addParameter(key='add_rd_points', string='Add RD points', val=10, field='OTH',
                          tip='Add RD points to avoid CIC and FIR filters issues')

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

        # Save larmor frequency
        self.mapVals['larmorFreq'] = hw.larmorFreq + self.freqOffset * 1e-6  # MHz

        # Bandwidth, sampling period, and true acquisition time
        bw = self.bw * 1e-6  # MHz
        sampling_period = 1 / bw / self.oversampling_factor  # us
        sampling_period = np.round(np.array(sampling_period) * hw.fpga_clk_freq_MHz).astype(
            np.uint32) / hw.fpga_clk_freq_MHz * self.oversampling_factor
        bw = 1 / sampling_period
        acq_time = self.nPoints * sampling_period * 1e-6  # s
        print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))
        self.mapVals['bw_MHz'] = bw
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
                               frequency=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                               bandwidth=bw,  # MHz
                               decimate='Normal',
                               oversampling_factor=self.oversampling_factor,
                               decimation_factor=self.decimation_factor,
                               )


if __name__ == '__main__':
    seq = Noise()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=False, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
