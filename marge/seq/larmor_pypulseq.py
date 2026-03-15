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
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import marge.configs.hw_config as hw
import marge.configs.units as units
import marge.controller.experiment_gui as ex
import marge.controller.controller_device as device
from marga_pulseq.interpreter import PSInterpreter
import pypulseq as pp


class LarmorPyPulseq(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(LarmorPyPulseq, self).__init__()
        # Input the parameters
        self.decimation_factor = None
        self.oversampling_factor = None
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
        self.rfReTime = None
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
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ', units=units.ms)
        self.addParameter(key='bw', string='Bandwidth (kHz)', val=50, field='RF', units=units.kHz)
        self.addParameter(key='dF', string='Frequency resolution (Hz)', val=100, field='RF')
        self.addParameter(key='shimming', string='Shimming', val=[0.0, 0.0, 0.0], field='OTH', units=units.sh)
        self.addParameter(key='oversampling_factor', string='Oversampling factor', val=6, field='OTH',
                          tip='Oversampling factor applied during readout')
        self.addParameter(key='decimation_factor', string='Decimation factor', val=3, field='OTH',
                          tip='Decimation applied to acquired data')
        self.addParameter(key='add_rd_points', string='Add RD points', val=10, field='OTH',
                          tip='Add RD points to avoid CIC and FIR filters issues')

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
        init_gpa = False  # Starts the gpa
        self.demo = demo

        '''
        Step 1: Define the interpreter for FloSeq/PSInterpreter.
        The interpreter is responsible for converting the high-level pulse sequence description into low-level
        instructions for the scanner hardware.
        '''

        # Define the interpreter. It should be updated on calibration
        flo_interpreter = PSInterpreter(tx_warmup=hw.blkTime,  # us
                                             rf_center=hw.larmorFreq * 1e6,  # Hz
                                             rf_amp_max=hw.b1Efficiency / (2 * np.pi) * 1e6,  # Hz
                                             gx_max=hw.gFactor[0] * hw.gammaB,  # Hz/m
                                             gy_max=hw.gFactor[1] * hw.gammaB,  # Hz/m
                                             gz_max=hw.gFactor[2] * hw.gammaB,  # Hz/m
                                             grad_max=np.max(hw.gFactor) * hw.gammaB,  # Hz/m
                                             )

        '''
        Step 2: Define system properties using PyPulseq (pp.Opts).
        These properties define the hardware capabilities of the MRI scanner, such as maximum gradient strengths,
        slew rates, and dead times. They are typically set based on the hardware configuration file (`hw_config`).
        '''

        # Define system properties according to hw_config file
        system = pp.Opts(
            rf_dead_time=hw.blkTime * 1e-6,  # s
            max_grad=hw.max_grad,  # mT/m
            grad_unit='mT/m',
            max_slew=hw.max_slew_rate,  # mT/m/ms
            slew_unit='mT/m/ms',
            grad_raster_time=hw.grad_raster_time,  # s
            rise_time=hw.grad_rise_time,  # s
            adc_raster_time=1e-9,
            rf_raster_time = 1e-9,
            block_duration_raster = 1e-9
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
        self.mapVals['nPoints'] = n_points
        self.mapVals['acqTime'] = acq_time
        self.mapVals['echoTime'] = echo_time

        '''
        Step 4: Define the experiment to get the true bandwidth
        In this step, student needs to get the real bandwidth used in the experiment. To get this bandwidth, an
        experiment must be defined and the sampling period should be obtained using get_
        '''

        # Initialize the experiment
        self.bw = n_points / acq_time  # Hz
        sampling_period = 1 / self.bw  # s
        self.mapVals['samplingPeriod'] = sampling_period
        if not self.demo:
            if hw.marcos_version=="MaRCoS":
                dev = ex.Experiment(
                    lo_freq=self.larmorFreq * 1e-6,  # MHz
                    rx_t=sampling_period * 1e6,  # us
                    init_gpa=init_gpa,
                    gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                    auto_leds=True
                )
            elif hw.marcos_version=="MIMO":
                # Define device arguments
                dev_kwargs = {
                    "lo_freq": self.larmorFreq * 1e-6,  # MHz
                    "rx_t": sampling_period * 1e6,  # us
                    "print_infos": True,
                    "assert_errors": True,
                    "halt_and_reset": False,
                    "fix_cic_scale": True,
                    "set_cic_shift": False,  # needs to be true for open-source cores
                    "flush_old_rx": False,
                    "init_gpa": False,
                    "gpa_fhdo_offset_time": 1 / 0.2 / 3.1,
                    "auto_leds": True,
                    "oversampling_factor": self.oversampling_factor,
                }

                # Define master arguments
                master_kwargs = {
                    'mimo_master': True,
                    'trig_output_time': 1e5,
                    'slave_trig_latency': 6.079
                }

                # Define experiment
                dev = device.Device(ip_address=hw.rp_ip_list[0], port=hw.rp_port[0], **(master_kwargs | dev_kwargs))

            else:
                print("Wrong MaRCoS version")

            sampling_period = dev.get_sampling_period()  # us
            self.bw = 1 / sampling_period  # MHz
            acq_time = n_points / self.bw * 1e-6  # s
            self.mapVals['bw_true'] = self.bw * 1e3  # kHz
            print("Acquisition bandwidth fixed to: %0.3f kHz" % (self.bw * 1e3))
            dev.__del__()

        else:
            sampling_period = sampling_period * 1e6  # us
            self.bw = 1 / sampling_period  # MHz
            acq_time = n_points / self.bw * 1e-6  # s


        '''
        Step 5: Define sequence blocks.
        In this step, you will define the building blocks of the MRI sequence, including the RF pulses, gradient pulses,
        and ADC blocks.
        '''
        round_rf = int(np.abs(np.log10(np.abs(system.rf_raster_time))))
        round_bl = int(np.abs(np.log10(np.abs(system.block_duration_raster))))

        # Create excitation rf event
        delay_rf_ex = self.repetitionTime-acq_time/2-echo_time-self.rfExTime/2
        rf_duration = np.round(self.rfExTime, decimals=round_rf)
        block_duration = np.round(delay_rf_ex + rf_duration, decimals=round_bl)
        delay_rf_ex = block_duration - rf_duration
        event_rf_ex = pp.make_block_pulse(
            flip_angle=self.rfExFA * np.pi / 180,
            duration=rf_duration,
            delay=delay_rf_ex,
            system=system,
            use="excitation",
        )

        # Create refocusing rf event
        delay_rf_re = echo_time/2-self.rfExTime/2-self.rfReTime/2
        rf_duration = np.round(self.rfReTime, decimals=round_rf)
        block_duration = np.round(delay_rf_re + rf_duration, decimals=round_bl)
        delay_rf_re = block_duration - rf_duration
        event_rf_re = pp.make_block_pulse(flip_angle=self.rfReFA * np.pi / 180,
                                          duration=rf_duration,
                                          delay=delay_rf_re,
                                          system=system,
                                          use="refocusing")

        # Create ADC event
        delay_adc = echo_time/2-self.rfReTime/2-acq_time/2
        block_duration = np.round(delay_adc + acq_time, decimals=round_bl)
        delay_adc = block_duration - acq_time
        event_adc = pp.make_adc(num_samples=n_points,
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

            ok, report = batches[batch_num].check_timing()

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
                               oversampling_factor=self.oversampling_factor,
                               decimation_factor=self.decimation_factor,
                               )

    def sequenceAnalysis(self, mode=None):
        super().sequenceAnalysis(mode=mode)

        if mode != 'Standalone':
            for sequence in self.sequence_list.values():
                if 'larmorFreq' in sequence.mapVals:
                    sequence.mapVals['larmorFreq'] = hw.larmorFreq

        return self.output


if __name__ == '__main__':
    seq = LarmorPyPulseq()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
