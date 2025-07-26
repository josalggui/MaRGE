"""
Created on Tue October 1st, 2024
@author: [Student's Name]
@Summary: [Sequence Name] class template
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
import marge.controller.experiment_gui as ex
import marge.configs.hw_config as hw  # Import the scanner hardware config
import marge.configs.units as units
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from marga_pulseq.interpreter import PSInterpreter  # Import the flocra-pulseq interpreter
import pypulseq as pp  # Import PyPulseq

# Template Class for MRI Sequences
class SEQUENCE_TEMPLATE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        """
        Defines the parameters for the sequence.

        Instructions for students:
        - Each parameter is defined using the `addParameter` method, which takes the following arguments:
          - key (str): A unique identifier for the parameter. This will be used to reference the value in other parts of the sequence.
          - string (str): A human-readable description of the parameter. This should clearly describe what the value represents.
          - val (int/float/str/list): The default value for the parameter. It can be a number, string, or list depending on the requirement.
          - units (optional): Units associated with the parameter (e.g., ms, cm, etc.). Use the `configs.units` module for common units.
          - field (str): The category to which the parameter belongs. It can be:
            - 'RF': Radio Frequency (parameters related to RF pulses).
            - 'IM': Imaging (parameters related to image acquisition).
            - 'SEQ': Sequence (parameters related to the sequence structure).
            - 'OTH': Other parameters that don't fit into the above categories.
          - tip (optional): Additional tips or information about the parameter, such as recommendations or constraints.
        """
        super(SEQUENCE_TEMPLATE, self).__init__()

        # Sequence name (Do not include 'field')
        self.addParameter(key='seqName', string='Sequence Name', val='Default_SeqName',
                          tip="The identifier name for the sequence.")

        # To automatically include the sequence into MaRGE.
        self.addParameter(key='toMaRGE', string='to MaRGE', val=False)

        # To let the code know that we are using tools associated to pypulseq
        self.addParameter(key='pypulseq', string='PyPulseq', val=True)

        # Number of scans
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM',
                          tip="Number of repetitions of the full scan.")

        # Number of repetitions
        self.addParameter(key='nRepetitions', string='Number of repetitions', val=5, field='SEQ',
                          tip="Number of repetitions.")

        # Excitation flip angle
        self.addParameter(key='rfExFA', string='Excitation Flip Angle (degrees)', val=90.0, field='RF',
                          tip="Flip angle of the excitation RF pulse in degrees. Common values are 90 or 180 degrees.")

        # Excitation time
        self.addParameter(key='rfExTime', string='Excitation Pulse Duration (us)', val=100.0, units=units.us,
                          field='RF',
                          tip="Duration of the RF excitation pulse in microseconds (us).")

        # Number of acquired points
        self.addParameter(key='nPoints', string='Number of Acquired Points', val=256, field='IM',
                          tip="Number of points acquired during the readout gradient (frequency-encoding).")

        # Acquisition bandwidth
        self.addParameter(key='bandwidth', string='Acquisition Bandwidth (kHz)', val=50, units=units.kHz, field='IM',
                          tip="The bandwidth of the acquisition (kHz9. This value affects resolution and SNR.")

        # Repetition time
        self.addParameter(key='repetitionTime', string='Repetition Time (ms)', val=10.0, units=units.ms, field='SEQ',
                          tip="The time between successive excitation pulses, in milliseconds.")

        # Dummy pulses
        self.addParameter(key='dummyPulses', string='Number of dummy pulses', val=1, field='SEQ',
                          tip='Number of dummy pulses at the beginning of each batch.')

        self.addParameter(key='shimming', string='Shimming', val=[0.0, 0.0, 0.0], field='SEQ')

    def sequenceInfo(self):
        """
        Description of the sequence. Students should customize this.
        """
        print("[Sequence Name] sequence")
        print("Author: [Student's Name]")
        print("Lab or Institution Name\n")
        print("More information")

    def sequenceTime(self):
        """
        Calculate the sequence time based on its parameters.
        Students can extend this method as needed.
        """
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime']

        seqTime = nScans * repetitionTime * 1e-3 / 60  # conversion to minutes
        seqTime = np.round(seqTime, decimals=2)
        return seqTime  # minutes

    def sequenceAtributes(self):
        """
        Additional sequence attributes or parameters.
        Extend this method with specific calculations or modifications.
        """
        super().sequenceAtributes()
        # Add more attributes or modifications as required.
        pass

    def sequenceRun(self, plotSeq=False, demo=False, standalone=False):
        """
        Run the MRI sequence.

        This method initializes batches and creates the full sequence by iterating through slices and phase-encoding steps.

        Instructions for students:
        - Batches divide the sequence into smaller, hardware-manageable sections.
        - `initializeBatch` sets up new sequence batches, while `createBatches` iterates through slices and phase-encoding to create the full sequence.

        Args:
            plotSeq (bool): If True, plots the sequence.
            demo (bool): If True, runs in demo mode.
            standalone (bool): If True, runs the sequence independently.

        Returns:
            bool: Indicates success or failure of the sequence run.
        """
        self.demo = demo  # Set demo mode
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
            grad_max=np.max(np.abs(hw.gFactor)) * hw.gammaB,  # Maximum gradient amplitude (Hz/m)
            grad_t=hw.grad_raster_time * 1e6,  # Gradient raster time (us)
        )

        '''
        Step 2: Define system properties using PyPulseq (pp.Opts).
        These properties define the hardware capabilities of the MRI scanner, such as maximum gradient strengths,
        slew rates, and dead times. They are typically set based on the hardware configuration file (`hw_config`).
        '''

        system = pp.Opts(
            rf_dead_time=hw.blkTime * 1e-6,  # Dead time between RF pulses (s)
            max_grad=np.max(np.abs(hw.gFactor)) * 1e3,  # Maximum gradient strength (mT/m)
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

        bw = self.bandwidth * 1e-6 # MHz
        sampling_period = 1 / bw  # us

        '''
        Step 4: Define the experiment to get the true bandwidth
        In this step, student needs to get the real bandwidth used in the experiment. To get this bandwidth, an
        experiment must be defined and the sampling period should be obtained using get_
        '''

        if not self.demo:
            expt = ex.Experiment(
                lo_freq=hw.larmorFreq,  # Larmor frequency in MHz
                rx_t=sampling_period,  # Sampling time in us
                init_gpa=False,  # Whether to initialize GPA board (False for True)
                gpa_fhdo_offset_time=(1 / 0.2 / 3.1),  # GPA offset time calculation
                auto_leds=True  # Automatic control of LEDs (False or True)
            )
            sampling_period = expt.get_sampling_period()  # us
            bw = 1 / sampling_period  # MHz
            print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))
            expt.__del__()
        self.mapVals['bw_MHz'] = bw
        self.mapVals['sampling_period_us'] = sampling_period

        '''
        Step 5: Define sequence blocks.
        In this step, you will define the building blocks of the MRI sequence, including the RF pulses, gradient pulses,
        and ADC blocks.
        '''

        ## Excitation pulse
        # Define the RF excitation pulse using PyPulseq. The flip angle is typically in radians.
        flip_ex = self.rfExFA * np.pi / 180  # Convert flip angle from degrees to radians
        rf_ex = pp.make_block_pulse(
            flip_angle=flip_ex,  # Set the flip angle for the RF pulse
            system=system,  # Use the system properties defined earlier
            duration=self.rfExTime,  # Set the RF pulse duration
            delay=0,  # Delay before the RF pulse (if any)
            phase_offset=0.0,  # Set the phase offset for the pulse (0 by default)
        )

        ## ADC block
        # Define the ADC block using PyPulseq. You need to specify number of samples and delay.
        adc = pp.make_adc(
            num_samples=self.nPoints,
            dwell=sampling_period * 1e-6,
            delay=hw.blkTime*1e-6 + self.rfExTime + hw.deadTime*1e-6
        )

        ## Repetition delay
        # Define the delay for repetition.
        delay_repetition = pp.make_delay(self.repetitionTime)

        # Additional considerations for students:
        # - Make sure timing calculations account for hardware limitations, such as gradient raster time and dead time.

        '''
        Step 6: Define your initializeBatch according to your sequence.
        In this step, you will create the initializeBatch method to create dummy pulses that will be initialized for
        each new batch.
        '''

        def initializeBatch():
            """
            Initializes a batch of MRI sequence blocks using PyPulseq for a given experimental configuration.

            Returns:
            --------
            tuple
                - batch (pp.Sequence): A PyPulseq sequence object containing the configured sequence blocks.
                - n_rd_points (int): Total number of readout points in the batch.
                - n_adc (int): Total number of ADC acquisitions in the batch.
            """

            # Instantiate pypulseq sequence object
            batch = pp.Sequence(system)
            n_rd_points = 0
            n_adc = 0

            # Add dummy pulses to batch
            for _ in range(self.dummyPulses):
                batch.add_block(rf_ex, delay_repetition)

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

            # Loop through all repetitions (e.g., slices)
            for repetition in range(self.nRepetitions):
                # Check if a new batch is needed (either first batch or exceeding readout points limit)
                if seq_idx == 0 or n_rd_points + self.nPoints > hw.maxRdPoints:
                    # If a previous batch exists, write and interpret it
                    if seq_idx > 0:
                        batches[batch_num].write(batch_num + ".seq")
                        waveforms[batch_num], param_dict = flo_interpreter.interpret(batch_num + ".seq")
                        print(f"{batch_num}.seq ready!")

                    # Update to the next batch
                    seq_idx += 1
                    n_rd_points_dict[batch_num] = n_rd_points  # Save readout points count
                    n_rd_points = 0
                    batch_num = f"batch_{seq_idx}"
                    batches[batch_num], n_rd_points, n_adc_0 = initializeBatch()  # Initialize new batch
                    n_adc += n_adc_0
                    print(f"Creating {batch_num}.seq...")

                # Add sequence blocks (RF, ADC, repetition delay) to the current batch
                batches[batch_num].add_block(rf_ex, adc, delay_repetition)
                n_rd_points += self.nPoints  # Accounts for additional acquired points in each adc block
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
                               bandwidth=bw,  # MHz
                               decimate='Normal',
                               )

    def sequenceAnalysis(self, mode=None):


        # create self.out to run in iterative mode
        self.output = []

        # save data once self.output is created
        self.saveRawData()

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        return self.output


if __name__ == "__main__":
    # main(plot=True, write_seq=True)

    seq = SEQUENCE_TEMPLATE()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=True, demo=True, standalone=True)
    # seq.sequenceAnalysis(mode='standalone')
    # seq.plotResults()