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
import experiment as ex
import scipy.signal as sig
import configs.hw_config as hw  # Import the scanner hardware config
import configs.units as units
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from flocra_pulseq.interpreter import PSInterpreter  # Import the flocra-pulseq interpreter
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

        # Step 1: Define the interpreter for FloSeq/PSInterpreter.
        # The interpreter is responsible for converting the high-level pulse sequence description into low-level
        # instructions for the scanner hardware. You will typically update the interpreter during scanner calibration.
        self.flo_interpreter = PSInterpreter(
            tx_warmup=hw.blkTime,  # Transmit chain warm-up time (us)
            rf_center=hw.larmorFreq * 1e6,  # Larmor frequency (Hz)
            rf_amp_max=hw.b1Efficiency / (2 * np.pi) * 1e6,  # Maximum RF amplitude (Hz)
            gx_max=hw.gFactor[0] * hw.gammaB,  # Maximum gradient amplitude for X (Hz/m)
            gy_max=hw.gFactor[1] * hw.gammaB,  # Maximum gradient amplitude for Y (Hz/m)
            gz_max=hw.gFactor[2] * hw.gammaB,  # Maximum gradient amplitude for Z (Hz/m)
            grad_max=np.max(hw.gFactor) * hw.gammaB,  # Maximum gradient amplitude (Hz/m)
            grad_t=hw.grad_raster_time * 1e6,  # Gradient raster time (us)
        )

        # Step 2: Define system properties using PyPulseq (pp.Opts).
        # These properties define the hardware capabilities of the MRI scanner, such as maximum gradient strengths,
        # slew rates, and dead times. They are typically set based on the hardware configuration file (`hw_config`).
        self.system = pp.Opts(
            rf_dead_time=(hw.blkTime + 5) * 1e-6,  # Dead time between RF pulses (s)
            max_grad=hw.max_grad,  # Maximum gradient strength (mT/m)
            grad_unit='mT/m',  # Units of gradient strength
            max_slew=hw.max_slew_rate,  # Maximum gradient slew rate (mT/m/ms)
            slew_unit='mT/m/ms',  # Units of gradient slew rate
            grad_raster_time=hw.grad_raster_time,  # Gradient raster time (s)
            rise_time=hw.grad_rise_time,  # Gradient rise time (s)
        )

        # Step 3: Perform any calculations required for the sequence.
        # In this step, students can implement the necessary calculations, such as timing calculations, RF amplitudes, and
        # gradient strengths, before defining the sequence blocks.
        bw = self.bandwidth * 1e-6 # MHz
        bw_ov = self.bandwdith - hw.oversamplingFactor  # MHz
        sampling_period = 1 / bw_ov  # us

        # Step 4: Define the experiment to get the true bandwidth
        # In this step, student need to get the real bandwidth used in the experiment. To get this bandwidth, an
        # experiment must be defined and the sampling period should be obtained using get_rx_ts()[0]


        if not self.demo:
            expt = ex.Experiment(
                lo_freq=hw.larmorFreq,  # Larmor frequency in MHz
                rx_t=sampling_period,  # Sampling time in us
                init_gpa=False,  # Whether to initialize GPA board (False for True)
                gpa_fhdo_offset_time=(1 / 0.2 / 3.1),  # GPA offset time calculation
                auto_leds=True  # Automatic control of LEDs (False or True)
            )
            sampling_period = expt.get_rx_ts()[0]  # us
            bw = 1 / sampling_period / hw.oversamplingFactor  # MHz
            print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))
            expt.__del__()
        self.mapVals['bw_MHz'] = bw
        self.mapVals['sampling_period_us'] = sampling_period

        # Step 5: Define sequence blocks.
        # In this step, you will define the building blocks of the MRI sequence, including the RF pulses and gradient pulses.

        ## Excitation pulse
        # Define the RF excitation pulse using PyPulseq. The flip angle is typically in radians.
        flip_ex = self.rfExFA * np.pi / 180  # Convert flip angle from degrees to radians
        rf_ex = pp.make_block_pulse(
            flip_angle=flip_ex,  # Set the flip angle for the RF pulse
            system=self.system,  # Use the system properties defined earlier
            duration=self.rfExTime,  # Set the RF pulse duration
            delay=0,  # Delay before the RF pulse (if any)
            phase_offset=0.0,  # Set the phase offset for the pulse (0 by default)
        )

        ## ADC block
        # Define the ADC block using PyPulseq. You need to specify number of samples and delay.
        adc = pp.make_adc(
            num_samples=self.nPoints, dwell=1 / self.bandwidth,
            delay=hw.blkTime*1e-6 + self.rfExTime + hw.deadTime*1e-6
        )

        ## Repetition delay
        # Define the delay for repetition.
        delay_repetition = pp.make_delay(self.repetitionTime)

        # Additional considerations for students:
        # - Make sure timing calculations account for hardware limitations, such as gradient raster time and dead time.

        # Step 6: Define your initializeBatch according to your sequence.
        # In this step, you will create the initializeBatch method to create dummy pulses that will be initialized for
        # each new batch.

        # Initialize batches dictionary to store different parts of the sequence.
        batches = {}

        def initializeBatch(name="pp_1"):
            """
            Initializes a new sequence batch.

            Args:
                name (str): Name of the batch. Defaults to "pp_1".
            """
            batches[name] = pp.Sequence(self.system)

            # Add dummy pulses to batch
            for _ in range(self.dummyPulses):
                batches[name].add_block(rf_ex, delay_repetition)

        # Step 7: Define your createBatches method.
        # In this step you will populate the batches adding the blocks previously defined in step 4, and accounting for
        # number of acquired points to check if a new batch is required.

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
            """
            n_rd_points = 0  # Initialize the readout points counter
            n_rd_points_dict = {}  # Dictionary to track readout points for each batch
            seq_idx = 0  # Sequence batch index
            seq_num = "batch_0"  # Initial batch name
            waveforms = {}  # Dictionary to store generated waveforms

            # Loop through all repetitions (e.g., slices)
            for repetition in range(self.nRepetitions):
                # Check if a new batch is needed (either first batch or exceeding readout points limit)
                if seq_idx == 0 or n_rd_points + self.nPoints > hw.maxRdPoints:
                    # If a previous batch exists, write and interpret it
                    if seq_idx > 0:
                        batches[seq_num].write(seq_num + ".seq")
                        waveforms[seq_num], param_dict = self.flo_interpreter.interpret(seq_num + ".seq")
                        print(f"{seq_num}.seq ready!")

                    # Update to the next batch
                    seq_idx += 1
                    n_rd_points_dict[seq_num] = n_rd_points  # Save readout points count
                    seq_num = f"batch_{seq_idx}"
                    initializeBatch(seq_num)  # Initialize new batch
                    n_rd_points = 0  # Reset readout points count
                    print(f"Creating {seq_num}.seq...")

                # Add sequence blocks (RF, ADC, repetition delay) to the current batch
                batches[seq_num].add_block(rf_ex, adc, delay_repetition)
                n_rd_points += self.nPoints  # Accounts for additional acquired points in each adc block

            # After final repetition, save and interpret the last batch
            batches[seq_num].write(seq_num + ".seq")
            waveforms[seq_num], param_dict = self.flo_interpreter.interpret(seq_num + ".seq")
            print(f"{seq_num}.seq ready!")
            print(f"{len(batches)} batches created. Sequence ready!")

            # Update the number of acquired ponits in the last batch
            n_rd_points_dict.pop('batch_0')
            n_rd_points_dict[seq_num] = n_rd_points

            return waveforms, n_rd_points_dict

        # Step 8: Run the batches
        # This step will handle the different batches, run it and get the resulting data. This should not be modified

        # Generate batches and get waveforms and readout points
        waveforms, n_readouts = createBatches()
        self.mapVals['n_readouts'] = list(n_readouts.values())
        self.mapVals['n_batches'] = len(n_readouts.values())

        # Initialize a list to hold oversampled data
        data_over = []

        # Iterate through each batch of waveforms
        for seq_num in waveforms.keys():
            # Initialize the experiment if not in demo mode
            if not self.demo:
                self.expt = ex.Experiment(
                    lo_freq=hw.larmorFreq,  # Larmor frequency in MHz
                    rx_t=1 / self.bandwidth * hw.oversamplingFactor * 1e6,  # Sampling time in us
                    init_gpa=False,  # Whether to initialize GPA board (False for now)
                    gpa_fhdo_offset_time=(1 / 0.2 / 3.1),  # GPA offset time calculation
                    auto_leds=True  # Automatic control of LEDs
                )

            # Convert the PyPulseq waveform to the Red Pitaya compatible format
            self.pypulseq2mriblankseq(waveforms=waveforms[seq_num], shimming=[0.0, 0.0, 0.0])

            # Load the waveforms into Red Pitaya if not in demo mode
            if not self.floDict2Exp():
                print("ERROR: Sequence waveforms out of hardware bounds")
                return False
            else:
                print("Sequence waveforms loaded successfully")

            # If not plotting the sequence, start scanning
            if not plotSeq:
                for scan in range(self.nScans):
                    print(f"Scan {scan + 1}, batch {seq_num.split('_')[-1]}/{len(n_readouts)} running...")
                    acquired_points = 0
                    expected_points = n_readouts[seq_num] * hw.oversamplingFactor  # Expected number of points

                    # Continue acquiring points until we reach the expected number
                    while acquired_points != expected_points:
                        if not self.demo:
                            rxd, msgs = expt.run()  # Run the experiment and collect data
                        else:
                            # In demo mode, generate random data as a placeholder
                            rxd = {'rx0': np.random.randn(expected_points) + 1j * np.random.randn(expected_points)}

                        # Update acquired points
                        acquired_points = np.size(rxd['rx0'])

                    # Concatenate acquired data into the oversampled data array
                    data_over = np.concatenate((data_over, rxd['rx0']), axis=0)
                    print(f"Acquired points = {acquired_points}, Expected points = {expected_points}")
                    print(f"Scan {scan + 1}, batch {seq_num[-1]}/{len(n_readouts)} ready!")

                # Decimate the oversampled data and store it
                self.mapVals['data_over'] = data_over

            elif plotSeq and standalone:
                # Plot the sequence if requested and return immediately
                self.sequencePlot(standalone=standalone)

            if not self.demo:
                self.expt.__del__()

        return True

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