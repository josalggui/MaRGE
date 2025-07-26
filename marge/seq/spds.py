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
from marga_pulseq.interpreter import PSInterpreter  # Import the marga_pulseq interpreter
import pypulseq as pp  # Import PyPulseq
from skimage.restoration import unwrap_phase as unwrap
from sklearn.preprocessing import PolynomialFeatures
from numpy.linalg import lstsq
from marge.marge_utils.utils import run_ifft


# Template Class for MRI Sequences
class spds(blankSeq.MRIBLANKSEQ):
    """
    Executes the SPDS (Single Point Double Shot) sequence, designed to estimate the B₀ map by acquiring data with two
    distinct acquisition windows.
    """

    def __init__(self):
        """
        Defines the parameters for the sequence.
        """
        super(spds, self).__init__()

        self.angulation = None
        self.mask = None
        self.axesOrientation = None
        self.rfExFA = None
        self.standalone = None
        self.dummyPulses = None
        self.repetitionTime = None
        self.rfExTime = None
        self.deadTime = None
        self.nPoints = None
        self.fov = None
        self.dfov = None
        self.bw = None
        self.plotSeq = None
        self.addParameter(key='seqName', string='Sequence Name', val='SPDS',
                          tip="The identifier name for the sequence.")
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='pypulseq', string='PyPulseq', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM',
                          tip='Number of repetitions of the full scan.')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.055, field='RF',
                          tip='Larmor frequency.')
        self.addParameter(key='rfExFA', string='Excitation Flip Angle (degrees)', val=90.0, field='RF',
                          tip="Flip angle of the excitation RF pulse in degrees")
        self.addParameter(key='rfExTime', string='Excitation time (us)', val=15.0, units=units.us, field='RF',
                          tip="Duration of the RF excitation pulse in microseconds (us).")
        self.addParameter(key='nPoints', string='Matrix size [rd, ph, sl]', val=[2, 2, 1], field='IM',
                          tip='Matrix size for the acquired images.')
        self.addParameter(key='fov', string='Field of View (cm)', val=[24.0, 24.0, 24.0], units=units.cm, field='IM',
                          tip='Field of View (cm).')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM',
                          tip="Position of the gradient isocenter")
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[0, 1, 2], field='IM',
                          tip="0=x, 1=y, 2=z")
        self.addParameter(key='repetitionTime', string='Repetition Time (ms)', val=30.0, units=units.ms, field='SEQ',
                          tip="The time between successive excitation pulses, in milliseconds (ms).")
        self.addParameter(key='deadTime', string='Dead times (us)', val=[350.0, 450.0], units=units.us, field='SEQ',
                          tip='Dead time for the two acquisitions in microseconds (us).')
        self.addParameter(key='dummyPulses', string='Number of dummy pulses', val=5, field='SEQ',
                          tip='Number of dummy pulses at the beginning of each batch.')
        self.addParameter(key='shimming', string='Shimming', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH',
                          tip='Shimming parameter to compensate B0 linear inhomogeneity.')
        self.addParameter(key='bw', string='Bandwidth (kHz)', val=50.0, units=units.kHz, field='IMG',
                          tip='Set acquisition bandwidth in kilohertz (kHz).')
        self.addParameter(key='angle', string='Angle (º)', val=0.0, field='IM',
                          tip='Angle in degrees to rotate the fov')
        self.addParameter(key='rotationAxis', string='Rotation axis', val=[0, 0, 1], field='IM',
                          tip='Axis of rotation')
        self.addParameter(key='interpOrder', string='Zero Padding Order', val=3, field='IM',
                          tip='Zero Padding Order')
        self.addParameter(key='fittingOrder', string='Poly Fitting Order', val=4, field='IM',
                          tip='Polynomics fitting order')
        self.addParameter(key='thresholdMask', string='% Threshold Mask', val=10, field='IM',
                          tip='% Threshold Mask')

    def sequenceInfo(self):
        """
        Description of the sequence.
        """
        print("SPDS")
        print("Contributor: PhD. J.M. Algarín")
        print("Contributor: PhD. J. Borreguero")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain \n")
        print("Single Point Double Shot protocol to measure B0 map")

    def sequenceTime(self):
        """
        Calculate the sequence time based on its parameters.

        Output:
        -------
        - time (float): sequence time in minutes
        """

        nPoints = self.mapVals['nPoints']
        kx = np.linspace(start=-1, stop=1, endpoint=False, num=nPoints[0])
        ky = np.linspace(start=-1, stop=1, endpoint=False, num=nPoints[1])
        if nPoints[2] > 1:
            kz = np.linspace(start=-1, stop=1, endpoint=False, num=nPoints[2])
        else:
            kz = np.array([0.0])
        ky, kz, kx = np.meshgrid(ky, kz, kx)
        k_norm = np.zeros(shape=(np.size(kx), 3))
        k_norm[:, 0] = np.reshape(kx, -1)
        k_norm[:, 1] = np.reshape(ky, -1)
        k_norm[:, 2] = np.reshape(kz, -1)
        distance = np.sqrt(np.sum(k_norm ** 2, axis=1))
        self.mask = distance <= 1
        n = np.sum(self.mask)

        tr = self.mapVals['repetitionTime'] * 1e-3  # s
        time = tr * n / 60 * 2 * self.mapVals['nScans']  # minutes

        return time  # minutes

    def sequenceRun(self, plotSeq=False, demo=False, standalone=False):
        """
        This method orchestrates the definition, preparation,
        and execution of the sequence.

        Parameters:
        -----------
        plotSeq : bool, optional
            If True, the sequence is plotted for visualization without execution (default: False).
        demo : bool, optional
            If True, runs the sequence in demonstration mode without hardware communication (default: False).
        standalone : bool, optional
            If True, runs the sequence independently, without external triggering or batch processing (default: False).

        Output:
        -------
        - Oversampled and decimated data stored in `self.mapVals['data_over']` and `self.mapVals['data_decimated']`.
        - Generated sequence files for verification and debugging.

        Notes:
        ------
        - This method is tailored for SPDS sequences, which are particularly useful for estimating
          B₀ inhomogeneities in low-field MRI systems under high inhomogeneities.
        """

        self.demo = demo
        self.plotSeq = plotSeq
        self.standalone = standalone
        self.angulation = 0

        '''
        Step 1: Define the interpreter for FloSeq/PSInterpreter.
        The interpreter is responsible for converting the high-level pulse sequence description into low-level
        instructions for the scanner hardware.
        '''

        flo_interpreter = PSInterpreter(
            tx_warmup=hw.blkTime,  # Transmit chain warm-up time (us)
            rf_center=self.mapVals['larmorFreq'] * 1e6,  # Larmor frequency (Hz)
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
            grad_raster_time=1e-6,  # Gradient raster time (s)
            rise_time=hw.grad_rise_time,  # Gradient rise time (s)
            rf_raster_time=1e-6,
            block_duration_raster=1e-6
        )

        '''
        Step 3: Perform any calculations required for the sequence.
        In this step, students can implement the necessary calculations, such as timing calculations, RF amplitudes, and
        gradient strengths, before defining the sequence blocks.
        '''

        # Set the fov
        self.dfov = self.getFovDisplacement()
        self.dfov = self.dfov[self.axesOrientation]
        self.fov = self.fov[self.axesOrientation]

        # Get k-space info
        dk = 1 / self.fov  # m^-1
        k_max = self.nPoints / (2 * self.fov)  # m^-1
        self.mapVals['dk'] = dk
        self.mapVals['k_max'] = k_max

        # Get bandwidths and acquisition window
        bw_a = self.bw  # Hz
        bw_b = self.bw  # Hz
        n_rd = 1 + 2 * hw.addRdPoints

        # Fix excitation time to avoid rounding issues
        rfExTime = 2 * system.block_duration_raster * np.ceil(self.rfExTime / 2 / system.block_duration_raster)
        if rfExTime != self.rfExTime:
            print(f"WARNING: Excitation time fixed to {rfExTime}.")
            self.rfExTime = rfExTime

        # Get timing parameters
        time_acq_a = n_rd / bw_a  # s
        time_acq_b = n_rd / bw_b  # s
        time_delay0_a = self.rfExTime / 2 + self.deadTime[0] + time_acq_a / 2  # s
        time_delay0_b = self.rfExTime / 2 + self.deadTime[1] + time_acq_b / 2  # s
        time_grad_a = self.repetitionTime - time_delay0_a - hw.grad_rise_time - self.rfExTime / 2  # s
        time_grad_b = self.repetitionTime - time_delay0_b - hw.grad_rise_time - self.rfExTime / 2  # s

        # Get cartesian points
        kx = np.linspace(start=-1, stop=1, endpoint=False, num=self.nPoints[0])
        ky = np.linspace(start=-1, stop=1, endpoint=False, num=self.nPoints[1])
        if self.nPoints[2] > 1:
            kz = np.linspace(start=-1, stop=1, endpoint=False, num=self.nPoints[2])
        else:
            kz = np.array([0.0])
        ky, kz, kx = np.meshgrid(ky, kz, kx)
        k_norm = np.zeros(shape=(np.size(kx), 3))
        k_norm[:, 0] = np.reshape(kx, -1)
        k_norm[:, 1] = np.reshape(ky, -1)
        k_norm[:, 2] = np.reshape(kz, -1)
        distance = np.sqrt(np.sum(k_norm ** 2, axis=1))
        k_cartesian = np.zeros_like(k_norm)
        k_cartesian[:, 0] = k_norm[:, 0] * k_max[0]  # m^-1
        k_cartesian[:, 1] = k_norm[:, 1] * k_max[1]  # m^-1
        k_cartesian[:, 2] = k_norm[:, 2] * k_max[2]  # m^-1
        self.mask = distance <= 1
        self.mapVals['k_cartesian'] = k_cartesian

        # Get gradients
        gradients_a = k_cartesian / (hw.gammaB * self.deadTime[0])  # T/m
        gradients_b = k_cartesian / (hw.gammaB * self.deadTime[1])  # T/m
        gradients_a = gradients_a[self.mask]
        gradients_b = gradients_b[self.mask]
        self.mapVals['gradients_a'] = gradients_a
        self.mapVals['gradients_b'] = gradients_b
        gradients_a = np.vstack([[0.0, 0.0, 0.0], gradients_a, [0.0, 0.0, 0.0]])
        gradients_b = np.vstack([[0.0, 0.0, 0.0], gradients_b, [0.0, 0.0, 0.0]])

        # Map the axis to "x", "y", and "z" according ot axesOrientation
        axes_map = {0: "x", 1: "y", 2: "z"}
        rd_channel = axes_map.get(self.axesOrientation[0], "")
        ph_channel = axes_map.get(self.axesOrientation[1], "")
        sl_channel = axes_map.get(self.axesOrientation[2], "")

        '''
        Step 4: Define the experiment to get the true bandwidth
        In this step, student needs to get the real bandwidth used in the experiment. To get this bandwidth, an
        experiment must be defined and the sampling period should be obtained using get_
        '''

        # Experiment A
        if not demo:
            expt = ex.Experiment(
                lo_freq=self.mapVals['larmorFreq'],  # Larmor frequency in MHz
                rx_t=1e6 / bw_a,  # Sampling time in us
                init_gpa=False,  # Whether to initialize GPA board (False for True)
                gpa_fhdo_offset_time=(1 / 0.2 / 3.1),  # GPA offset time calculation
                auto_leds=True  # Automatic control of LEDs (False or True)
            )
            sampling_period = expt.get_sampling_period()  # us
            bw_a = 1 / sampling_period  # MHz
            print("Acquisition bandwidth A fixed to: %0.3f kHz" % (bw_a * 1e3))
            expt.__del__()
        else:
            bw_a *= 1e-6  # MHz
        self.mapVals['bw_a_kHz'] = bw_a * 1e3

        # Experiment B
        if not demo:
            expt = ex.Experiment(
                lo_freq=self.mapVals['larmorFreq'],  # Larmor frequency in MHz
                rx_t=1e6 / bw_b,  # Sampling time in us
                init_gpa=False,  # Whether to initialize GPA board (False for True)
                gpa_fhdo_offset_time=(1 / 0.2 / 3.1),  # GPA offset time calculation
                auto_leds=True  # Automatic control of LEDs (False or True)
            )
            sampling_period = expt.get_sampling_period()  # us
            bw_b = 1 / sampling_period  # MHz
            print("Acquisition bandwidth B fixed to: %0.3f kHz" % (bw_b * 1e3))
            expt.__del__()
        else:
            bw_b *= 1e-6  # MHz
        self.mapVals['bw_b_kHz'] = bw_b * 1e3

        '''
        Step 5: Define sequence blocks.
        In this step, you will define the building blocks of the MRI sequence, including the RF pulses, gradient pulses,
        and ADC blocks.
        '''

        # First delay, sequence will start after 1 repetition time, this ensure gradient and ADC latency is not an issue.
        delay_first_a = pp.make_delay(time_delay0_a)
        delay_first_b = pp.make_delay(time_delay0_b)

        ## Excitation pulse
        # Define the RF excitation pulse using PyPulseq. The flip angle is typically in radians.
        flip_ex = self.rfExFA * np.pi / 180  # Convert flip angle from degrees to radians
        block_rf_a = pp.make_block_pulse(
            flip_angle=flip_ex,  # Set the flip angle for the RF pulse
            system=system,  # Use the system properties defined earlier
            duration=self.rfExTime,  # Set the RF pulse duration
            delay=hw.grad_rise_time + time_grad_a,  # Delay before the RF pulse (if any)
            phase_offset=0.0,  # Set the phase offset for the pulse (0 by default)
        )
        block_rf_b = pp.make_block_pulse(
            flip_angle=flip_ex,  # Set the flip angle for the RF pulse
            system=system,  # Use the system properties defined earlier
            duration=self.rfExTime,  # Set the RF pulse duration
            delay=hw.grad_rise_time + time_grad_b,  # Delay before the RF pulse (if any)
            phase_offset=0.0,  # Set the phase offset for the pulse (0 by default)
        )

        ## ADC block
        # Define the ADC block using PyPulseq. You need to specify number of samples and delay.
        block_adc_a = pp.make_adc(
            num_samples=n_rd,
            dwell=1 / bw_a * 1e-6,
            delay=self.repetitionTime - n_rd / bw_a * 1e-6,
        )
        block_adc_b = pp.make_adc(
            num_samples=n_rd,
            dwell=1 / bw_b * 1e-6,
            delay=self.repetitionTime - n_rd / bw_b * 1e-6,
        )

        '''
        Step 6: Define your initializeBatch according to your sequence.
        In this step, you will create the initializeBatch method to create dummy pulses that will be initialized for
        each new batch.
        '''

        def initialize_batch(case='a'):
            # Instantiate pypulseq sequence object
            batch = pp.Sequence(system)
            n_rd_points = 0
            n_adc = 0

            # Add dummy pulses
            if case == 'a':
                batch.add_block(delay_first_a)
                for _ in range(self.dummyPulses):
                    batch.add_block(block_rf_a, pp.make_delay(self.repetitionTime))
            elif case == 'b':
                batch.add_block(delay_first_b)
                for _ in range(self.dummyPulses):
                    batch.add_block(block_rf_b, pp.make_delay(self.repetitionTime))

            return batch, n_rd_points, n_adc

        '''
        Step 7: Define your createBatches method.
        In this step you will populate the batches adding the blocks previously defined in step 4, and accounting for
        number of acquired points to check if a new batch is required.
        '''

        def create_batches(case='a'):
            batches = {}  # Dictionary to save batches PyPulseq sequences
            waveforms = {}  # Dictionary to store generated waveforms per each batch
            n_rd_points_dict = {}  # Dictionary to track readout points for each batch
            n_adc = 0  # To account for number of adc windows
            batch_name = f'sequence_{case}'

            # Initialize new batch
            batches[batch_name], n_rd_points, n_adc_0 = initialize_batch(case=case)
            n_adc += n_adc_0

            # Populate the sequence
            for ii in range(1, np.size(gradients_a, 0)):
                if case == 'a':
                    g_rd_amp = np.array([gradients_a[ii - 1, 0], gradients_a[ii, 0], gradients_a[ii, 0]]) * hw.gammaB
                    g_ph_amp = np.array([gradients_a[ii - 1, 1], gradients_a[ii, 1], gradients_a[ii, 1]]) * hw.gammaB
                    g_sl_amp = np.array([gradients_a[ii - 1, 2], gradients_a[ii, 2], gradients_a[ii, 2]]) * hw.gammaB
                    g_time = np.array([0.0, hw.grad_rise_time, self.repetitionTime])
                    batches[batch_name].add_block(
                        pp.make_extended_trapezoid(channel=rd_channel,
                                                   amplitudes=g_rd_amp,
                                                   times=g_time),
                        pp.make_extended_trapezoid(channel=ph_channel,
                                                   amplitudes=g_ph_amp,
                                                   times=g_time),
                        pp.make_extended_trapezoid(channel=sl_channel,
                                                   amplitudes=g_sl_amp,
                                                   times=g_time),
                        block_rf_a,
                        block_adc_a
                    )
                    n_adc += 1
                    n_rd_points += n_rd
                elif case == 'b':
                    g_rd_amp = np.array([gradients_b[ii - 1, 0], gradients_b[ii, 0], gradients_b[ii, 0]]) * hw.gammaB
                    g_ph_amp = np.array([gradients_b[ii - 1, 1], gradients_b[ii, 1], gradients_b[ii, 1]]) * hw.gammaB
                    g_sl_amp = np.array([gradients_b[ii - 1, 2], gradients_b[ii, 2], gradients_b[ii, 2]]) * hw.gammaB
                    g_time = np.array([0.0, hw.grad_rise_time, self.repetitionTime])
                    batches[batch_name].add_block(
                        pp.make_extended_trapezoid(channel=rd_channel,
                                                   amplitudes=g_rd_amp,
                                                   times=g_time),
                        pp.make_extended_trapezoid(channel=ph_channel,
                                                   amplitudes=g_ph_amp,
                                                   times=g_time),
                        pp.make_extended_trapezoid(channel=sl_channel,
                                                   amplitudes=g_sl_amp,
                                                   times=g_time),
                        block_rf_b,
                        block_adc_b
                    )
                    n_adc += 1
                    n_rd_points += n_rd

            # Write sequence file and interpret the sequence to get the waveform
            batches[batch_name].write(batch_name + ".seq")
            waveforms[batch_name], param_dict = flo_interpreter.interpret(batch_name + ".seq")
            print(f"{batch_name}.seq ready!")
            n_rd_points_dict[batch_name] = n_rd_points

            return waveforms, n_rd_points_dict, n_adc

        '''
        Step 8: Run the batches
        This step will handle the different batches, run it and get the resulting data. This should not be modified.
        Oversampled data will be available in self.mapVals['data_over']
        Decimated data will be available in self.mapVals['data_decimated']
        The decimated data is shifted to account for CIC delay, so data is synchronized with real-time signal
        '''

        # Create batches
        waveforms_a, n_readouts_a, n_adc_a = create_batches(case='a')
        waveforms_b, n_readouts_b, n_adc_b = create_batches(case='b')

        # Run sequence a
        if self.runBatches(waveforms=waveforms_a,
                           n_readouts=n_readouts_a,
                           n_adc=n_adc_a,
                           frequency=self.mapVals['larmorFreq'],  # MHz
                           bandwidth=bw_a,  # MHz
                           decimate='Normal',
                           hardware=True,
                           output='a',
                           angulation=self.angulation,
                           ):
            pass
        else:
            return False

        # Run sequence b
        return self.runBatches(waveforms=waveforms_b,
                               n_readouts=n_readouts_b,
                               n_adc=n_adc_b,
                               frequency=self.mapVals['larmorFreq'],  # MHz
                               bandwidth=bw_b,  # MHz
                               decimate='Normal',
                               hardware=True,
                               output='b',
                               angulation=self.angulation,
                               )

    def sequenceAnalysis(self, mode=None):
        """
        Analyzes the data acquired from the SPDS (Single Point Double Shot) sequence to estimate the B₀ map,
        generate k-space and spatial domain images, and prepare the outputs for visualization.

        Parameters:
        -----------
        mode : str, optional
            Execution mode of the analysis. If set to 'Standalone', the results are plotted immediately
            after analysis (default: None).

        Outputs:
        --------
        - `output` (list): A list of dictionaries defining the data and parameters for visualization.
          Includes:
            - Spatial domain magnitude images for channels A and B.
            - B₀ field map.
            - k-space magnitude images for channels A and B.
        - Updates `self.mapVals` with intermediate results, including k-space, spatial images, and the
          B₀ field map.
        - If `mode == 'Standalone'`, plots the results.

        Notes:
        ------
        - Assumes that the k-space mask and orientation settings are correctly preconfigured.
        """

        def zero_padding(data, order):
            original_shape = data.shape
            if len(original_shape) == 3:
                if original_shape[0] == 1:
                    new_shape = (1, original_shape[1] * order, original_shape[2] * order)
                else:
                    new_shape = tuple(dim * order for dim in original_shape)
            else:
                raise ValueError("Error of matrix shape")

            k_dataZP_a = np.zeros(new_shape, dtype=data.dtype)
            start_indices = tuple((new_dim - old_dim) // 2 for new_dim, old_dim in zip(new_shape, original_shape))
            end_indices = tuple(start + old_dim for start, old_dim in zip(start_indices, original_shape))
            if original_shape[0] == 1:
                k_dataZP_a[0, start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]] = data[0]
            else:
                k_dataZP_a[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1],
                start_indices[2]:end_indices[2]] = data

            return k_dataZP_a

        # Pass mode to the self, it will be required by the mriBlankSeq
        self.mode = mode

        # Load data
        data_a = self.mapVals['data_decimated_a']
        data_b = self.mapVals['data_decimated_b']
        k_points = self.mapVals['k_cartesian']
        mask = self.mask

        # Delete the addRdPoints and last readout
        data_a = np.reshape(data_a, (-1, 1 + 2 * hw.addRdPoints))
        data_b = np.reshape(data_b, (-1, 1 + 2 * hw.addRdPoints))
        data_a = data_a[0:-1, hw.addRdPoints]
        data_b = data_b[0:-1, hw.addRdPoints]

        # Fill k_space
        k_data_a = np.zeros(np.size(k_points, 0), dtype=complex)
        k_data_b = np.zeros(np.size(k_points, 0), dtype=complex)
        jj = 0
        for ii in range(np.size(mask)):
            if mask[ii]:
                k_data_a[ii] = data_a[jj]
                k_data_b[ii] = data_b[jj]
                jj += 1

        # Get images
        k_data_aRaw = (np.reshape(k_data_a, (self.nPoints[2], self.nPoints[1], self.nPoints[0])))
        k_data_bRaw = (np.reshape(k_data_b, (self.nPoints[2], self.nPoints[1], self.nPoints[0])))
        k_data_a = zero_padding(k_data_aRaw, self.mapVals['interpOrder'])
        k_data_b = zero_padding(k_data_bRaw, self.mapVals['interpOrder'])

        i_data_a = run_ifft(k_data_a)
        i_data_b = run_ifft(k_data_b)
        self.mapVals['space_k_a'] = k_data_a
        self.mapVals['space_k_b'] = k_data_b
        self.mapVals['space_i_a'] = i_data_a
        self.mapVals['space_i_b'] = i_data_b

        # Plots in GUI
        if self.nPoints[2] == 1:
            i_data_a = np.squeeze(i_data_a)
            i_data_b = np.squeeze(i_data_b)

            # Generate mask
            p_max = np.max(np.abs(i_data_a))
            mask = np.abs(i_data_a) < p_max * self.mapVals['thresholdMask']/100

            # Get phase
            RawPhase1 = np.angle(i_data_a)
            RawPhase1[mask] = 0
            RawPhase2 = np.angle(i_data_b)
            RawPhase2[mask] = 0

            i_phase_a = unwrap(RawPhase1)
            i_phase_b = unwrap(RawPhase2)

            # Get magnetic field
            b_field = ((i_phase_b - i_phase_a) / (2 * np.pi * hw.gammaB * (self.deadTime[1] - self.deadTime[0])))
            b_field[mask] = 0
            self.mapVals['b_field'] = b_field

            NX = self.nPoints[0]*self.mapVals['interpOrder']
            NY = self.nPoints[1]*self.mapVals['interpOrder']
            dx = self.fov[0] / NX
            dy = self.fov[1] / NY

            # Here we define the grid of the full FOV and select the indexs where B0 is no null
            ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
            condition = b_field != 0
            ii = ii[condition]
            jj = jj[condition]

            # Here we define the coordinates of the FOV where the B0 is no null
            x = (-(NX - 1) / 2 + ii) * dx
            y = (-(NY - 1) / 2 + jj) * dy

            # Store in values the B0 value in the indexs that accomplishes B0 different of 0
            values = b_field[condition]

            # Save in mapList all the {╥x,y,z,B0} data where B0 is no null
            mapList = np.column_stack((x, y, values))
            self.mapVals['mapList']=mapList

            # And now we proceed with the fitting
            x_fit = mapList[:, 0]
            y_fit = mapList[:, 1]
            B_fit = mapList[:, 2]
            degree = self.mapVals['fittingOrder']
            poly = PolynomialFeatures(degree)
            coords = np.vstack((x_fit, y_fit)).T  # Combinamos x, y como entrada para PolynomialFeatures
            X_poly = poly.fit_transform(coords)  # Genera todas las combinaciones polinómicas hasta grado 4
            coeffs, _, _, _ = lstsq(X_poly, B_fit, rcond=None)
            terms = terms = poly.powers_
            polynomial_expression = ""
            for i, coeff in enumerate(coeffs):
                if coeff != 0:  # Ignore null coefficients
                    powers = terms[i]  # Powers (x**i, y**j)
                    term = f"{coeff}"
                    if any(powers):
                        if powers[0] > 0:
                            term += f"*(x**{powers[0]})"
                        if powers[1] > 0:
                            term += f"*(y**{powers[1]})"
                    polynomial_expression += f" + {term}" if coeff > 0 and i > 0 else f" {term}"

            print("B0 fitting:")
            print(polynomial_expression)

            result1 = {}
            result1['widget'] = 'image'
            result1['data'] = np.real(b_field.reshape(1, NX, NY))
            result1['xLabel'] = "xx"
            result1['yLabel'] = "xx"
            result1['title'] = "B0 field"
            result1['row'] = 0
            result1['col'] = 3

            result4 = {}
            result4['widget'] = 'image'
            result4['data'] = np.real(RawPhase1.reshape(1, NX, NY))
            result4['xLabel'] = "xx"
            result4['yLabel'] = "xx"
            result4['title'] = "Raw Phase Image Td1"
            result4['row'] = 0
            result4['col'] = 1

            result5 = {}
            result5['widget'] = 'image'
            result5['data'] = np.real(RawPhase2.reshape(1, NX, NY))
            result5['xLabel'] = "xx"
            result5['yLabel'] = "xx"
            result5['title'] = "Raw Phase Image Td2"
            result5['row'] = 1
            result5['col'] = 1

            result2 = {}
            result2['widget'] = 'image'
            result2['data'] = np.real(i_phase_a.reshape(1, NX, NY))
            result2['xLabel'] = "xx"
            result2['yLabel'] = "xx"
            result2['title'] = "Unwrapped Phase Image Td1"
            result2['row'] = 0
            result2['col'] = 2

            result3 = {}
            result3['widget'] = 'image'
            result3['data'] = np.real(i_phase_b.reshape(1, NX, NY))
            result3['xLabel'] = "xx"
            result3['yLabel'] = "xx"
            result3['title'] = "Unwrapped Phase Image Td2"
            result3['row'] = 1
            result3['col'] = 2

            result6 = {}
            result6['widget'] = 'image'
            result6['data'] = np.abs(i_data_a.reshape(1, NX, NY))
            result6['xLabel'] = "xx"
            result6['yLabel'] = "xx"
            result6['title'] = "Raw Abs Image Td1"
            result6['row'] = 0
            result6['col'] = 0

            result7 = {}
            result7['widget'] = 'image'
            result7['data'] = np.abs(i_data_b.reshape(1, NX, NY))
            result7['xLabel'] = "xx"
            result7['yLabel'] = "xx"
            result7['title'] = "Raw Abs Image Td2"
            result7['row'] = 1
            result7['col'] = 0

            self.output = [result1, result2, result3, result4, result5, result6, result7]

        if self.nPoints[0] > 1 and self.nPoints[1] > 1 and self.nPoints[2] > 1:
            # Generate mask
            p_max = np.max(np.abs(i_data_a))
            mask = np.abs(i_data_a) < p_max * self.mapVals['thresholdMask']/100

            # Get phase
            RawPhase1 = np.angle(i_data_a)
            RawPhase1[mask] = 0
            RawPhase2 = np.angle(i_data_b)
            RawPhase2[mask] = 0

            i_phase_a = unwrap(RawPhase1)
            i_phase_b = unwrap(RawPhase2)

            # Get magnetic field
            b_field = -(i_phase_b - i_phase_a) / (2 * np.pi * hw.gammaB * (self.deadTime[1] - self.deadTime[0]))
            b_field[mask] = 0
            self.mapVals['b_field'] = b_field
            B0mapReorganized = np.flip(np.flip(np.flip(np.transpose(b_field, (2, 1, 0)), axis=0), axis=1), axis=2)
            self.mapVals['B0mapReorganized'] = B0mapReorganized

            NX = self.nPoints[0]*self.mapVals['interpOrder']
            NY = self.nPoints[1] * self.mapVals['interpOrder']
            NZ = self.nPoints[2] * self.mapVals['interpOrder']
            dx = self.fov[0] / NX
            dy = self.fov[1] / NY
            dz = self.fov[2] / NZ

            mapList = []
            cont = 0

            for ii in range(NX):
                for jj in range(NX):
                    for kk in range(NX):
                        if B0mapReorganized[ii, jj, kk] != 0:
                            z_coord = (-(NZ - 1) / 2 + kk) * dz
                            y_coord = (-(NY - 1) / 2 + jj) * dy
                            x_coord = (-(NX - 1) / 2 + ii) * dx
                            value = B0mapReorganized[ii, jj, kk]

                            mapList.append([x_coord, y_coord, z_coord, value])
                            cont += 1

            mapList = np.array(mapList)

            # And now we proceed with the fitting
            x_fit = mapList[:, 0]
            y_fit = mapList[:, 1]
            z_fit = mapList[:, 2]
            B_fit = mapList[:, 3]
            degree = self.mapVals['fittingOrder']
            poly = PolynomialFeatures(degree)
            coords = np.vstack((x_fit, y_fit, z_fit)).T
            X_poly = poly.fit_transform(coords)
            coeffs, _, _, _ = lstsq(X_poly, B_fit, rcond=None)
            terms = poly.powers_
            polynomial_expression = ""
            polynomial_expressionGUI = ""
            for i, coeff in enumerate(coeffs):
                if coeff != 0:  # Ignore null coefficients
                    powers = terms[i]  # Powers (x**i, y**j, z**k)
                    term = f"{coeff}"
                    termGUI = f"{coeff}"
                    if any(powers):
                        if powers[2] > 0:
                            term += f"*(z**{powers[2]})"
                            termGUI += f"*(z^{powers[2]})"
                        if powers[1] > 0:
                            term += f"*(y**{powers[1]})"
                            termGUI += f"*(y^{powers[1]})"
                        if powers[0] > 0:
                            term += f"*(x**{powers[0]})"
                            termGUI += f"*(x^{powers[0]})"
                    polynomial_expression += f" + {term}" if coeff > 0 and i > 0 else f" {term}"
                    polynomial_expressionGUI += f" + {termGUI}" if coeff > 0 and i > 0 else f" {termGUI}"

            print("B0 fitting:")
            print(polynomial_expressionGUI)

            result1 = {}
            result1['widget'] = 'image'
            result1['data'] = np.real(b_field)
            result1['xLabel'] = "xx"
            result1['yLabel'] = "xx"
            result1['title'] = "B0 field"
            result1['row'] = 0
            result1['col'] = 3

            result4 = {}
            result4['widget'] = 'image'
            result4['data'] = np.real(RawPhase1)
            result4['xLabel'] = "xx"
            result4['yLabel'] = "xx"
            result4['title'] = "Raw Phase Image Td1"
            result4['row'] = 0
            result4['col'] = 1

            result5 = {}
            result5['widget'] = 'image'
            result5['data'] = np.real(RawPhase2)
            result5['xLabel'] = "xx"
            result5['yLabel'] = "xx"
            result5['title'] = "Raw Phase Image Td2"
            result5['row'] = 1
            result5['col'] = 1

            result2 = {}
            result2['widget'] = 'image'
            result2['data'] = np.real(i_phase_a)
            result2['xLabel'] = "xx"
            result2['yLabel'] = "xx"
            result2['title'] = "Unwrapped Phase Image Td1"
            result2['row'] = 0
            result2['col'] = 2

            result3 = {}
            result3['widget'] = 'image'
            result3['data'] = np.real(i_phase_b)
            result3['xLabel'] = "xx"
            result3['yLabel'] = "xx"
            result3['title'] = "Unwrapped Phase Image Td2"
            result3['row'] = 1
            result3['col'] = 2

            result6 = {}
            result6['widget'] = 'image'
            result6['data'] = np.abs(i_data_a)
            result6['xLabel'] = "xx"
            result6['yLabel'] = "xx"
            result6['title'] = "Raw Abs Image Td1"
            result6['row'] = 0
            result6['col'] = 0

            result7 = {}
            result7['widget'] = 'image'
            result7['data'] = np.abs(i_data_b)
            result7['xLabel'] = "xx"
            result7['yLabel'] = "xx"
            result7['title'] = "Raw Abs Image Td2"
            result7['row'] = 1
            result7['col'] = 0

            self.output = [result1, result2, result3, result4, result5, result6, result7]

        # save data once self.output is created
        self.saveRawData()

        # Export in txt the model fitted
        if not os.path.exists('b0_maps'):
            os.makedirs('b0_maps')
        output_file = "b0_maps/"+self.mapVals['fileName'][:-4]+".txt"
        with open(output_file, "w") as f:
            f.write(polynomial_expression + "\n")
        print(f"Fitting exported to '{output_file}'")

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        return self.output

if __name__ == "__main__":
    seq = spds()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')