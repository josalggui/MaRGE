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
import controller.experiment_gui as ex
import configs.hw_config as hw  # Import the scanner hardware config
import configs.units as units
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from marga_pulseq.interpreter import PSInterpreter  # Import the marga_pulseq interpreter
import pypulseq as pp  # Import PyPulseq


# Template Class for MRI Sequences
class spds(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        """
        Defines the parameters for the sequence.
        """
        super(spds, self).__init__()

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
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM',
                          tip='Number of repetitions of the full scan.')
        self.addParameter(key='FreqOffset', string='Frequency offset (kHz)', val=0, units=units.kHz, field='RF',
                          tip='Frequency offset respect Larmor.')
        self.addParameter(key='rfExFA', string='Excitation Flip Angle (degrees)', val=90.0, field='RF',
                          tip="Flip angle of the excitation RF pulse in degrees")
        self.addParameter(key='rfExTime', string='Excitation time (us)', val=100.0, units=units.us, field='RF',
                          tip="Duration of the RF excitation pulse in microseconds (us).")
        self.addParameter(key='nPoints', string='Matrix size [rd, ph, sl]', val=[2, 2, 2], field='IM',
                          tip='Matrix size for the acquired images.')
        self.addParameter(key='fov', string='Field of View (cm)', val=[20.0, 20.0, 20.0], field='IM',
                          tip='Field of View (cm).')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[2, 1, 0], field='IM',
                          tip="0=x, 1=y, 2=z")
        self.addParameter(key='repetitionTime', string='Repetition Time (ms)', val=5.0, units=units.ms, field='SEQ',
                          tip="The time between successive excitation pulses, in milliseconds (ms).")
        self.addParameter(key='deadTime', string='Dead times (us)', val=[100.0, 200.0], units=units.us, field='SEQ',
                          tip='Dead time for the two acquisitions in microseconds (us).')
        self.addParameter(key='dummyPulses', string='Number of dummy pulses', val=1, field='SEQ',
                          tip='Number of dummy pulses at the beginning of each batch.')
        self.addParameter(key='shimming', string='Shimming', val=[0.0, 0.0, 0.0], units=units.sh, field='SEQ',
                          tip='Shimming parameter to compensate B0 linear inhomogeneity.')

    def sequenceInfo(self):
        """
        Description of the sequence. Students should customize this.
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
        Students can extend this method as needed.
        """

        k_max = np.array(self.mapVals['nPoints']) / (2 * np.array(self.mapVals['fov']) * 1e-2)
        kx = np.linspace(start=-1, stop=1, endpoint=False, num=self.nPoints[0])
        ky = np.linspace(start=-1, stop=1, endpoint=False, num=self.nPoints[1])
        kz = np.linspace(start=-1, stop=1, endpoint=False, num=self.nPoints[2])
        ky, kz, kx = np.meshgrid(ky, kz, kx)
        k_norm = np.zeros(shape=(np.size(kx), 3))
        k_norm[:, 0] = np.reshape(kx, -1)
        k_norm[:, 1] = np.reshape(ky, -1)
        k_norm[:, 2] = np.reshape(kz, -1)
        distance = np.sqrt(np.sum(k_norm ** 2, axis=1))
        k_cartesian = np.zeros_like(k_norm)
        k_cartesian[:, 0] = k_norm[:, 0] * k_max[0]
        k_cartesian[:, 1] = k_norm[:, 1] * k_max[1]
        k_cartesian[:, 2] = k_norm[:, 2] * k_max[2]
        self.mask = distance <= 1
        n = np.sum(self.mask)

        tr = self.mapVals['repetitionTime'] * 1e-3  # s
        time = tr * n / 60 * 2 * self.mapVals['nScans']  # minutes

        return time  # minutes

    def sequenceAtributes(self):
        """
        Additional sequence attributes or parameters.
        Extend this method with specific calculations or modifications.
        """
        super().sequenceAtributes()

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
            grad_raster_time=10e-6,  # Gradient raster time (s)
            rise_time=hw.grad_rise_time,  # Gradient rise time (s)
            rf_raster_time=1e-6,
            block_duration_raster=1e-6
        )

        '''
        Step 3: Perform any calculations required for the sequence.
        In this step, students can implement the necessary calculations, such as timing calculations, RF amplitudes, and
        gradient strengths, before defining the sequence blocks.
        '''

        # Get k-space info
        dk = 1 / self.fov
        k_max = self.nPoints / (2 * self.fov)
        self.mapVals['dk'] = dk
        self.mapVals['k_max'] = k_max

        # Get bandwidths and acquisition window
        bw_a = np.max(self.nPoints) / (2 * self.deadTime[0])  # Hz
        bw_b = np.max(self.nPoints) / (2 * self.deadTime[1])  # Hz
        hw.addRdPoints = 1
        n_rd = 1 + 2 * hw.addRdPoints

        # Get timing parameters
        time_acq_a = n_rd / bw_a  # s
        time_acq_b = n_rd / bw_b  # s
        time_delay0_a = self.rfExTime / 2 + self.deadTime[0] + time_acq_a / 2
        time_delay0_b = self.rfExTime / 2 + self.deadTime[1] + time_acq_b / 2
        time_grad_a = self.repetitionTime - time_delay0_a - hw.grad_rise_time - self.rfExTime / 2
        time_grad_b = self.repetitionTime - time_delay0_b - hw.grad_rise_time - self.rfExTime / 2

        # Get cartesian points
        kx = np.linspace(start=-1, stop=1, endpoint=False, num=self.nPoints[0])
        ky = np.linspace(start=-1, stop=1, endpoint=False, num=self.nPoints[1])
        kz = np.linspace(start=-1, stop=1, endpoint=False, num=self.nPoints[2])
        ky, kz, kx = np.meshgrid(ky, kz, kx)
        k_norm = np.zeros(shape=(np.size(kx), 3))
        k_norm[:, 0] = np.reshape(kx, -1)
        k_norm[:, 1] = np.reshape(ky, -1)
        k_norm[:, 2] = np.reshape(kz, -1)
        distance = np.sqrt(np.sum(k_norm ** 2, axis=1))
        k_cartesian = np.zeros_like(k_norm)
        k_cartesian[:, 0] = k_norm[:, 0] * k_max[0]
        k_cartesian[:, 1] = k_norm[:, 1] * k_max[1]
        k_cartesian[:, 2] = k_norm[:, 2] * k_max[2]
        self.mask = distance <= 1
        self.mapVals['k_cartesian'] = k_cartesian

        # Get gradients
        gradients_a = k_cartesian / (hw.gammaB * self.deadTime[0])
        gradients_b = k_cartesian / (hw.gammaB * self.deadTime[1])
        gradients_a = gradients_a[self.mask]
        gradients_b = gradients_b[self.mask]
        self.mapVals['gradients_a'] = gradients_a
        self.mapVals['gradients_b'] = gradients_b
        gradients_a = np.vstack([[0.0, 0.0, 0.0], gradients_a])
        gradients_b = np.vstack([[0.0, 0.0, 0.0], gradients_b])

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
                lo_freq=hw.larmorFreq,  # Larmor frequency in MHz
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
                lo_freq=hw.larmorFreq,  # Larmor frequency in MHz
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
            delay=self.repetitionTime - time_acq_a,
        )
        block_adc_b = pp.make_adc(
            num_samples=n_rd,
            dwell=1 / bw_b * 1e-6,
            delay=self.repetitionTime - time_acq_b,
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
                               frequency=hw.larmorFreq,  # MHz
                               bandwidth=bw_a,  # MHz
                               decimate='Normal',
                               hardware=True,
                               output='a'
                               ):
            pass
        else:
            return False

        # Run sequence b
        return self.runBatches(waveforms=waveforms_b,
                               n_readouts=n_readouts_b,
                               n_adc=n_adc_b,
                               frequency=hw.larmorFreq,  # MHz
                               bandwidth=bw_b,  # MHz
                               decimate='Normal',
                               hardware=True,
                               output='b'
                               )

    def sequenceAnalysis(self, mode=None):
        # Pass mode to the self, it will be required by the mriBlankSeq
        self.mode = mode

        # Load data
        data_a = self.mapVals['data_decimated_a']
        data_b = self.mapVals['data_decimated_b']
        k_points = self.mapVals['k_cartesian']
        mask = self.mask

        # Delete the addRdPoints
        data_a = np.reshape(data_a, (-1, 1 + 2 * hw.addRdPoints))
        data_b = np.reshape(data_b, (-1, 1 + 2 * hw.addRdPoints))
        data_a = data_a[:, hw.addRdPoints]
        data_b = data_b[:, hw.addRdPoints]

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
        k_data_a = np.reshape(k_data_a, (self.nPoints[2], self.nPoints[1], self.nPoints[0]))
        k_data_b = np.reshape(k_data_b, (self.nPoints[2], self.nPoints[1], self.nPoints[0]))
        i_data_a = self.runIFFT(k_data_a)
        i_data_b = self.runIFFT(k_data_b)

        # Get phase
        i_phase_a = np.angle(i_data_a)
        i_phase_b = np.angle(i_data_b)

        # Get magnetic field
        b_field = (i_phase_b - i_phase_a) / (2 * np.pi * hw.gammaB * (self.deadTime[1] - self.deadTime[0]))

        # Create the outputs to be plotted
        output_0 = self.fix_image_orientation(np.abs(i_data_a), axes=self.axesOrientation)
        output_0['row'] = 0
        output_0['col'] = 0

        # Create the outputs to be plotted
        output_1 = self.fix_image_orientation(np.abs(i_data_b), axes=self.axesOrientation)
        output_1['row'] = 0
        output_1['col'] = 1

        # Create the outputs to be plotted
        output_2 = self.fix_image_orientation(b_field, axes=self.axesOrientation)
        output_2['row'] = 0
        output_2['col'] = 2

        # create self.out to run in iterative mode
        self.output = [output_0, output_1, output_2]

        # save data once self.output is created
        self.saveRawData()

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        return self.output


if __name__ == "__main__":
    seq = spds()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=True, demo=True, standalone=True)
    # seq.sequenceAnalysis(mode='Standalone')