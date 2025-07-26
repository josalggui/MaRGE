"""
Created on Tuesday, September 17th 2024
@author: Prof. Dr. Maxim Zaitsev, Department of Diagnostic and Interventional Radiology, University of Freiburg, Germany
@author: Dr. J.M. Algarín, MRILab, i3M, CSIC, Valencia, Spain
@Summary: mse sequence class coded with pypulseq compatible with MaRGE
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

import math

import numpy as np
import scipy.signal as sig

import pypulseq as pp

import marge.marcos.marcos_client.experiment
import marge.configs.hw_config as hw
import marge.configs.units as units
import marge.seq.mriBlankSeq as blankSeq
from marga_pulseq.interpreter import PSInterpreter


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class MSE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(MSE, self).__init__()

        # Input parameters
        self.nScans = None
        self.shimming = None
        self.dummyPulses = None
        self.freqOffset = None
        self.rfReFA = None
        self.rfExFA = None
        self.rfReTime = None
        self.rfExTime = None
        self.rdGradTime = None
        self.acqTime = None
        self.repetitionTime = None
        self.echoSpacing = None
        self.etl = None
        self.nPoints = None
        self.fov = None
        self.axesOrientation = None
        self.addParameter(key='seqName', string='MSEInfo', val='MSE_PyPulseq')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=False)
        self.addParameter(key='pypulseq', string='PyPulseq', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, units=units.kHz,
                          field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=60.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=120.0, units=units.us, field='RF')
        self.addParameter(key='deadTime', string='Dead time (us)', val=20.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, units=units.ms, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=40., units=units.ms, field='SEQ')
        self.addParameter(key='fov', string='FOV[x,y,z] (cm)', val=[55.0, 12.0, 12.0], units=units.cm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM',
                          tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[40, 2, 1], field='IM')
        self.addParameter(key='etl', string='Echo train length', val=2, field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=2.0, units=units.ms, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[2, 1, 0], field='IM',
                          tip="0=x, 1=y, 2=z")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=3.0, units=units.ms, field='OTH')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=0, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='unlock_orientation', string='Unlock image orientation', val=0, field='OTH',
                          tip='0: Images oriented according to standard. 1: Image raw orientation')
        self.addParameter(key='preemphasis', string='Preemphasis', val=1.0, field='OTH')
        self.addParameter(key='fsp_r', string='fsp_r', val=1.0, field='OTH')

    def sequenceInfo(self):
        print("3D MSE sequence with PyPulseq")
        print("Author: Prof. Dr. Maxim Zaitsev")
        print("University of Freiburg, Germany")
        print("Author: Dr. José Miguel Algarín")
        print("mriLab @ i3M, CSIC, Spain \n")

    def sequenceTime(self):
        nRD, nPH, nSL = np.array(self.mapVals["nPoints"])
        repetition_time = self.mapVals["repetitionTime"] * 1e-3
        nScans = self.mapVals["nScans"]
        scan_time = nScans * nPH * nSL * repetition_time / 60  # minutes
        scan_time = np.round(scan_time, decimals=1)
        return scan_time  # minutes

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        """
        Runs a multi-spin echo (MSE) sequence using PyPulseq to control the pulse sequence timing and hardware settings.

        This method initiates the running of an MSE sequence, handling various hardware and sequence parameters such as
        field of view (FOV), resolution, readout gradients, and timing. It prepares and configures the experiment for data
        acquisition, creates pulse sequence batches, and can either run the experiment or plot the sequence.

        Parameters
        ----------
        plotSeq : int, optional
            If set to 1, the sequence will be plotted instead of executed. Defaults to 0.

        demo : bool, optional
            If True, the method will simulate the sequence execution in demo mode without actual hardware. Defaults to False.

        standalone : bool, optional
            If True, the method will run in standalone mode and plot the sequence. Defaults to False.

        Key Parameters Used
        -------------------
        - `self.fov`: Field of view (FOV) for the imaging sequence (X, Y, Z).
        - `self.nPoints`: Number of readout points in each direction (RD, PH, SL).
        - `self.etl`: Echo train length (number of echoes in the sequence).
        - `self.echoSpacing`: Time between echoes (TE).
        - `self.repetitionTime`: Repetition time (TR) between sequences.
        - `self.acqTime`: Acquisition time during readout.
        - `self.rfExFA`: Flip angle for excitation pulse.
        - `self.rfReFA`: Flip angle for refocusing pulse.
        - `self.dummyPulses`: Number of dummy pulses to add before the actual acquisition starts.
        - `self.shimming`: Shimming parameters for adjusting magnetic field homogeneity.

        Key Process Steps
        -----------------
        1. Initializes the experiment and configures the system parameters, such as bandwidth and gradient times.
        2. Defines the readout, phase, and slice gradients according to the `axesOrientation`.
        3. Creates RF excitation and refocusing pulses.
        4. Prepares and adds sequence blocks (RF pulses, gradients, and delays).
        5. Creates and processes batches of the sequence, performing slice and phase encoding sweeps.
        6. Executes the pulse sequence or plots it based on the provided arguments.
        7. Handles data acquisition and decimation for oversampled data.

        Returns
        -------
        bool
            True if the sequence execution or plotting was successful, False if there were errors in configuration or timing.

        Raises
        ------
        RuntimeError
            Raised if the sequence timing is incorrect or the hardware configuration is out of bounds.

        Notes
        -----
        - The method assumes access to global hardware settings (`hw`) and a `Experiment` object (`expt`).
        - In case of sequence plotting, the `plotSeq` argument should be set, and the sequence will be visualized
          instead of executed.
        """

        print("Run MSE powered by PyPulseq")
        init_gpa = False
        self.demo = demo

        # Define the interpreter. It should be updated on calibration
        self.flo_interpreter = PSInterpreter(
            tx_warmup=hw.blkTime,  # Transmit chain warm-up time (us)
            rf_center=hw.larmorFreq * 1e6,  # Larmor frequency (Hz)
            rf_amp_max=hw.b1Efficiency / (2 * np.pi) * 1e6,  # Maximum RF amplitude (Hz)
            gx_max=hw.gFactor[0] * hw.gammaB,  # Maximum gradient amplitude for X (Hz/m)
            gy_max=hw.gFactor[1] * hw.gammaB,  # Maximum gradient amplitude for Y (Hz/m)
            gz_max=hw.gFactor[2] * hw.gammaB,  # Maximum gradient amplitude for Z (Hz/m)
            grad_max=np.max(np.abs(hw.gFactor)) * hw.gammaB,  # Maximum gradient amplitude (Hz/m)
            grad_t=hw.grad_raster_time * 1e6,  # Gradient raster time (us)
        )

        # Define system properties according to hw_config file
        self.system = pp.Opts(
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

        # Get Parameters
        self.dfov = self.getFovDisplacement()
        self.dfov = self.dfov[self.axesOrientation]
        self.fov = self.fov[self.axesOrientation]
        resolution = self.fov / self.nPoints
        self.mapVals['resolution'] = resolution
        fov_mm = self.fov * 1e3
        nRD, nPH, nSL = self.nPoints  # this is actually nRd, nPh and nSl, axes given by axesOrientation
        n_echo = self.etl
        TE = self.echoSpacing
        TR = self.repetitionTime
        dG = hw.grad_rise_time
        self.mapVals['grad_rise_time'] = hw.grad_rise_time * 1e6
        sampling_time = self.acqTime
        if self.rdGradTime >= self.acqTime:
            ro_flattop_add = (self.rdGradTime - self.acqTime) / 2
        else:
            print("ERROR: readout gradient time must be longer than acquisition time.")
            return False
        nRD_pre = hw.addRdPoints
        nRD_post = hw.addRdPoints
        self.mapVals['nRD_pre'] = hw.addRdPoints
        self.mapVals['nRD_post'] = hw.addRdPoints
        n_rd_points_per_train = n_echo * (nRD + nRD_post + nRD_pre)
        os = hw.oversamplingFactor
        self.mapVals['oversamplingFactor'] = os
        t_ex = self.rfExTime
        t_ref = self.rfReTime
        fsp_r = self.fsp_r  # Not sure about what this parameter does.
        fsp_s = 0.5  # Not sure about what this parameter does. It is not used in the code.

        # Derived and modified parameters
        fov = np.array(fov_mm) * 1e-3
        TE = round(
            TE / self.system.grad_raster_time / 2) * self.system.grad_raster_time * 2  # TE (=ESP) should be divisible to a double gradient raster, which simplifies calcuations
        ro_flattop_time = sampling_time + 2 * ro_flattop_add
        rf_add = math.ceil(max(self.system.rf_dead_time,
                               self.system.rf_ringdown_time) / self.system.grad_raster_time) * self.system.grad_raster_time  # round up dead times to the gradient raster time to enable correct TE & ESP calculation
        t_sp = round(
            (0.5 * (
                    TE - ro_flattop_time - t_ref) - rf_add) / self.system.grad_raster_time) * self.system.grad_raster_time
        t_spex = round(
            (0.5 * (TE - t_ex - t_ref) - rf_add) / self.system.grad_raster_time) * self.system.grad_raster_time
        rf_ex_phase = np.pi / 2
        rf_ref_phase = 0

        # Map the axis to "x", "y", and "z" according ot axesOrientation
        axes_map = {0: "x", 1: "y", 2: "z"}
        rd_channel = axes_map.get(self.axesOrientation[0], "")
        ph_channel = axes_map.get(self.axesOrientation[1], "")
        sl_channel = axes_map.get(self.axesOrientation[2], "")

        # ======
        # CREATE EVENTS
        # ======
        flip_ex = self.rfExFA * np.pi / 180
        rf_ex = pp.make_block_pulse(
            flip_angle=flip_ex,
            system=self.system,
            duration=t_ex,
            delay=rf_add,
            phase_offset=rf_ex_phase,
        )
        d_ex = pp.make_delay(t_ex + rf_add * 2)

        flip_ref = self.rfReFA * np.pi / 180
        rf_ref = pp.make_block_pulse(
            flip_angle=flip_ref,
            system=self.system,
            duration=t_ref,
            delay=rf_add,
            phase_offset=rf_ref_phase,
            use="refocusing",
        )
        d_ref = pp.make_delay(t_ref + rf_add * 2)

        delta_krd = 1 / fov[0]
        ro_amp = nRD * delta_krd / sampling_time

        gr_acq = pp.make_trapezoid(
            channel=rd_channel,
            system=self.system,
            amplitude=ro_amp,
            flat_time=ro_flattop_time,
            delay=t_sp,
            rise_time=dG,
        )
        adc = pp.make_adc(
            num_samples=(nRD_pre + nRD + nRD_post) * os, dwell=sampling_time / nRD / os,
            delay=0.5 * (TE - t_ref - (nRD + nRD_post + nRD_pre) * sampling_time / nRD) - rf_add
        )
        gr_spr = pp.make_trapezoid(
            channel=rd_channel,
            system=self.system,
            area=gr_acq.area * fsp_r,
            duration=t_sp,
            rise_time=dG,
        )

        agr_spr = gr_spr.area
        agr_preph = gr_acq.area / 2 + agr_spr
        gr_preph = pp.make_trapezoid(
            channel=rd_channel, system=self.system, area=agr_preph, duration=0.0018, rise_time=dG
        )
        delay_preph = pp.make_delay(t_spex)
        # Phase-encoding
        delta_kph = 1 / fov[1]
        gp_max = pp.make_trapezoid(
            channel=ph_channel,
            system=self.system,
            area=delta_kph * nPH / 2,
            duration=t_sp,
            rise_time=dG,
        )
        delta_ksl = 1 / fov[2]
        gs_max = pp.make_trapezoid(
            channel=sl_channel,
            system=self.system,
            area=delta_ksl * nSL / 2,
            duration=t_sp,
            rise_time=dG,
        )

        # combine parts of the read gradient
        gc_times = np.array(
            [
                0,
                gr_spr.rise_time,
                gr_spr.flat_time,
                gr_spr.fall_time,
                gr_acq.flat_time,
                gr_spr.fall_time,
                gr_spr.flat_time,
                gr_spr.rise_time,
            ])
        gc_times = np.cumsum(gc_times)

        gr_amp = np.array([0, gr_spr.amplitude, gr_spr.amplitude, gr_acq.amplitude, gr_acq.amplitude, gr_spr.amplitude,
                           gr_spr.amplitude, 0])
        gr = pp.make_extended_trapezoid(channel=rd_channel, times=gc_times, amplitudes=gr_amp)

        gp_amp = np.array([0, gp_max.amplitude, gp_max.amplitude, 0, 0, -gp_max.amplitude, -gp_max.amplitude, 0])
        gp_max = pp.make_extended_trapezoid(channel=ph_channel, times=gc_times, amplitudes=gp_amp)

        gs_amp = np.array([0, gs_max.amplitude, gs_max.amplitude, 0, 0, -gs_max.amplitude, -gs_max.amplitude, 0])
        gs_max = pp.make_extended_trapezoid(channel=sl_channel, times=gc_times, amplitudes=gs_amp)

        # Fill-times
        t_ex = pp.calc_duration(d_ex) + pp.calc_duration(delay_preph)
        t_ref = pp.calc_duration(d_ref) + pp.calc_duration(gr)

        t_train = t_ex + n_echo * t_ref

        TR_fill = TR - t_train
        # Round to gradient raster
        TR_fill = self.system.grad_raster_time * np.round(TR_fill / self.system.grad_raster_time)
        if TR_fill < 0:
            # print("ERROR: Repetition time too short.")
            return 0
        delay_TR = pp.make_delay(TR_fill)

        # Initialize batches dictionary where batches will be saved
        batches = {}

        def initializeBatch(name="pp_1"):
            """
            Initialize a sequence with specified name and add predefined blocks for MRI pulse
            sequence corresponding to dummy pulses.

            This function initializes a new sequence with the given name (or "pp_1" by default)
            and adds blocks for dummy pulses, RF excitation, pre-phasing gradients, echo cycles,
            and TR delay. The slice and phase encoding gradients are set to zero, and multiple
            dummy pulses and echo blocks are added to the sequence.

            Parameters:
            ----------
            name : str, optional
                The name of the sequence to initialize. Defaults to "pp_1".

            Actions:
            --------
            1. Initializes a new sequence object.
            2. Sets the slice and phase gradients to 0.
            3. Adds dummy pulses followed by excitation, prephasing, echo, and TR blocks.

            Notes:
            ------
            The function assumes that the following variables are defined globally or
            accessible in the scope:

            - `batches`: a dictionary to store the batches.
            - `pp`: pypulseq class responsible for handling sequences and gradients.
            - `gs_max`, `gp_max`: maximum values for slice and phase gradients.
            - `rf_ex`, `d_ex`: RF excitation pulse and corresponding duration.
            - `gr_preph`: pre-phasing gradient block.
            - `n_echo`: number of echo cycles.
            - `rf_ref`, `d_ref`: RF refocusing pulse and corresponding duration.
            - `gs`, `gp`, `gr`: slice, phase, and readout gradients.
            - `delay_TR`: delay block for the repetition time (TR).
            """

            # Instantiate pypulseq sequence object and save it into the batches dictionary
            batches[name] = pp.Sequence(self.system)

            # Set slice and phase gradients to 0
            gs = pp.scale_grad(gs_max, 0.0)
            gp = pp.scale_grad(gp_max, 0.0)

            # Create dummy pulses
            for dummy in range(self.dummyPulses):
                # Add excitation and pre-phasing
                batches[name].add_block(rf_ex, d_ex)
                batches[name].add_block(pp.scale_grad(gr_preph, self.preemphasis), delay_preph)

                # Add echo train
                for k_echo in range(n_echo):
                    batches[name].add_block(rf_ref, d_ref)
                    batches[name].add_block(gs, gp, gr)

                # Add repetition delay
                batches[name].add_block(delay_TR)

        def createBatches():
            """
            Create MRI pulse sequence based on slice and phase sweeps, manage readout points,
            and ensure timing correctness for each sequence.

            This method generates multiple batches by sweeping across slice and phase encoding
            gradients, adding excitation, refocusing pulses, and gradient blocks. It dynamically
            divides the readout points between batches and ensures that no one exceeds
            the maximum allowable readout points. The batches are checked for timing errors,
            and the finalized batches are written to files.

            Workflow:
            ---------
            1. Loop over slice positions (`Cz`) and phase positions (`Cy`) to generate batches.
            2. Initialize a new sequence when needed based on the readout points limit.
            3. Add excitation, refocusing pulses, gradients (slice, phase, readout), and ADC blocks.
            4. Ensure correct timing of the final sequence and generate a timing error report if needed.
            5. Write each sequence to a file and interpret the sequence to generate waveforms.

            Returns:
            --------
            waveforms : dict
                Dictionary containing waveforms for each sequence.

            n_rd_points_dict : dict
                Dictionary tracking the number of readout points for each sequence.

            Notes:
            ------
            The function assumes that the following variables are defined globally or accessible
            in the scope:

            - `nSL`: number of slices.
            - `nPH`: number of phase encoding steps.
            - `n_rd_points_per_train`: number of readout points per echo train.
            - `hw.maxRdPoints`: hardware maximum allowable readout points.
            - `rf_ex`, `d_ex`: RF excitation pulse and corresponding duration.
            - `rf_ref`, `d_ref`: RF refocusing pulse and corresponding duration.
            - `gr_preph`: pre-phasing gradient block.
            - `gs_max`, `gp_max`: maximum slice and phase gradient amplitudes.
            - `n_echo`: number of echoes per sequence.
            - `nRD`, `nRD_post`, `nRD_pre`: readout duration and pre/post readout periods.
            - `adc`: analog-to-digital converter settings for readout.
            - `delay_TR`: delay block for the repetition time (TR).
            - `pp`: a module or class for gradient scaling and sequence management.

            Timing Check:
            -------------
            The function performs a timing check after generating the batches. If the timing
            is incorrect, an error report is printed with details.

            File Output:
            ------------
            The sequence files are saved with the `.seq` extension, and the waveforms are
            interpreted using the `flo_interpreter`.
            """

            n_rd_points = 0
            n_rd_points_dict = {}
            seq_idx = 0
            seq_num = "batch_0"
            waveforms = {}

            # Slice sweep
            for Cz in range(nSL):

                # Get slice gradient amplitude
                Nph_range = range(nPH)

                # Phase sweep
                for Cy in Nph_range:
                    # Initialize new sequence with corresponding dummy pulses
                    if seq_idx == 0 or n_rd_points + n_rd_points_per_train > hw.maxRdPoints:
                        # Write seq file
                        if seq_idx > 0:
                            batches[seq_num].write(seq_num + ".seq")
                            waveforms[seq_num], param_dict = self.flo_interpreter.interpret(seq_num + ".seq")
                            print(seq_num + ".seq ready!")

                        # Create new batch
                        seq_idx += 1
                        n_rd_points_dict[seq_num] = n_rd_points
                        seq_num = "batch_%i" % seq_idx
                        initializeBatch(seq_num)
                        n_rd_points = 0
                        print("Creating " + seq_num + ".seq...")

                    # Fix the phase and slice amplitude
                    sl_scale = (Cz - nSL / 2) / nSL * 2
                    pe_scale = (Cy - nPH / 2) / nPH * 2
                    gs = pp.scale_grad(gs_max, sl_scale)
                    gp = pp.scale_grad(gp_max, pe_scale)

                    # Add excitation pulse and readout de-phasing gradient
                    batches[seq_num].add_block(rf_ex, d_ex)
                    batches[seq_num].add_block(pp.scale_grad(gr_preph, self.preemphasis), delay_preph)

                    # Add the echo train
                    for k_echo in range(n_echo):
                        # Add refocusing pulse
                        batches[seq_num].add_block(rf_ref, d_ref)
                        # Add slice, phase and readout gradients
                        batches[seq_num].add_block(gs, gp, gr, adc)
                        n_rd_points += nRD + nRD_post + nRD_pre

                    # Add time delay to next repetition
                    batches[seq_num].add_block(delay_TR)

            # Get the rd point list
            n_rd_points_dict.pop('batch_0')
            n_rd_points_dict[seq_num] = n_rd_points

            # Write the sequence files
            batches[seq_num].write(seq_num + ".seq")
            waveforms[seq_num], param_dict = self.flo_interpreter.interpret(seq_num + ".seq")
            print(seq_num + ".seq ready!")
            print("%i batches created." % len(batches))
            print("Sequence ready!")

            return waveforms, n_rd_points_dict

        # Create the batches
        waveforms, n_readouts = createBatches()
        self.mapVals['n_readouts'] = list(n_readouts.values())
        self.mapVals['n_batches'] = len(n_readouts.values())
        scan_time = (nPH * nSL + self.mapVals['n_batches'] * self.dummyPulses) * self.repetitionTime * self.nScans
        self.mapVals['Scan_time_s'] = scan_time

        # Execute the batches
        data_over = []  # To save oversampled data
        for seq_num in waveforms.keys():
            # Initialize the experiment
            bw = nRD / sampling_time * hw.oversamplingFactor  # Hz
            sampling_period = 1 / bw  # s
            self.mapVals['Sampling_Period_s'] = sampling_period
            if not self.demo:
                self.expt = ex.Experiment(lo_freq=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                                          rx_t=sampling_period * 1e6,  # us
                                          init_gpa=init_gpa,
                                          gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                          auto_leds=True
                                          )
                sampling_period = self.expt.get_rx_ts()[0]  # us
                bw = 1 / sampling_period / hw.oversamplingFactor  # MHz
                print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))
            else:
                sampling_period = sampling_period * 1e6  # us
                bw = 1 / sampling_period / hw.oversamplingFactor  # MHz
                sampling_time = nRD / bw * 1e-6  # s
            self.mapVals['Bandwidth_Hz'] = bw * 1e6  # Hz
            self.mapVals['Sampling_Time_s'] = sampling_time
            self.mapVals['larmorFreq'] = hw.larmorFreq

            # Save the waveforms into the mriBlankSeq dictionaries
            self.pypulseq2mriblankseq(waveforms=waveforms[seq_num], shimming=self.shimming)

            # Load the waveforms into the red pitaya
            if not self.demo:
                if self.floDict2Exp():
                    print("Sequence waveforms loaded successfully")
                    pass
                else:
                    print("ERROR: sequence waveforms out of hardware bounds")
                    return False

            # Run the experiment or plot the sequence
            if not plotSeq:
                for scan in range(self.nScans):
                    print("Scan %i, batch %s/%i running..." % ((scan + 1), seq_num[-1], len(n_readouts.values())))
                    acq_points = 0
                    while acq_points != n_readouts[seq_num] * hw.oversamplingFactor:
                        if not self.demo:
                            rxd, msgs = self.expt.run()
                        else:
                            rxd = {'rx0': np.random.randn(n_readouts[seq_num] * hw.oversamplingFactor) +
                                          1j * np.random.randn(n_readouts[seq_num] * hw.oversamplingFactor)}
                        acq_points = np.size([rxd['rx0']])
                    data_over = np.concatenate((data_over, rxd['rx0']), axis=0)
                    print("Acquired points = %i" % acq_points)
                    print("Expected points = %i" % (n_readouts[seq_num] * hw.oversamplingFactor))
                    print("Scan %i ready!" % (scan + 1))

            elif plotSeq and standalone:
                self.sequencePlot(standalone=standalone)
                return True

            # Close the experiment
            if not self.demo:
                self.expt.__del__()

        # Process data to be plotted
        if not plotSeq:
            self.mapVals['data_over'] = data_over
            data_full = sig.decimate(data_over, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['data_full'] = data_full

        return True

    def sequenceAnalysis(self, mode=None):
        self.mode = mode

        # Get data
        data_full = self.mapVals['data_full']
        nRD, nPH, nSL = self.nPoints
        nRD = nRD + 2 * hw.addRdPoints
        n_batches = self.mapVals['n_batches']

        # Reorganize data_full
        data_prov = np.zeros([self.nScans, nRD * nPH * nSL * self.etl], dtype=complex)
        if n_batches > 1:
            n_rds = self.mapVals['n_readouts']
            data_full_a = data_full[0:sum(n_rds[0:-1]) * self.nScans]
            data_full_b = data_full[sum(n_rds[0:-1]) * self.nScans:]
            data_full_a = np.reshape(data_full_a, newshape=(n_batches - 1, self.nScans, -1, nRD))
            data_full_b = np.reshape(data_full_b, newshape=(1, self.nScans, -1, nRD))
            for scan in range(self.nScans):
                data_scan_a = np.reshape(data_full_a[:, scan, :, :], -1)
                data_scan_b = np.reshape(data_full_b[:, scan, :, :], -1)
                data_prov[scan, :] = np.concatenate((data_scan_a, data_scan_b), axis=0)
        else:
            data_full = np.reshape(data_full, (1, self.nScans, -1, nRD))
            for scan in range(self.nScans):
                data_prov[scan, :] = np.reshape(data_full[:, scan, :, :], -1)
        data_full = np.reshape(data_prov, -1)

        # Average data
        data_full = np.reshape(data_full, newshape=(self.nScans, -1))
        data = np.average(data_full, axis=0)
        self.mapVals['data'] = data

        # Generate different k-space data
        data_ind = np.zeros(shape=(self.etl, nSL, nPH, nRD), dtype=complex)
        data = np.reshape(data, newshape=(nSL, nPH, self.etl, nRD))
        for echo in range(self.etl):
            data_ind[echo, :, :, :] = np.squeeze(data[:, :, echo, :])

        # Remove added data in readout direction
        data_ind = data_ind[:, :, :, hw.addRdPoints: nRD - hw.addRdPoints]
        self.mapVals['kSpace'] = data_ind

        # Get images
        image_ind = np.zeros_like(data_ind)
        for echo in range(self.etl):
            image_ind[echo] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data_ind[echo])))
        self.mapVals['iSpace'] = image_ind

        # Prepare data to plot (plot central slice)
        axes_dict = {'x': 0, 'y': 1, 'z': 2}
        axes_keys = list(axes_dict.keys())
        axes_vals = list(axes_dict.values())
        axes_str = ['', '', '']
        n = 0
        for val in self.axesOrientation:
            index = axes_vals.index(val)
            axes_str[n] = axes_keys[index]
            n += 1

        # Normalize image
        k_space = np.zeros((self.etl * nSL, nPH, nRD - 2 * hw.addRdPoints))
        image = np.zeros((self.etl * nSL, nPH, nRD - 2 * hw.addRdPoints))
        n = 0
        for slice in range(nSL):
            for echo in range(self.etl):
                k_space[n, :, :] = np.abs(data_ind[echo, slice, :, :])
                image[n, :, :] = np.abs(image_ind[echo, slice, :, :])
                n += 1
        image = image / np.max(image) * 100

        imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        if not self.unlock_orientation:  # Image orientation
            pass
            if self.axesOrientation[2] == 2:  # Sagittal
                title = "Sagittal"
                if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 1:  # OK
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    k_space = np.flip(k_space, axis=2)
                    k_space = np.flip(k_space, axis=1)
                    x_label = "(-Y) A | PHASE | P (+Y)"
                    y_label = "(-X) I | READOUT | S (+X)"
                    imageOrientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
                else:
                    image = np.transpose(image, (0, 2, 1))
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    k_space = np.transpose(k_space, (0, 2, 1))
                    k_space = np.flip(k_space, axis=2)
                    k_space = np.flip(k_space, axis=1)
                    x_label = "(-Y) A | READOUT | P (+Y)"
                    y_label = "(-X) I | PHASE | S (+X)"
                    imageOrientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
            elif self.axesOrientation[2] == 1:  # Coronal
                title = "Coronal"
                if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 2:  # OK
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    image = np.flip(image, axis=0)
                    k_space = np.flip(k_space, axis=2)
                    k_space = np.flip(k_space, axis=1)
                    k_space = np.flip(k_space, axis=0)
                    x_label = "(+Z) R | PHASE | L (-Z)"
                    y_label = "(-X) I | READOUT | S (+X)"
                    imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                else:
                    image = np.transpose(image, (0, 2, 1))
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    image = np.flip(image, axis=0)
                    k_space = np.transpose(k_space, (0, 2, 1))
                    k_space = np.flip(k_space, axis=2)
                    k_space = np.flip(k_space, axis=1)
                    k_space = np.flip(k_space, axis=0)
                    x_label = "(+Z) R | READOUT | L (-Z)"
                    y_label = "(-X) I | PHASE | S (+X)"
                    imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            elif self.axesOrientation[2] == 0:  # Transversal
                title = "Transversal"
                if self.axesOrientation[0] == 1 and self.axesOrientation[1] == 2:
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    k_space = np.flip(k_space, axis=2)
                    k_space = np.flip(k_space, axis=1)
                    x_label = "(+Z) R | PHASE | L (-Z)"
                    y_label = "(+Y) P | READOUT | A (-Y)"
                    imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                else:  # OK
                    image = np.transpose(image, (0, 2, 1))
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    k_space = np.transpose(k_space, (0, 2, 1))
                    k_space = np.flip(k_space, axis=2)
                    k_space = np.flip(k_space, axis=1)
                    x_label = "(+Z) R | READOUT | L (-Z)"
                    y_label = "(+Y) P | PHASE | A (-Y)"
                    imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        else:
            x_label = "%s axis" % axes_str[1]
            y_label = "%s axis" % axes_str[0]
            title = "Image"

        result1 = {'widget': 'image',
                   'data': image,
                   'xLabel': x_label,
                   'yLabel': y_label,
                   'title': title,
                   'row': 0,
                   'col': 0}

        result2 = {'widget': 'image',
                   'data': np.log10(k_space),
                   'xLabel': x_label,
                   'yLabel': y_label,
                   'title': "k_space",
                   'row': 0,
                   'col': 1}

        # Dicom tags
        image_DICOM = np.transpose(image, (0, 2, 1))
        slices, rows, columns = image_DICOM.shape
        self.meta_data["Columns"] = columns
        self.meta_data["Rows"] = rows
        self.meta_data["NumberOfSlices"] = slices
        self.meta_data["NumberOfFrames"] = slices
        img_full_abs = np.abs(image_DICOM) * (2 ** 15 - 1) / np.amax(np.abs(image_DICOM))
        img_full_int = np.int16(np.abs(img_full_abs))
        img_full_int = np.reshape(img_full_int, newshape=(slices, rows, columns))
        arr = img_full_int
        self.meta_data["PixelData"] = arr.tobytes()
        self.meta_data["WindowWidth"] = 26373
        self.meta_data["WindowCenter"] = 13194
        self.meta_data["ImageOrientationPatient"] = imageOrientation_dicom
        resolution = self.mapVals['resolution'] * 1e3
        self.meta_data["PixelSpacing"] = [resolution[0], resolution[1]]
        self.meta_data["SliceThickness"] = resolution[2]
        # Sequence parameters
        self.meta_data["RepetitionTime"] = self.mapVals['repetitionTime']
        self.meta_data["EchoTime"] = self.mapVals['echoSpacing']
        self.meta_data["EchoTrainLength"] = self.mapVals['etl']

        # create self.out to run in iterative mode
        self.output = [result1, result2]

        # save data once self.output is created
        self.saveRawData()

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        return self.output


if __name__ == "__main__":
    # main(plot=True, write_seq=True)

    seq = MSE()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=True, demo=True, standalone=True)
    # seq.sequenceAnalysis(mode='Standalone')
