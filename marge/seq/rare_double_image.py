"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: rare sequence class
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

import ismrmrd
import ismrmrd.xsd
from marga_pulseq.interpreter import PSInterpreter
import pypulseq as pp
from marge.marge_utils import utils

from marge.marge_tyger import tyger_rare
from marge.marge_tyger import tyger_denoising_double
import marge.marge_tyger.tyger_config as tyger_conf

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class RareDoubleImage(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RareDoubleImage, self).__init__()
        # Input the parameters
        self.decimation_factor = None
        self.add_rd_points = None
        self.oversampling_factor = None
        self.tyger_denoising_echoes = None
        self.tyger_denoising = None
        self.boFit_file = None
        self.recon_type = None
        self.tyger_recon = None
        self.axes_orientation = None
        self.rd_direction = None
        self.image_orientation_dicom = None
        self.full_plot = None
        self.sequence_list = None
        self.unlock_orientation = None
        self.rdDephTime = None
        self.dummyPulses = None
        self.nScans = None
        self.standalone = None
        self.repetitionTime = None
        self.inversionTime = None
        self.preExTime = None
        self.sweepMode = None
        self.expt = None
        self.echoSpacing = None
        self.phGradTime = None
        self.rdGradTime = None
        self.parFourierFraction = None
        self.etl = None
        self.rfReFA = None
        self.rfExFA = None
        self.rfReTime = None
        self.rfExTime = None
        self.acqTime = None
        self.freqOffset = None
        self.nPoints = None
        self.dfov = None
        self.fov = None
        self.system = None
        self.axesOrientation = None
        self.rdPreemphasis = None
        self.addParameter(key='seqName', string='RAREInfo', val='RareDoubleImage')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, units=units.kHz,
                          field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=50.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=100.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, units=units.ms, field='SEQ')
        self.addParameter(key='preExTime', string='Preexitation time (ms)', val=0.0, units=units.ms, field='SEQ')
        self.addParameter(key='inversionTime', string='Inversion time (ms)', val=0.0, units=units.ms, field='SEQ',
                          tip="0 to ommit this pulse")
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=50., units=units.ms, field='SEQ',
                          tip="0 to ommit this pulse")
        self.addParameter(key='fov', string='FOV[x,y,z] (cm)', val=[12.0, 12.0, 12.0], units=units.cm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM',
                          tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[40, 40, 1], field='IM')
        self.addParameter(key='etl', string='Echo train length', val=4, field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, units=units.ms, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[2, 1, 0], field='IM',
                          tip="0=x, 1=y, 2=z")
        self.addParameter(key='sweepMode', string='Sweep mode', val=1, field='SEQ',
                          tip="0: sweep from -kmax to kmax. 1: sweep from 0 to kmax. 2: sweep from kmax to 0")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=5.0, units=units.ms, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='phGradTime', string='Ph gradient time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=1, field='SEQ',
                          tip="Use last dummy pulse to calibrate k = 0")
        self.addParameter(key='nNoise', string='Noise acquisitions', val=1, field='SEQ', tip="Noise noise acquisitions")
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='parFourierFraction', string='Partial fourier fraction', val=0.7, field='OTH',
                          tip="Fraction of k planes aquired in slice direction")
        self.addParameter(key='unlock_orientation', string='Unlock image orientation', val=0, field='OTH',
                          tip='0: Images oriented according to standard. 1: Image raw orientation')
        self.addParameter(key='full_plot', string='Full plot', val=True, field='OTH',
                          tip="'True' or 'False' to plot odd and even images separately")
        self.addParameter(key='k_fill', string='Filling method', val='ZP', field='PRO',
                          tip="'ZP': Zero Padding, 'POCS': Projection Onto Convex Sets")
        self.addParameter(key='tyger_denoising_echoes', string='Input echoes to Tyger', val='all', field='PRO',
                          tip='Echoes to send (odd, even or all)')
        self.addParameter(key='tyger_denoising', string='Denoising (SNRAware)', val=0, field='PRO',
                          tip='To denoising with Tyger (0 = Disabled; 1 = Enabled)')
        self.addParameter(key='tyger_recon', string='Distortion correction', val=0, field='PRO',
                          tip='B0 map acquired with SPDS required (0 = Disabled; 1 = Enabled)')
        self.addParameter(key='recon_type', string='Reconstruction type', val='cp', field='PRO',
                          tip='Options: cp or artpk.')
        self.addParameter(key='boFit_file', string='Bo Fit file', val='boFit_default.txt', field='PRO',
                          tip='Path to the Bo Fit file inside [b0_maps] folder.')
        self.addParameter(key='rd_direction', string='Rd direction', val=1, field='SEQ',
                          tip='Set the readout direction to positive (1) or negative (-1)')
        self.addParameter(key='oversampling_factor', string='Oversampling factor', val=6, field='OTH',
                          tip='Oversampling factor applied during readout')
        self.addParameter(key='decimation_factor', string='Decimation factor', val=3, field='OTH',
                          tip='Decimation applied to acquired data')
        self.addParameter(key='add_rd_points', string='Add RD points', val=10, field='OTH',
                          tip='Add RD points to avoid CIC and FIR filters issues')

        self.acq = ismrmrd.Acquisition()
        self.img = ismrmrd.Image()
        self.header = ismrmrd.xsd.ismrmrdHeader()

    def sequenceInfo(self):
        print("3D RARE sequence powered by PyPulseq")
        print("It gets two images, one from even echoes and a second one from odd echoes")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain \n")

    def sequenceTime(self):
        n_scans = self.mapVals['nScans']
        n_points = np.array(self.mapVals['nPoints'])
        etl = self.mapVals['etl']
        repetition_time = self.mapVals['repetitionTime']
        par_fourier_fraction = self.mapVals['parFourierFraction']

        # check if rf amplitude is too high
        rf_ex_fa = self.mapVals['rfExFA'] / 180 * np.pi  # rads
        rf_re_fa = self.mapVals['rfReFA'] / 180 * np.pi  # rads
        rf_ex_time = self.mapVals['rfExTime']  # us
        rf_re_time = self.mapVals['rfReTime']  # us
        rf_ex_amp = rf_ex_fa / (rf_ex_time * hw.b1Efficiency)
        rf_re_amp = rf_re_fa / (rf_re_time * hw.b1Efficiency)
        if rf_ex_amp > 1 or rf_re_amp > 1:
            print("ERROR: RF amplitude is too high, try with longer RF pulse time.")
            return 0

        seq_time = n_points[1] / etl * n_points[2] * repetition_time * 1e-3 * n_scans * par_fourier_fraction / 60
        seq_time = np.round(seq_time, decimals=1) * 2
        return seq_time  # minutes, scanTime

        # TODO: check for min and max values for all fields

    def sequenceRun(self, plotSeq=False, demo=False, standalone=False):
        """
        Runs the RARE MRI pulse sequence.

        This method orchestrates the execution of the RARE sequence by performing several key steps:
        1. Define the interpreter (FloSeq/PSInterpreter) to convert the sequence description into scanner instructions.
        2. Set system properties using PyPulseq (`pp.Opts`), which define hardware capabilities such as maximum gradient strengths and slew rates.
        3. Perform any necessary calculations for the sequence, such as timing, RF amplitudes, and gradient strengths.
        4. Define the experiment to determine the true bandwidth by using `get_sampling_period()` with an experiment defined as a class property (`self.expt`).
        5. Define sequence blocks including RF and gradient pulses that form the building blocks of the MRI sequence.
        6. Implement the `initializeBatch` method to create dummy pulses for each new batch.
        7. Define and populate the `createBatches` method, which accounts for the number of acquired points to determine when a new batch is needed.
        8. Run the batches and return the resulting data. Oversampled data is stored in `self.mapVals['data_over']`, and decimated data in `self.mapVals['data_decimated']`.

        Parameters:
        - plot_seq (bool): If True, plots the pulse sequence.
        - demo (bool): If True, runs the sequence in demo mode with simulated hardware.
        - standalone (bool): If True, runs the sequence as a standalone operation.

        Returns:
        - result (bool): The result of running the sequence, including oversampled and decimated data.
        """

        self.demo = demo
        self.plotSeq = plotSeq
        self.standalone = standalone
        print('RARE run...')

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

        # Set the fov
        self.dfov = self.dfov[self.axesOrientation]
        self.fov = self.fov[self.axesOrientation]

        # Check for used axes
        axes_enable = []
        for ii in range(3):
            if self.nPoints[ii] == 1:
                axes_enable.append(0)
            else:
                axes_enable.append(1)
        self.mapVals['axes_enable'] = axes_enable

        # Miscellaneous
        self.freqOffset = self.freqOffset * 1e6  # MHz
        resolution = self.fov / self.nPoints
        rf_ex_amp = self.rfExFA / (self.rfExTime * 1e6 * hw.b1Efficiency) * np.pi / 180
        rf_re_amp = self.rfReFA / (self.rfReTime * 1e6 * hw.b1Efficiency) * np.pi / 180
        self.mapVals['rf_ex_amp'] = rf_ex_amp
        self.mapVals['rf_re_amp'] = rf_re_amp
        self.mapVals['resolution'] = resolution
        self.mapVals['grad_rise_time'] = hw.grad_rise_time
        self.mapVals['larmorFreq'] = hw.larmorFreq + self.freqOffset
        if rf_ex_amp > 1 or rf_re_amp > 1:
            print("ERROR: RF amplitude is too high, try with longer RF pulse time.")
            return 0

        # Matrix size
        n_rd = self.nPoints[0] + 2 * self.add_rd_points
        n_ph = self.nPoints[1]
        n_sl = self.nPoints[2]

        # ETL if etl>n_ph
        if self.etl > n_ph:
            self.etl = n_ph

        # Miscellaneous
        n_rd_points_per_train = self.etl * n_rd

        # par_acq_lines in case par_acq_lines = 0
        par_acq_lines = int(int(self.nPoints[2] * self.parFourierFraction) - self.nPoints[2] / 2)
        self.mapVals['partialAcquisition'] = par_acq_lines

        # BW
        bw = self.nPoints[0] / self.acqTime * 1e-6  # MHz
        sampling_period = 1 / bw  # us

        # Readout gradient time
        if self.rdGradTime < self.acqTime:
            self.rdGradTime = self.acqTime
            print("Readout gradient time set to %0.1f ms" % (self.rdGradTime * 1e3))
        self.mapVals['rdGradTime'] = self.rdGradTime * 1e3  # ms

        # Phase and slice de- and re-phasing time
        if self.phGradTime == 0 or self.phGradTime > self.echoSpacing / 2 - self.rfExTime / 2 - self.rfReTime / 2 - 2 * hw.grad_rise_time:
            self.phGradTime = self.echoSpacing / 2 - self.rfExTime / 2 - self.rfReTime / 2 - 2 * hw.grad_rise_time
            print("Phase and slice gradient time set to %0.1f ms" % (self.phGradTime * 1e3))
        self.mapVals['phGradTime'] = self.phGradTime * 1e3  # ms

        # Max gradient amplitude
        rd_grad_amplitude = self.nPoints[0] / (hw.gammaB * self.fov[0] * self.acqTime) * axes_enable[0] * self.rd_direction
        ph_grad_amplitude = n_ph / (2 * hw.gammaB * self.fov[1] * (self.phGradTime + hw.grad_rise_time)) * axes_enable[
            1]
        sl_grad_amplitude = n_sl / (2 * hw.gammaB * self.fov[2] * (self.phGradTime + hw.grad_rise_time)) * axes_enable[
            2]
        self.mapVals['rd_grad_amplitude'] = rd_grad_amplitude
        self.mapVals['ph_grad_amplitude'] = ph_grad_amplitude
        self.mapVals['sl_grad_amplitude'] = sl_grad_amplitude

        # Readout dephasing amplitude
        rd_deph_amplitude = 0.5 * rd_grad_amplitude * (hw.grad_rise_time + self.rdGradTime) / (
                hw.grad_rise_time + self.rdDephTime)
        self.mapVals['rd_deph_amplitude'] = rd_deph_amplitude
        print("Max rd gradient amplitude: %0.1f mT/m" % (max(rd_grad_amplitude, rd_deph_amplitude) * 1e3))
        print("Max ph gradient amplitude: %0.1f mT/m" % (ph_grad_amplitude * 1e3))
        print("Max sl gradient amplitude: %0.1f mT/m" % (sl_grad_amplitude * 1e3))

        # Phase and slice gradient vector
        ph_gradients = np.linspace(-ph_grad_amplitude, ph_grad_amplitude, num=n_ph, endpoint=False)
        sl_gradients = np.linspace(-sl_grad_amplitude, sl_grad_amplitude, num=n_sl, endpoint=False)

        # Now fix the number of slices to partially acquired k-space
        n_sl = (int(self.nPoints[2] / 2) + par_acq_lines) * axes_enable[2] + (1 - axes_enable[2])
        print("Number of acquired slices: %i" % n_sl)

        # Set phase vector to given sweep mode
        ind = self.getIndex(self.etl // 2, n_ph, self.sweepMode)
        self.mapVals['sweepOrder'] = ind
        ph_gradients = ph_gradients[ind]
        self.mapVals['ph_gradients'] = ph_gradients.copy()
        self.mapVals['sl_gradients'] = sl_gradients.copy()

        # Normalize gradient list
        if ph_grad_amplitude != 0:
            ph_gradients /= ph_grad_amplitude
        if sl_grad_amplitude != 0:
            sl_gradients /= sl_grad_amplitude

        # Map the axis to "x", "y", and "z" according ot axesOrientation
        axes_map = {0: "x", 1: "y", 2: "z"}
        rd_channel = axes_map.get(self.axesOrientation[0], "")
        ph_channel = axes_map.get(self.axesOrientation[1], "")
        sl_channel = axes_map.get(self.axesOrientation[2], "")

        '''
        # Step 4: Define the experiment to get the true bandwidth
        # In this step, student need to get the real bandwidth used in the experiment. To get this bandwidth, an
        # experiment must be defined and the sampling period should be obtained using get_sampling_period()
        # Note: experiment must be passed as a class property named self.expt
        '''

        if not self.demo:
            self.expt = ex.Experiment(lo_freq=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                                      rx_t=sampling_period,  # us
                                      init_gpa=False,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                      auto_leds=True,
                                      oversampling_factor=self.oversampling_factor)
            sampling_period = self.expt.get_sampling_period()  # us
            bw = 1 / sampling_period  # MHz
            sampling_time = sampling_period * n_rd * 1e-6  # s
            print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))
            self.expt.__del__()
        else:
            sampling_time = sampling_period * n_rd * 1e-6  # s
        self.mapVals['bw_MHz'] = bw
        self.mapVals['sampling_period_us'] = sampling_period
        self.mapVals['sampling_time_s'] = sampling_time

        '''
        # Step 5: Define sequence blocks.
        # In this step, you will define the building blocks of the MRI sequence, including the RF pulses and gradient pulses.
        '''

        # First delay, sequence will start after 1 repetition time, this ensure gradient and ADC latency is not an issue.
        if self.inversionTime == 0 and self.preExTime == 0:
            delay = self.repetitionTime - self.rfExTime / 2 - system.rf_dead_time
        elif self.inversionTime > 0 and self.preExTime == 0:
            delay = self.repetitionTime - self.inversionTime - self.rfReTime / 2 - system.rf_dead_time
        elif self.inversionTime == 0 and self.preExTime > 0:
            delay = self.repetitionTime - self.preExTime - self.rfExTime / 2 - system.rf_dead_time
        else:
            delay = self.repetitionTime - self.preExTime - self.inversionTime - self.rfExTime / 2 - system.rf_dead_time
        delay_first = pp.make_delay(delay)

        # ADC to get noise
        delay = 100e-6
        block_adc_noise = pp.make_adc(
            num_samples=n_rd,
            dwell=sampling_period * 1e-6,
            delay=delay,
        )

        # Pre-excitation pulse
        if self.preExTime > 0:
            flip_pre = self.rfExFA * np.pi / 180
            delay = 0
            block_rf_pre_excitation = pp.make_block_pulse(
                flip_angle=flip_pre,
                system=system,
                duration=self.rfExTime,
                phase_offset=0.0,
                delay=0,
            )
            if self.inversionTime == 0:
                delay = self.preExTime
            else:
                delay = self.rfExTime / 2 - self.rfReTime / 2 + self.preExTime
            delay_pre_excitation = pp.make_delay(delay)

        # Inversion pulse
        if self.inversionTime > 0:
            flip_inv = self.rfReFA * np.pi / 180
            block_rf_inversion = pp.make_block_pulse(
                flip_angle=flip_inv,
                system=system,
                duration=self.rfReTime,
                phase_offset=0.0,
                delay=0,
            )
            delay = self.rfReTime / 2 - self.rfExTime / 2 + self.inversionTime
            delay_inversion = pp.make_delay(delay)

        # Excitation pulse
        flip_ex = self.rfExFA * np.pi / 180
        block_rf_excitation = pp.make_block_pulse(
            flip_angle=flip_ex,
            system=system,
            duration=self.rfExTime,
            phase_offset=0.0,
            delay=0.0,
            use='excitation'
        )

        # De-phasing gradient
        delay = system.rf_dead_time + self.rfExTime
        block_gr_rd_preph = pp.make_trapezoid(
            channel=rd_channel,
            system=system,
            amplitude=rd_deph_amplitude * hw.gammaB * self.rdPreemphasis,
            flat_time=self.rdDephTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Delay to re-focusing pulse
        delay_preph = pp.make_delay(self.echoSpacing / 2 + self.rfExTime / 2 - self.rfReTime / 2)

        # Refocusing pulse odd
        flip_re = self.rfReFA * np.pi / 180
        block_rf_refocusing_odd = pp.make_block_pulse(
            flip_angle=flip_re,
            system=system,
            duration=self.rfReTime,
            phase_offset=np.pi / 2,
            delay=0,
            use='refocusing'
        )

        # Refocusing pulse
        flip_re = self.rfReFA * np.pi / 180
        block_rf_refocusing_eve = pp.make_block_pulse(
            flip_angle=flip_re,
            system=system,
            duration=self.rfReTime,
            phase_offset=np.pi / 2,
            delay=0,
            use='refocusing'
        )

        # Delay to next refocusing pulse
        delay_reph = pp.make_delay(self.echoSpacing)

        # Phase gradient de-phasing
        delay = system.rf_dead_time + self.rfReTime
        block_gr_ph_deph = pp.make_trapezoid(
            channel=ph_channel,
            system=system,
            amplitude=ph_grad_amplitude * hw.gammaB + float(ph_grad_amplitude == 0),
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Slice gradient de-phasing
        delay = system.rf_dead_time + self.rfReTime
        block_gr_sl_deph = pp.make_trapezoid(
            channel=sl_channel,
            system=system,
            amplitude=sl_grad_amplitude * hw.gammaB + float(sl_grad_amplitude == 0),
            flat_time=self.phGradTime,
            delay=delay,
            rise_time=hw.grad_rise_time,
        )

        # Readout gradient
        delay = system.rf_dead_time + self.rfReTime / 2 + self.echoSpacing / 2 - self.rdGradTime / 2 - \
                hw.grad_rise_time
        block_gr_rd_reph = pp.make_trapezoid(
            channel=rd_channel,
            system=system,
            amplitude=rd_grad_amplitude * hw.gammaB,
            flat_time=self.rdGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # ADC to get the signal
        delay = system.rf_dead_time + self.rfReTime / 2 + self.echoSpacing / 2 - sampling_time / 2
        block_adc_signal = pp.make_adc(
            num_samples=n_rd,
            dwell=sampling_period * 1e-6,
            delay=delay,
        )

        # Phase gradient re-phasing
        delay = system.rf_dead_time + self.rfReTime / 2 - self.echoSpacing / 2 + sampling_time / 2
        block_gr_ph_reph = pp.make_trapezoid(
            channel=ph_channel,
            system=system,
            amplitude=ph_grad_amplitude * hw.gammaB + float(ph_grad_amplitude == 0),
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Slice gradient re-phasing
        delay = system.rf_dead_time + self.rfReTime / 2 - self.echoSpacing / 2 + sampling_time / 2
        block_gr_sl_reph = pp.make_trapezoid(
            channel=sl_channel,
            system=system,
            amplitude=sl_grad_amplitude * hw.gammaB + float(sl_grad_amplitude == 0),
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Delay TR
        delay = self.repetitionTime + self.rfReTime / 2 - self.rfExTime / 2 - (self.etl + 0.5) * self.echoSpacing - \
                self.inversionTime - self.preExTime
        if self.inversionTime > 0 and self.preExTime == 0:
            delay -= self.rfExTime / 2
        delay_tr = pp.make_delay(delay)

        '''
        # Step 6: Define your initializeBatch according to your sequence.
        # In this step, you will create the initializeBatch method to create dummy pulses that will be initialized for
        # each new batch.
        '''

        def initialize_batch():
            # Instantiate pypulseq sequence object
            batch = pp.Sequence(system)
            n_rd_points = 0
            n_adc = 0

            # Set slice and phase gradients to 0
            gr_ph_deph = pp.scale_grad(block_gr_ph_deph, scale=0.0)
            gr_sl_deph = pp.scale_grad(block_gr_sl_deph, scale=0.0)
            gr_ph_reph = pp.scale_grad(block_gr_ph_reph, scale=0.0)
            gr_sl_reph = pp.scale_grad(block_gr_sl_reph, scale=0.0)

            # Add first delay and first noise measurement
            batch.add_block(delay_first, block_adc_noise)
            n_rd_points += n_rd
            n_adc += 1

            # Create dummy pulses
            for dummy in range(self.dummyPulses):
                # Pre-excitation pulse
                if self.preExTime > 0:
                    gr_rd_preex = pp.scale_grad(block_gr_rd_preph, scale=1.0)
                    batch.add_block(block_rf_pre_excitation,
                                    gr_rd_preex,
                                    delay_pre_excitation)

                # Inversion pulse
                if self.inversionTime > 0:
                    gr_rd_inv = pp.scale_grad(block_gr_rd_preph, scale=-1.0)
                    batch.add_block(block_rf_inversion,
                                    gr_rd_inv,
                                    delay_inversion)

                # Add excitation pulse and readout de-phasing gradient
                batch.add_block(block_gr_rd_preph,
                                block_rf_excitation,
                                delay_preph)

                # Add echo train
                for echo in range(self.etl):
                    if dummy == self.dummyPulses - 1:
                        batch.add_block(block_rf_refocusing_odd,
                                        block_gr_rd_reph,
                                        gr_ph_deph,
                                        gr_sl_deph,
                                        block_adc_signal,
                                        delay_reph)
                        batch.add_block(gr_ph_reph,
                                        gr_sl_reph)
                        n_rd_points += n_rd
                        n_adc += 1
                    else:
                        batch.add_block(block_rf_refocusing_odd,
                                        block_gr_rd_reph,
                                        gr_ph_deph,
                                        gr_sl_deph,
                                        delay_reph)
                        batch.add_block(gr_ph_reph,
                                        gr_sl_reph)

                # Add time delay to next repetition
                batch.add_block(delay_tr)

            return batch, n_rd_points, n_adc

        def initialize_batch_0():
            # Instantiate pypulseq sequence object
            batch = pp.Sequence(system)
            n_rd_points = 0
            n_adc = 0

            # Set slice and phase gradients to 0
            gr_ph_deph = pp.scale_grad(block_gr_ph_deph, scale=0.0)
            gr_sl_deph = pp.scale_grad(block_gr_sl_deph, scale=0.0)
            gr_ph_reph = pp.scale_grad(block_gr_ph_reph, scale=0.0)
            gr_sl_reph = pp.scale_grad(block_gr_sl_reph, scale=0.0)

            # Add first delay and first noise measurement
            # self.nNoise = 1 # Delete if added as input to the sequence
            for nNoise in range(self.nNoise):
                batch.add_block(delay_first, block_adc_noise)
                n_rd_points += n_rd
                n_adc += 1

            # Create dummy pulses
            for dummy in range(self.dummyPulses):
                # Pre-excitation pulse
                if self.preExTime > 0:
                    gr_rd_preex = pp.scale_grad(block_gr_rd_preph, scale=1.0)
                    batch.add_block(block_rf_pre_excitation,
                                    gr_rd_preex,
                                    delay_pre_excitation)

                # Inversion pulse
                if self.inversionTime > 0:
                    gr_rd_inv = pp.scale_grad(block_gr_rd_preph, scale=-1.0)
                    batch.add_block(block_rf_inversion,
                                    gr_rd_inv,
                                    delay_inversion)

                # Add excitation pulse and readout de-phasing gradient
                batch.add_block(block_gr_rd_preph,
                                block_rf_excitation,
                                delay_preph)

                # Add echo train
                for echo in range(self.etl):
                    if dummy == self.dummyPulses - 1:
                        batch.add_block(block_rf_refocusing_odd,
                                        block_gr_rd_reph,
                                        gr_ph_deph,
                                        gr_sl_deph,
                                        block_adc_signal,
                                        delay_reph)
                        batch.add_block(gr_ph_reph,
                                        gr_sl_reph)
                        n_rd_points += n_rd
                        n_adc += 1
                    else:
                        batch.add_block(block_rf_refocusing_odd,
                                        block_gr_rd_reph,
                                        gr_ph_deph,
                                        gr_sl_deph,
                                        delay_reph)
                        batch.add_block(gr_ph_reph,
                                        gr_sl_reph)

                # Add time delay to next repetition
                batch.add_block(delay_tr)

            return batch, n_rd_points, n_adc

        '''
        Step 7: Define your createBatches method.
        In this step you will populate the batches adding the blocks previously defined in step 4, and accounting for
        number of acquired points to check if a new batch is required.
        '''

        def create_batches():
            batches = {}  # Dictionary to save batches PyPulseq sequences
            waveforms = {}  # Dictionary to store generated waveforms per each batch
            n_rd_points_dict = {}  # Dictionary to track readout points for each batch
            n_rd_points = 0  # To account for number of acquired rd points
            seq_idx = 0  # Sequence batch index
            n_adc = 0  # To account for number of adc windows
            batch_num = "batch_0"  # Initial batch name

            # Slice sweep
            for sl_idx in range(n_sl):
                ph_idx = 0
                # Phase sweep
                while ph_idx < n_ph:
                    # Check if a new batch is needed (either first batch or exceeding readout points limit)
                    if seq_idx == 0 or n_rd_points + n_rd_points_per_train > hw.maxRdPoints:
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
                        if batch_num == "batch_1":
                            batches[batch_num], n_rd_points, n_adc_0 = initialize_batch_0()  # Initialize new batch
                        else:
                            batches[batch_num], n_rd_points, n_adc_0 = initialize_batch()  # Initialize new batch
                        n_adc += n_adc_0
                        print(f"Creating {batch_num}.seq...")

                    # Pre-excitation pulse
                    if self.preExTime > 0:
                        gr_rd_preex = pp.scale_grad(block_gr_rd_preph, scale=+1.0)
                        batches[batch_num].add_block(block_rf_pre_excitation,
                                                     gr_rd_preex,
                                                     delay_pre_excitation)

                    # Inversion pulse
                    if self.inversionTime > 0:
                        gr_rd_inv = pp.scale_grad(block_gr_rd_preph, scale=-1.0)
                        batches[batch_num].add_block(block_rf_inversion,
                                                     gr_rd_inv,
                                                     delay_inversion)

                    # Add excitation pulse and readout de-phasing gradient
                    batches[batch_num].add_block(block_gr_rd_preph,
                                                 block_rf_excitation,
                                                 delay_preph)

                    # Add echo train
                    for echo in range(self.etl):
                        # Fix the phase and slice amplitude
                        gr_ph_deph = pp.scale_grad(block_gr_ph_deph, ph_gradients[ph_idx])
                        gr_sl_deph = pp.scale_grad(block_gr_sl_deph, sl_gradients[sl_idx])
                        gr_ph_reph = pp.scale_grad(block_gr_ph_reph, - ph_gradients[ph_idx])
                        gr_sl_reph = pp.scale_grad(block_gr_sl_reph, - sl_gradients[sl_idx])

                        # Add blocks
                        if echo % 2 == 0:
                            batches[batch_num].add_block(block_rf_refocusing_odd,
                                                         block_gr_rd_reph,
                                                         gr_ph_deph,
                                                         gr_sl_deph,
                                                         block_adc_signal,
                                                         delay_reph)
                            batches[batch_num].add_block(gr_ph_reph,
                                                         gr_sl_reph)
                        else:
                            batches[batch_num].add_block(block_rf_refocusing_eve,
                                                         block_gr_rd_reph,
                                                         gr_ph_deph,
                                                         gr_sl_deph,
                                                         block_adc_signal,
                                                         delay_reph)
                            batches[batch_num].add_block(gr_ph_reph,
                                                         gr_sl_reph)
                        n_rd_points += n_rd
                        n_adc += 1

                        if echo % 2 == 1:
                            ph_idx += 1
                        # else:
                        #     ph_idx += 1

                    # Add time delay to next repetition
                    batches[batch_num].add_block(delay_tr)

            # After final repetition, save and interpret the last batch
            batches[batch_num].write(batch_num + ".seq")
            waveforms[batch_num], param_dict = flo_interpreter.interpret(batch_num + ".seq")
            print(f"{batch_num}.seq ready!")
            print(f"{len(batches)} batches created. Sequence ready!")

            # Update the number of acquired points in the last batch
            n_rd_points_dict.pop('batch_0')
            n_rd_points_dict[batch_num] = n_rd_points

            return waveforms, n_rd_points_dict, n_adc

        ''' 
        Step 8: Run the batches
        This step will handle the different batches, run it and get the resulting data. This should not be modified.
        Oversampled data will be available in self.mapVals['data_over']
        Decimated data will be available in self.mapVals['data_decimated']
        '''
        waveforms, n_readouts, n_adc = create_batches()
        return self.runBatches(waveforms=waveforms,
                               n_readouts=n_readouts,
                               n_adc=n_adc,
                               frequency=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                               bandwidth=bw,  # MHz
                               decimate='Normal',
                               oversampling_factor=self.oversampling_factor,
                               decimation_factor=self.decimation_factor,
                               hardware=True,
                               )

    def sequenceAnalysis(self, mode=None):
        super().sequenceAnalysis(mode=mode)

        # Get axes in strings
        axes = self.mapVals['axesOrientation']
        axesDict = {'x': 0, 'y': 1, 'z': 2}
        axesKeys = list(axesDict.keys())
        axesVals = list(axesDict.values())
        axesStr = ['', '', '']
        n = 0
        for val in axes:
            index = axesVals.index(val)
            axesStr[n] = axesKeys[index]
            n += 1
            
        ## Tyger Reconstruction
        input_echoes = self.tyger_denoising_echoes
        out_field = 'image3D_den'
        out_field_k = 'kSpace3D_den'
        result_Tyger = None
        if self.mapVals['axes_enable'] == [1, 1, 1] and self.tyger_denoising == 1:
            try:
                rawData_path = self.directory_mat + '/' + self.file_name + '.mat'
                imgTyger = tyger_denoising_double.denoisingTyger_double(rawData_path, out_field, out_field_k,
                                                                        input_echoes)
                imageTyger = np.abs(imgTyger[0])
                imageTyger = imageTyger / np.max(np.reshape(imageTyger, -1)) * 100

                ## Image plot
                # Tyger
                if self.mapVals['unlock_orientation'] == 0:
                    result_Tyger, _, _ = utils.fix_image_orientation(imageTyger, axes=self.axesOrientation)
                    result_Tyger['row'] = 0
                    result_Tyger['col'] = 1
                    result_Tyger['title'] = "Tyger"
                else:
                    result_Tyger = {'widget': 'image', 'data': imageTyger, 'xLabel': "%s" % axesStr[1],
                                    'yLabel': "%s" % axesStr[0], 'title': "Tyger", 'row': 0, 'col': 1}
            except Exception as e:
                print('Tyger reconstruction failed.')
                print(f'Error: {e}')
                
        if self.mapVals['axes_enable'] == [1, 1, 1] and self.tyger_recon == 1:
            if self.tyger_denoising == 1:
                if input_echoes == 'even': 
                    input_field = out_field_k + '_even'
                elif input_echoes == 'odd': 
                    input_field = out_field_k + '_odd'
                elif input_echoes == 'all': 
                    input_field = out_field_k + '_all'
            else:
                input_field =''
                
            if self.full_plot == 'False' or self.full_plot is False:
                print('Preparing Tyger enviroment...')
                rawData_path = self.directory_mat + '/' + self.file_name + '.mat'
                sign_rarepp = [-1, -1, -1, 1, 1, 1, 1, 1, tyger_conf.cp_batchsize_RARE]
                if self.recon_type == 'cp':
                    output_field = 'imgTygerCP'
                # elif self.recon_type == 'art':
                #     output_field = 'imgTygerART'
                elif self.recon_type == 'artpk':
                    output_field = 'imgTygerARTPK'
                elif self.recon_type == 'fft':
                    output_field = 'imgTygerFFT'
                else:
                    print('Reconstruction type not available in tyger. Reassigned to FFT.')
                    self.recon_type == 'fft'
                    output_field = 'imgTygerFFT'
                boFit_path = 'b0_maps/fits/' + self.boFit_file

                try:
                    imgTyger = tyger_rare.reconTygerRARE(rawData_path, self.recon_type, boFit_path, sign_rarepp, output_field, input_field)
                    imageTyger = np.abs(imgTyger[0])
                    imageTyger = imageTyger / np.max(np.reshape(imageTyger, -1)) * 100

                    ## Image plot
                    # Tyger
                    if self.mapVals['unlock_orientation'] == 0:
                        result_Tyger, _, _ = utils.fix_image_orientation(imageTyger, axes=self.axesOrientation)
                        result_Tyger['row'] = 0
                        result_Tyger['col'] = 1
                        result_Tyger['title'] = "Tyger"
                    else:
                        result_Tyger = {'widget': 'image', 'data': imageTyger, 'xLabel': "%s" % axesStr[1],
                                        'yLabel': "%s" % axesStr[0], 'title': "Tyger", 'row': 0, 'col': 1}

                except Exception as e:
                    print('Tyger reconstruction failed.')
                    print(f'Error: {e}')
            else:
                print('Tyger reconstruction not available for Full plot True.')
                print('Change this input parameter to False.')
            
            if result_Tyger is not None:
                self.output.append(result_Tyger)

        return self.output

if __name__ == '__main__':
    seq = RareDoubleImage()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=False, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
