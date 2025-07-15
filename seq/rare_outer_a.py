"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: rare sequence class. It samples only the outer k-space without the center
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
import configs.hw_config as hw # Import the scanner hardware config
import configs.units as units
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.

from marga_pulseq.interpreter import PSInterpreter
import pypulseq as pp

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class RareOuter(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RareOuter, self).__init__()
        # Input the parameters
        self.nPoints0 = None
        self.angulation = None
        self.image_orientation_dicom = None
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
        self.rotationAxis = None
        self.rotation = None
        self.angle = None
        self.echoMode = None
        self.axesOrientation = None
        self.addParameter(key='seqName', string='RAREInfo', val='Rare Outer A')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='pypulseq', string='PyPulseq', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=10, field='IM') ## number of scans
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=50.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=100.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, units=units.ms, field='SEQ')
        self.addParameter(key='echoMode', string='Echoes', val='All', field='SEQ', tip="'All', 'Odd', 'Even'")
        self.addParameter(key='preExTime', string='Preexitation time (ms)', val=0.0, units=units.ms, field='SEQ')
        self.addParameter(key='inversionTime', string='Inversion time (ms)', val=0.0, units=units.ms, field='SEQ', tip="0 to ommit this pulse")
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=300., units=units.ms, field='SEQ', tip="0 to ommit this pulse")
        self.addParameter(key='fov', string='FOV[x,y,z] (cm)', val=[12.0, 12.0, 12.0], units=units.cm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM', tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[40, 40, 1], field='IM')
        self.addParameter(key='nPoints0', string='nPoints0[rd, ph, sl]', val=[40, 8, 1], field='IM')
        self.addParameter(key='etl', string='Echo train length', val=4, field='SEQ') ## nm of peaks in 1 repetition
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, units=units.ms, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[2, 1, 0], field='IM', tip="0=x, 1=y, 2=z")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=5.0, units=units.ms, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='phGradTime', string='Ph gradient time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')
        self.addParameter(key='rfPhase', string='RF phase (º)', val=0.0, field='OTH')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=1, field='SEQ', tip="Use last dummy pulse to calibrate k = 0")
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='par_fourier_ph', string='Partial fourier in phase', val=0, field='OTH',
                          tip="Acquires only half of phase encoding lines")
        self.addParameter(key='par_fourier_sl', string='Partial fourier in slice', val=0, field='OTH',
                          tip="Acquires only half of slice encoding lines")
        self.addParameter(key='unlock_orientation', string='Unlock image orientation', val=0, field='OTH',
                          tip='0: Images oriented according to standard. 1: Image raw orientation')

    def sequenceInfo(self):
        print("3D RARE sequence powered by PyPulseq")
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
        if rf_ex_amp>1 or rf_re_amp>1:
            print("ERROR: RF amplitude is too high, try with longer RF pulse time.")
            return 0

        seq_time = n_points[1]/etl*n_points[2]*repetition_time*1e-3*n_scans*par_fourier_fraction/60
        seq_time = np.round(seq_time, decimals=1)
        return seq_time  # minutes, scanTime

        # TODO: check for min and max values for all fields

    def sequenceRun(self, plot_seq=False, demo=False, standalone=False):
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
        self.plotSeq = plot_seq
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
        rf_ex_amp = self.rfExFA/(self.rfExTime*1e6*hw.b1Efficiency)*np.pi/180
        rf_re_amp = self.rfReFA/(self.rfReTime*1e6*hw.b1Efficiency)*np.pi/180
        self.mapVals['rf_ex_amp'] = rf_ex_amp
        self.mapVals['rf_re_amp'] = rf_re_amp
        self.mapVals['grad_rise_time'] = hw.grad_rise_time
        self.mapVals['addRdPoints'] = hw.addRdPoints
        self.mapVals['larmorFreq'] = hw.larmorFreq
        if rf_ex_amp > 1 or rf_re_amp > 1:
            print("ERROR: RF amplitude is too high, try with longer RF pulse time.")
            return 0

        # Set timing to multiples of gradient raster time for consistency (use us to get more accuracy)
        self.echoSpacing = (self.echoSpacing * 1e6) // (hw.grad_raster_time * 1e6) * hw.grad_raster_time
        self.repetitionTime = (self.repetitionTime * 1e6) // (hw.grad_raster_time * 1e6) * hw.grad_raster_time
        self.rdGradTime = (self.rdGradTime * 1e6) // (hw.grad_raster_time * 1e6) * hw.grad_raster_time
        self.phGradTime = (self.phGradTime * 1e6) // (hw.grad_raster_time * 1e6) * hw.grad_raster_time
        self.rdDephTime = (self.rdDephTime * 1e6) // (hw.grad_raster_time * 1e6) * hw.grad_raster_time
        print("Echo spacing: %0.3f ms" % (self.echoSpacing * 1e3))
        print("Repetition time: %0.3f ms" % (self.repetitionTime * 1e3))

        # Matrix size
        n_rd = self.nPoints[0] + 2 * hw.addRdPoints
        n_ph = self.nPoints[1]
        n_ph0 = self.nPoints0[1]
        n_sl = self.nPoints[2]
        n_sl0 = self.nPoints0[2]


        # ETL if etl>n_ph
        if self.etl>n_ph-n_ph0:
            print("ETL is too long")
            return 0

        # Miscellaneous
        n_rd_points_per_train = self.etl * n_rd

        # BW
        bw = self.nPoints[0] / self.acqTime * 1e-6  # MHz
        sampling_period = 1 / bw  # us

        # Readout gradient time
        if self.rdGradTime<self.acqTime:
            self.rdGradTime = self.acqTime
            print("Readout gradient time set to %0.1f ms" % (self.rdGradTime * 1e3))
        self.mapVals['rdGradTime'] = self.rdGradTime * 1e3  # ms

        # Phase and slice de- and re-phasing time
        if self.phGradTime == 0 or self.phGradTime > self.echoSpacing/2-self.rfExTime/2-self.rfReTime/2-2*hw.grad_rise_time:
            self.phGradTime = self.echoSpacing/2-self.rfExTime/2-self.rfReTime/2-2*hw.grad_rise_time
            print("Phase and slice gradient time set to %0.1f ms" % (self.phGradTime * 1e3))
        self.mapVals['phGradTime'] = self.phGradTime*1e3  # ms

        # Max gradient amplitude
        rd_grad_amplitude = self.nPoints[0]/(hw.gammaB*self.fov[0]*self.acqTime)*axes_enable[0]
        ph_grad_amplitude = n_ph/(2*hw.gammaB*self.fov[1]*(self.phGradTime+hw.grad_rise_time))*axes_enable[1]
        sl_grad_amplitude = n_sl/(2*hw.gammaB*self.fov[2]*(self.phGradTime+hw.grad_rise_time))*axes_enable[2]
        self.mapVals['rd_grad_amplitude'] = rd_grad_amplitude
        self.mapVals['ph_grad_amplitude'] = ph_grad_amplitude
        self.mapVals['sl_grad_amplitude'] = sl_grad_amplitude

        # Readout dephasing amplitude
        rd_deph_amplitude = 0.5*rd_grad_amplitude*(hw.grad_rise_time+self.rdGradTime)/(hw.grad_rise_time+self.rdDephTime)
        self.mapVals['rd_deph_amplitude'] = rd_deph_amplitude
        print("Max rd gradient amplitude: %0.1f mT/m" % (max(rd_grad_amplitude, rd_deph_amplitude) * 1e3))
        print("Max ph gradient amplitude: %0.1f mT/m" % (ph_grad_amplitude * 1e3))
        print("Max sl gradient amplitude: %0.1f mT/m" % (sl_grad_amplitude * 1e3))

        # Phase and slice gradient vector
        ph_gradients = np.linspace(-ph_grad_amplitude,ph_grad_amplitude,num=n_ph,endpoint=False)
        sl_gradients = np.linspace(-sl_grad_amplitude,sl_grad_amplitude,num=n_sl,endpoint=False)

        # Set phase vector to given sweep mode
        ind_ph, ind_sl = self.get_sweep_order()
        self.mapVals['phase_sweep_order'] = ind_ph
        self.mapVals['slice_sweep_order'] = ind_sl
        ph_gradients = ph_gradients[ind_ph]
        sl_gradients = sl_gradients[ind_sl]
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
                                      auto_leds=True)
            sampling_period = self.expt.get_sampling_period() # us
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
        if self.inversionTime==0 and self.preExTime==0:
            delay = self.repetitionTime - self.rfExTime / 2 - system.rf_dead_time
        elif self.inversionTime>0 and self.preExTime==0:
            delay = self.repetitionTime - self.inversionTime - self.rfReTime / 2 - system.rf_dead_time
        elif self.inversionTime==0 and self.preExTime>0:
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
        if self.preExTime>0:
            flip_pre = self.rfExFA * np.pi / 180
            delay = 0
            block_rf_pre_excitation = pp.make_block_pulse(
                flip_angle=flip_pre,
                system=system,
                duration=self.rfExTime,
                phase_offset=0.0,
                delay=0,
            )
            if self.inversionTime==0:
                delay = self.preExTime
            else:
                delay = self.rfExTime / 2 - self.rfReTime / 2 + self.preExTime
            delay_pre_excitation = pp.make_delay(delay)

        # Inversion pulse
        if self.inversionTime>0:
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
            use = 'excitation'
        )

        # De-phasing gradient
        delay = system.rf_dead_time + self.rfExTime / 2 + ((self.rfExTime / 2* 1e6) // (hw.grad_raster_time * 1e6) + 1) * hw.grad_raster_time  # multiple of grad_raster_time
        block_gr_rd_preph = pp.make_trapezoid(
            channel=rd_channel,
            system=system,
            amplitude=rd_deph_amplitude * hw.gammaB,
            flat_time=self.rdDephTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Delay to re-focusing pulse
        delay_preph = pp.make_delay(self.echoSpacing / 2 + self.rfExTime / 2 - self.rfReTime / 2)

        # Refocusing pulse
        flip_re = self.rfReFA * np.pi / 180
        block_rf_refocusing = pp.make_block_pulse(
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
        delay = system.rf_dead_time + self.rfReTime / 2 + ((self.rfReTime / 2 * 1e6) // (hw.grad_raster_time * 1e6) + 1) * hw.grad_raster_time
        block_gr_ph_deph = pp.make_trapezoid(
            channel=ph_channel,
            system=system,
            amplitude=ph_grad_amplitude * hw.gammaB + float(ph_grad_amplitude==0),
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Slice gradient de-phasing
        delay = system.rf_dead_time + self.rfReTime / 2 + ((self.rfReTime / 2 * 1e6) // (hw.grad_raster_time * 1e6) + 1) * hw.grad_raster_time
        block_gr_sl_deph = pp.make_trapezoid(
            channel=sl_channel,
            system=system,
            amplitude=sl_grad_amplitude * hw.gammaB + float(sl_grad_amplitude==0),
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
        delay = (system.rf_dead_time + self.rfReTime / 2 - self.echoSpacing / 2 +
                 ((self.nPoints[0] / 2 / bw) // (hw.grad_raster_time * 1e6) + 1) * hw.grad_raster_time)
        block_gr_ph_reph = pp.make_trapezoid(
            channel=ph_channel,
            system=system,
            amplitude=ph_grad_amplitude * hw.gammaB + float(ph_grad_amplitude==0),
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Slice gradient re-phasing
        delay = (system.rf_dead_time + self.rfReTime / 2 - self.echoSpacing / 2 +
                 ((self.nPoints[0] / 2 / bw) // (hw.grad_raster_time * 1e6) + 1) * hw.grad_raster_time)
        block_gr_sl_reph = pp.make_trapezoid(
            channel=sl_channel,
            system=system,
            amplitude=sl_grad_amplitude * hw.gammaB + float(sl_grad_amplitude==0),
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
            """
            Initializes a batch of MRI sequence blocks using PyPulseq for a given experimental configuration.

            Returns:
            --------
            tuple
                - `batch` (pp.Sequence): A PyPulseq sequence object containing the configured sequence blocks.
                - `n_rd_points` (int): Total number of readout points in the batch.
                - `n_adc` (int): Total number of ADC acquisitions in the batch.

            Workflow:
            ---------
            1. **Create PyPulseq Sequence Object**:
                - Instantiates a new PyPulseq sequence object (`pp.Sequence`) and initializes counters for
                  readout points (`n_rd_points`) and ADC events (`n_adc`).

            2. **Set Gradients to Zero**:
                - Initializes slice and phase gradients (`gr_ph_deph`, `gr_sl_deph`, `gr_ph_reph`, `gr_sl_reph`) to zero
                  by scaling predefined gradient blocks with a factor of 0.

            3. **Add Initial Delay and Noise Measurement**:
                - Adds an initial delay block (`delay_first`) and a noise measurement ADC block (`block_adc_noise`)
                  to the sequence.

            4. **Generate Dummy Pulses**:
                - Creates a specified number of dummy pulses (`self.dummyPulses`) to prepare the system for data acquisition:
                    - **Pre-excitation Pulse**:
                        - If `self.preExTime > 0`, adds a pre-excitation pulse with a readout pre-phasing gradient.
                    - **Inversion Pulse**:
                        - If `self.inversionTime > 0`, adds an inversion pulse with a scaled readout pre-phasing gradient.
                    - **Excitation Pulse**:
                        - Adds an excitation pulse followed by a readout de-phasing gradient (`block_gr_rd_preph`).

                - For each dummy pulse:
                    - **Echo Train**:
                        - For the last dummy pulse, appends an echo train that includes:
                            - A refocusing pulse.
                            - Gradients for readout re-phasing, phase de-phasing, and slice de-phasing.
                            - ADC signal acquisition block (`block_adc_signal`).
                            - Gradients for phase and slice re-phasing.
                        - For other dummy pulses, excludes the ADC signal acquisition.

                    - **Repetition Time Delay**:
                        - Adds a delay (`delay_tr`) to separate repetitions.

            5. **Return Results**:
                - Returns the configured sequence (`batch`), total readout points (`n_rd_points`), and number of ADC events (`n_adc`).

            """
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
                if self.preExTime>0:
                    gr_rd_preex = pp.scale_grad(block_gr_rd_preph, scale=1.0)
                    batch.add_block(block_rf_pre_excitation,
                                            gr_rd_preex,
                                            delay_pre_excitation)

                # Inversion pulse
                if self.inversionTime>0:
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
                    if dummy == self.dummyPulses-1:
                        batch.add_block(block_rf_refocusing,
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
                        batch.add_block(block_rf_refocusing,
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
            """
            Creates and processes multiple batches of MRI sequence blocks for slice and phase encoding sweeps.

            Returns:
            --------
            tuple
                - `waveforms` (dict): Dictionary of interpreted waveform data for each batch.
                - `n_rd_points_dict` (dict): Dictionary mapping each batch to its total number of readout points.
                - `n_adc` (int): Total number of ADC acquisitions across all batches.

            Workflow:
            ---------
            1. **Initialization**:
                - Initializes dictionaries to store batches (`batches`), waveforms (`waveforms`), and readout points (`n_rd_points_dict`).
                - Tracks the current readout point count (`n_rd_points`), ADC window count (`n_adc`), and batch index (`seq_idx`).

            2. **Slice and Phase Sweep**:
                - Iterates over slices (`n_sl`) and phases (`n_ph`) to build and organize sequence blocks:
                    - **Batch Management**:
                        - Creates a new batch if no batch exists or the current batch exceeds the hardware limit (`hw.maxRdPoints`).
                        - Writes the previous batch to disk, interprets it using `flo_interpreter`, and updates dictionaries.
                        - Initializes the next batch with `initializeBatch()`.

                    - **Pre-Excitation and Inversion Pulses**:
                        - Optionally adds a pre-excitation pulse (`block_rf_pre_excitation`) and an inversion pulse (`block_rf_inversion`), if respective times (`self.preExTime`, `self.inversionTime`) are greater than zero.

                    - **Excitation and Echo Train**:
                        - Adds an excitation pulse followed by an echo train for phase and slice gradients:
                            - Gradients are scaled based on the current slice (`sl_idx`) and phase (`ph_idx`) indices.
                            - Includes ADC acquisition blocks (`block_adc_signal`) and refocusing pulses for each echo.

                    - **Repetition Time Delay**:
                        - Adds a delay (`delay_tr`) between repetitions.

            3. **Final Batch Processing**:
                - Writes and interprets the last batch after completing all slices and phases.
                - Updates the total readout points for the final batch.

            Returns:
            --------
            - `waveforms`: Interpreted waveforms for each batch, generated using the `flo_interpreter`.
            - `n_rd_points_dict`: Maps batch names to the total readout points per batch.
            - `n_adc`: Total number of ADC acquisition windows across all batches.
            """
            batches = {}  # Dictionary to save batches PyPulseq sequences
            waveforms = {}  # Dictionary to store generated waveforms per each batch
            n_rd_points_dict = {}  # Dictionary to track readout points for each batch
            n_rd_points = 0  # To account for number of acquired rd points
            seq_idx = 0  # Sequence batch index
            n_adc = 0  # To account for number of adc windows
            batch_num = "batch_0"  # Initial batch name

            # Slice sweep
            for sl_idx in range(np.size(sl_gradients)):
                ph_idx = 0
                # Phase sweep
                while ph_idx < np.size(ph_gradients):
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
                        if self.echoMode=='All':
                            batches[batch_num].add_block(block_rf_refocusing,
                                                    block_gr_rd_reph,
                                                    gr_ph_deph,
                                                    gr_sl_deph,
                                                    block_adc_signal,
                                                    delay_reph)
                            batches[batch_num].add_block(gr_ph_reph,
                                                    gr_sl_reph)
                            n_rd_points += n_rd
                            n_adc += 1
                            ph_idx += 1
                        elif self.echoMode=='Odd' and echo%2==0:
                            batches[batch_num].add_block(block_rf_refocusing,
                                                         block_gr_rd_reph,
                                                         gr_ph_deph,
                                                         gr_sl_deph,
                                                         block_adc_signal,
                                                         delay_reph)
                            batches[batch_num].add_block(gr_ph_reph,
                                                         gr_sl_reph)
                            n_rd_points += n_rd
                            n_adc += 1
                            ph_idx += 1
                        elif self.echoMode=='Odd' and echo%2==1:
                            batches[batch_num].add_block(block_rf_refocusing,
                                                         block_gr_rd_reph,
                                                         gr_ph_deph,
                                                         gr_sl_deph,
                                                         delay_reph)
                            batches[batch_num].add_block(gr_ph_reph,
                                                         gr_sl_reph)
                        elif self.echoMode=='Even' and echo%2==1:
                            batches[batch_num].add_block(block_rf_refocusing,
                                                         block_gr_rd_reph,
                                                         gr_ph_deph,
                                                         gr_sl_deph,
                                                         block_adc_signal,
                                                         delay_reph)
                            batches[batch_num].add_block(gr_ph_reph,
                                                         gr_sl_reph)
                            n_rd_points += n_rd
                            n_adc += 1
                            ph_idx += 1
                        elif self.echoMode=='Even' and echo%2==0:
                            batches[batch_num].add_block(block_rf_refocusing,
                                                         block_gr_rd_reph,
                                                         gr_ph_deph,
                                                         gr_sl_deph,
                                                         delay_reph)
                            batches[batch_num].add_block(gr_ph_reph,
                                                         gr_sl_reph)

                        if ph_idx == n_ph:
                            break

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
                               frequency=hw.larmorFreq,  # MHz
                               bandwidth=bw,  # MHz
                               decimate='Normal',
                               hardware=True,
                               angulation=self.angulation
                               )

    def sequenceAnalysis(self, mode=None):
        """
        Analyzes the sequence data and performs several steps including data extraction, processing,
        noise estimation, dummy pulse separation, signal decimation, data reshaping, Fourier transforms,
        and image reconstruction.

        Parameters:
        mode (str, optional): A string indicating the mode of operation. If set to 'Standalone',
                               additional plotting will be performed. Default is None.

        The method performs the following key operations:
        1. Extracts relevant data from `self.mapVals`, including the data for readouts, signal,
           noise, and dummy pulses.
        2. Decimates the signal data to match the desired bandwidth and reorganizes the data for
           further analysis.
        3. Performs averaging on the data and reorganizes it according to sweep order.
        4. Computes the central line and adjusts for any drift in the k-space data.
        5. Applies zero-padding to the data to match the expected resolution.
        6. Computes the k-space trajectory (kRD, kPH, kSL) and applies the phase correction.
        7. Performs inverse Fourier transforms to reconstruct the 3D image data.
        8. Saves the processed data and produces plots for visualization based on the mode of operation.
        9. Optionally outputs sampled data and performs DICOM formatting for medical imaging storage.

        The method also handles the creation of various output results that can be plotted in the GUI,
        including signal curves, frequency spectra, and 3D images. It also updates the metadata for
        DICOM storage.

        The sequence of operations ensures the data is processed correctly according to the
        hardware setup and scan parameters.

        Results are saved in `self.mapVals` and visualized depending on the provided mode. The method
        also ensures proper handling of rotation angles and field-of-view (dfov) values, resetting
        them as necessary.
        """

        self.mode = mode

        # Get data
        axes_enable = self.mapVals['axes_enable']
        data_decimated = self.mapVals['data_decimated']
        n_rd1, n_ph1, n_sl1 = self.nPoints
        n_rd0, n_ph0, n_sl0 = self.nPoints0
        n_rd = n_rd1
        n_rd = n_rd + 2 * hw.addRdPoints
        n_batches = self.mapVals['n_batches']
        n_readouts = self.mapVals['n_readouts']

        # Get noise data, dummy data and signal data
        data_noise = []
        data_dummy = []
        data_signal = []
        points_per_rd = n_rd
        points_per_train = points_per_rd * self.etl
        idx_0 = 0
        idx_1 = 0
        for batch in range(n_batches):
            n_rds = n_readouts[batch]
            for scan in range(self.nScans):
                idx_1 += n_rds
                data_prov = data_decimated[idx_0:idx_1]
                data_noise = np.concatenate((data_noise, data_prov[0:points_per_rd]), axis=0)
                if self.dummyPulses > 0:
                    data_dummy = np.concatenate((data_dummy, data_prov[points_per_rd:points_per_rd+points_per_train]), axis=0)
                data_signal = np.concatenate((data_signal, data_prov[points_per_rd+points_per_train::]), axis=0)
                idx_0 = idx_1
            n_readouts[batch] += -n_rd - n_rd * self.etl
        self.mapVals['data_noise'] = data_noise
        self.mapVals['data_dummy'] = data_dummy
        self.mapVals['data_signal'] = data_signal

        # Save results
        self.saveRawData()

        self.output = []

        if self.mode == 'Standalone':
            self.plotResults()

        return self.output

    def get_sweep_order(self):
        # Phase index
        if self.nPoints[1]>1:
            ph_idx = np.arange(self.nPoints[1])
            start = (self.nPoints[1] - self.nPoints0[1]) // 2
            end = start + self.nPoints0[1]
            ph_idx = np.delete(ph_idx, np.s_[start:end])
            n = np.size(ph_idx)
            ph_p = ph_idx[n//2:]
            ph_n = ph_idx[:n//2]
            ph_n = ph_n[::-1]
            n = np.size(ph_p)
            n_trains = n // self.etl
            ph_steps = n // n_trains
            ph_p_reordered = []
            ph_n_reordered = []
            for ii in range(n_trains):
                ph_p_partial = ph_p[ii::n_trains]
                ph_n_partial = ph_n[ii::n_trains]
                for jj in range(ph_steps):
                    ph_p_reordered.append(ph_p_partial[jj])
                    ph_n_reordered.append(ph_n_partial[jj])
            ph_reordered = ph_p_reordered
            ph_reordered.extend(ph_n_reordered)
        else:
            ph_reordered = [0]

        # Slice index
        if self.nPoints[2]>1:
            sl_idx = np.arange(self.nPoints[2])
            start = (self.nPoints[2] - self.nPoints0[2]) // 2
            end = start + self.nPoints0[2]
            sl_idx = np.delete(sl_idx, np.s_[start:end])
        else:
            sl_idx = [0]

        return ph_reordered, sl_idx

if __name__ == '__main__':
    seq = RareOuter()
    seq.sequenceAtributes()
    # ind_ph, ind_sl = seq.get_sweep_order()
    seq.sequenceRun(plot_seq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
