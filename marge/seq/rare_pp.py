"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: rare sequence class
"""

import os
import sys

from phantominator import shepp_logan

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
import marge.configs.hw_config as hw # Import the scanner hardware config
import marge.configs.units as units
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from marge.marge_utils import utils

from datetime import datetime
import ismrmrd
import ismrmrd.xsd
import datetime
import ctypes
from marga_pulseq.interpreter import PSInterpreter
import pypulseq as pp

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class RarePyPulseq(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RarePyPulseq, self).__init__()
        # Input the parameters
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
        self.addParameter(key='seqName', string='RAREInfo', val='RarePyPulseq')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='pypulseq', string='PyPulseq', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM') ## number of scans 
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, units=units.kHz, field='RF')
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
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[40, 40, 10], field='IM')
        self.addParameter(key='angle', string='Angle (º)', val=0.0, field='IM')
        self.addParameter(key='rotationAxis', string='Rotation axis', val=[0, 0, 1], field='IM')
        self.addParameter(key='etl', string='Echo train length', val=4, field='SEQ') ## nm of peaks in 1 repetition
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, units=units.ms, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[2, 1, 0], field='IM', tip="0=x, 1=y, 2=z")
        self.addParameter(key='axesEnable', string='Axes enable', val=[1, 1, 1], tip="Use 0 for directions with matrix size 1, use 1 otherwise.")
        self.addParameter(key='sweepMode', string='Sweep mode', val=1, field='SEQ', tip="0: sweep from -kmax to kmax. 1: sweep from 0 to kmax. 2: sweep from kmax to 0")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=5.0, units=units.ms, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='phGradTime', string='Ph gradient time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')
        self.addParameter(key='rfPhase', string='RF phase (º)', val=0.0, field='OTH')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=1, field='SEQ', tip="Use last dummy pulse to calibrate k = 0")
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='parFourierFraction', string='Partial fourier fraction', val=0.7, field='OTH', tip="Fraction of k planes aquired in slice direction")
        self.addParameter(key='echo_shift', string='Echo time shift', val=0.0, units=units.us, field='OTH', tip='Shift the gradient echo time respect to the spin echo time.')
        self.addParameter(key='unlock_orientation', string='Unlock image orientation', val=0, field='OTH', tip='0: Images oriented according to standard. 1: Image raw orientation')
        self.addParameter(key='bm4d', string='BM4D std', val=-1, field='PRO',
                          tip='Perform BM4D filter: -1 -> do not apply, 0 -> automatic std calculation, >0 to use manual'
                              ' std value.')
        self.addParameter(key='cosbell_factor', string='Cosbell factor', val=0, field='PRO',
                          tip='Perform cosbell filter in k-space')
        self.addParameter(key='padding', string='Final size (rd, ph, sl)' , val=[0, 0, 0], field='PRO',
                          tip='Final matrix size after processing zero padding')
        self.addParameter(key='partial_method', string='Partial recon method', val='POCS', field='PRO',
                          tip="'zero' -> zero padding, pocs -> Projection Onto Convex Sets")
        self.addParameter(key='angulation', string='Angulation', val=1, field='OTH',
                          tip='1: Consider FOV angulation. 0: Keep the image unangled (it reduced the number of instructions).')

        self.acq = ismrmrd.Acquisition()
        self.img = ismrmrd.Image()
        self.header = ismrmrd.xsd.ismrmrdHeader()
        
       
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
        self.dfov = self.getFovDisplacement()
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
        self.freqOffset = self.freqOffset*1e6  # MHz
        resolution = self.fov/self.nPoints
        rf_ex_amp = self.rfExFA/(self.rfExTime*1e6*hw.b1Efficiency)*np.pi/180
        rf_re_amp = self.rfReFA/(self.rfReTime*1e6*hw.b1Efficiency)*np.pi/180
        self.mapVals['rf_ex_amp'] = rf_ex_amp
        self.mapVals['rf_re_amp'] = rf_re_amp
        self.mapVals['resolution'] = resolution
        self.mapVals['grad_rise_time'] = hw.grad_rise_time
        self.mapVals['addRdPoints'] = hw.addRdPoints
        self.mapVals['larmorFreq'] = hw.larmorFreq + self.freqOffset
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
        n_sl = self.nPoints[2]

        # ETL if etl>n_ph
        if self.etl>n_ph:
            self.etl = n_ph

        # Miscellaneous
        n_rd_points_per_train = self.etl * n_rd

        # par_acq_lines in case par_acq_lines = 0
        par_acq_lines = int(int(self.nPoints[2]*self.parFourierFraction)-self.nPoints[2]/2)
        self.mapVals['partialAcquisition'] = par_acq_lines

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

        # Now fix the number of slices to partially acquired k-space
        n_sl = (int(self.nPoints[2]/2)+par_acq_lines)*axes_enable[2]+(1-axes_enable[2])
        print("Number of acquired slices: %i" % n_sl)

        # Set phase vector to given sweep mode
        ind = self.getIndex(self.etl, n_ph, self.sweepMode)
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
                               frequency=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
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
        data_over = self.mapVals['data_over']
        data_decimated = self.mapVals['data_decimated']
        n_rd, n_ph, n_sl = self.nPoints
        n_rd = n_rd + 2 * hw.addRdPoints
        n_sl = (n_sl // 2 + self.mapVals['partialAcquisition'] * axes_enable[2] + (1 - axes_enable[2]))
        n_batches = self.mapVals['n_batches']
        n_readouts = self.mapVals['n_readouts']
        ind = self.getParameter('sweepOrder')

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

        # Decimate data to get signal in desired bandwidth
        data_full = data_signal

        # Reorganize data_full
        data_prov = np.zeros(shape=[self.nScans, n_sl * n_ph * n_rd], dtype=complex)
        if n_batches > 1:
            data_full_a = data_full[0:sum(n_readouts[0:-1]) * self.nScans]
            data_full_b = data_full[sum(n_readouts[0:-1]) * self.nScans:]
            data_full_a = np.reshape(data_full_a, newshape=(n_batches - 1, self.nScans, -1, n_rd))
            data_full_b = np.reshape(data_full_b, newshape=(1, self.nScans, -1, n_rd))
            for scan in range(self.nScans):
                data_scan_a = np.reshape(data_full_a[:, scan, :, :], -1)
                data_scan_b = np.reshape(data_full_b[:, scan, :, :], -1)
                data_prov[scan, :] = np.concatenate((data_scan_a, data_scan_b), axis=0)
        else:
            data_full = np.reshape(data_full, newshape=(1, self.nScans, -1, n_rd))
            for scan in range(self.nScans):
                data_prov[scan, :] = np.reshape(data_full[:, scan, :, :], -1)
        data_full = np.reshape(data_prov, -1)

        # Save data_full to save it in .h5
        self.data_fullmat = data_full

        # Get index for krd = 0
        # Average data
        data_prov = np.reshape(data_full, newshape=(self.nScans, n_rd * n_ph * n_sl))
        data_prov = np.average(data_prov, axis=0)
        # Reorganize the data according to sweep mode
        data_prov = np.reshape(data_prov, newshape=(n_sl, n_ph, n_rd))
        data_temp = np.zeros_like(data_prov)
        for ii in range(n_ph):
            data_temp[:, ind[ii], :] = data_prov[:, ii, :]
        data_prov = data_temp
        # Get central line
        data_prov = data_prov[int(self.nPoints[2] / 2), int(n_ph / 2), :]
        ind_krd_0 = np.argmax(np.abs(data_prov))
        if ind_krd_0 < n_rd / 2 - hw.addRdPoints or ind_krd_0 > n_rd / 2 + hw.addRdPoints:
            ind_krd_0 = int(n_rd / 2)

        # Get individual images
        data_full = np.reshape(data_full, newshape=(self.nScans, n_sl, n_ph, n_rd))
        data_full = data_full[:, :, :, ind_krd_0 - int(self.nPoints[0] / 2):ind_krd_0 + int(self.nPoints[0] / 2)]
        data_temp = np.zeros_like(data_full)
        for ii in range(n_ph):
            data_temp[:, :, ind[ii], :] = data_full[:, :, ii, :]
        data_full = data_temp
        self.mapVals['data_full'] = data_full

        # Average data
        data = np.average(data_full, axis=0)

        # Do zero padding
        data_temp = np.zeros(shape=(self.nPoints[2], self.nPoints[1], self.nPoints[0]), dtype=complex)
        data_temp[0:n_sl, :, :] = data
        if self.demo:
            phantom_image = shepp_logan((self.nPoints[2], self.nPoints[1], self.nPoints[0]))
            data_temp = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(phantom_image)))
        data = np.reshape(data_temp, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))

        # Fix the position of the sample according to dfov
        bw = self.getParameter('bw_MHz')
        time_vector = np.linspace(-self.nPoints[0] / bw / 2 + 0.5 / bw, self.nPoints[0] / bw / 2 - 0.5 / bw,
                                       self.nPoints[0]) * 1e-6  # s
        kMax = np.array(self.nPoints) / (2 * np.array(self.fov)) * np.array(self.mapVals['axes_enable'])
        kRD = time_vector * hw.gammaB * self.getParameter('rd_grad_amplitude')
        kPH = np.linspace(-kMax[1], kMax[1], num=self.nPoints[1], endpoint=False)
        kSL = np.linspace(-kMax[2], kMax[2], num=self.nPoints[2], endpoint=False)
        kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
        kRD = np.reshape(kRD, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))
        kPH = np.reshape(kPH, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))
        kSL = np.reshape(kSL, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))
        dPhase = np.exp(2 * np.pi * 1j * (self.dfov[0] * kRD + self.dfov[1] * kPH + self.dfov[2] * kSL))
        data = np.reshape(data * dPhase, newshape=(self.nPoints[2], self.nPoints[1], self.nPoints[0]))
        self.mapVals['kSpace3D'] = data
        img = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(data)))
        self.mapVals['image3D'] = img
        data = np.reshape(data, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))

        # Create sampled data
        kRD = np.reshape(kRD, newshape=(self.nPoints[0] * self.nPoints[1] * self.nPoints[2], 1))
        kPH = np.reshape(kPH, newshape=(self.nPoints[0] * self.nPoints[1] * self.nPoints[2], 1))
        kSL = np.reshape(kSL, newshape=(self.nPoints[0] * self.nPoints[1] * self.nPoints[2], 1))
        data = np.reshape(data, newshape=(self.nPoints[0] * self.nPoints[1] * self.nPoints[2], 1))
        self.mapVals['kMax_1/m'] = kMax
        self.mapVals['sampled'] = np.concatenate((kRD, kPH, kSL, data), axis=1)
        self.mapVals['sampledCartesian'] = self.mapVals['sampled']  # To sweep
        data = np.reshape(data, newshape=(self.nPoints[2], self.nPoints[1], self.nPoints[0]))

        axes_enable = self.mapVals['axes_enable']

        # Get axes in strings
        axes = self.mapVals['axesOrientation']
        axesDict = {'x':0, 'y':1, 'z':2}
        axesKeys = list(axesDict.keys())
        axesVals = list(axesDict.values())
        axesStr = ['','','']
        n = 0
        for val in axes:
            index = axesVals.index(val)
            axesStr[n] = axesKeys[index]
            n += 1

        if (axes_enable[1] == 0 and axes_enable[2] == 0):
            bw = self.mapVals['bw']*1e-3 # kHz
            acqTime = self.mapVals['acqTime'] # ms
            tVector = np.linspace(-acqTime/2, acqTime/2, self.nPoints[0])
            sVector = self.mapVals['sampled'][:, 3]
            fVector = np.linspace(-bw/2, bw/2, self.nPoints[0])
            iVector = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(sVector)))

            # Plots to show into the GUI
            result_1 = {}
            result_1['widget'] = 'curve'
            result_1['xData'] = tVector
            result_1['yData'] = [np.abs(sVector), np.real(sVector), np.imag(sVector)]
            result_1['xLabel'] = 'Time (ms)'
            result_1['yLabel'] = 'Signal amplitude (mV)'
            result_1['title'] = "Signal"
            result_1['legend'] = ['Magnitude', 'Real', 'Imaginary']
            result_1['row'] = 0
            result_1['col'] = 0

            result_2 = {}
            result_2['widget'] = 'curve'
            result_2['xData'] = fVector
            result_2['yData'] = [np.abs(iVector)]
            result_2['xLabel'] = 'Frequency (kHz)'
            result_2['yLabel'] = "Amplitude (a.u.)"
            result_2['title'] = "Spectrum"
            result_2['legend'] = ['Spectrum magnitude']
            result_2['row'] = 1
            result_2['col'] = 0

            self.output = [result_1, result_2]
            
        else:
            # Plot image
            image = np.abs(self.mapVals['image3D'])
            image = image/np.max(np.reshape(image,-1))*100

            # Image plot
            if self.mapVals['unlock_orientation'] == 0:
                result_1, _, _ = utils.fix_image_orientation(image, axes=self.axesOrientation)
                result_1['row'] = 0
                result_1['col'] = 0
            else:
                result_1 = {'widget': 'image', 'data': image, 'xLabel': "%s" % axesStr[1],
                            'yLabel': "%s" % axesStr[0], 'title': "k-Space", 'row': 0, 'col': 0}

            # k-space plot
            result_2 = {'widget': 'image'}
            if self.parFourierFraction==1:
                result_2['data'] = np.log10(np.abs(self.mapVals['kSpace3D']))
            else:
                result_2['data'] = np.abs(self.mapVals['kSpace3D'])
            result_2['xLabel'] = "k%s"%axesStr[1]
            result_2['yLabel'] = "k%s"%axesStr[0]
            result_2['title'] = "k-Space"
            result_2['row'] = 0
            result_2['col'] = 1

            # Dicom parameters
            self.meta_data["RepetitionTime"] = self.mapVals['repetitionTime']
            self.meta_data["EchoTime"] = self.mapVals['echoSpacing']
            self.meta_data["EchoTrainLength"] = self.mapVals['etl']

            # Add results into the output attribute (result_1 must be the image to save in dicom)
            self.output = [result_1, result_2]

        # Reset rotation angle and dfov to zero
        self.mapVals['angle'] = self.angle
        self.mapVals['dfov'] = np.array(self.mapVals['dfov'])
        self.mapVals['dfov'][self.axesOrientation] = self.dfov.reshape(-1)
        self.mapVals['dfov'] = list(self.mapVals['dfov'])

        # Save results
        self.saveRawData()
        self.save_ismrmrd()

        self.mapVals['angle'] = 0.0
        self.mapVals['dfov'] = [0.0, 0.0, 0.0]
        try:
            self.sequence_list['RarePyPulseq'].mapVals['angle'] = 0.0
            self.sequence_list['RarePyPulseq'].mapVals['dfov'] = [0.0, 0.0, 0.0]
        except:
            pass
        hw.dfov = [0.0, 0.0, 0.0]

        if self.mode == 'Standalone':
            self.plotResults()

        return self.output
        
    def save_ismrmrd(self):
        """
        Save the current instance's data in ISMRMRD format.

        This method saves the raw data, header information, and reconstructed images to an HDF5 file
        using the ISMRMRD (Image Storage and Reconstruction format for MR Data) format.

        Steps performed:
        1. Generate a timestamp-based filename and directory path for the output file.
        2. Initialize the ISMRMRD dataset with the generated path.
        3. Populate the header and write the XML header to the dataset. Informations can be added.
        4. Reshape the raw data matrix and iterate over scans, slices, and phases to write each acquisition. WARNING : RARE sequence follows ind order to fill the k-space.
        5. Set acquisition flags and properties.
        6. Append the acquisition data to the dataset.
        7. Reshape and save the reconstructed images.
        8. Close the dataset.

        Attribute:
        - self.data_full_mat (numpy.array): Full matrix of raw data to be reshaped and saved.

        Returns:
        None. It creates an HDF5 file with the ISMRMRD format.
        """
        
        directory_rmd = self.directory_rmd
        name = datetime.datetime.now()
        name_string = name.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3]
        self.mapVals['name_string'] = name_string
        if hasattr(self, 'raw_data_name'):
            file_name = "%s.%s" % (self.raw_data_name, name_string)
        else:
            self.raw_data_name = self.mapVals['seqName']
            file_name = "%s.%s" % (self.mapVals['seqName'], name_string)
            
        path= "%s/%s.h5" % (directory_rmd, file_name)
        
        dset = ismrmrd.Dataset(path, f'/dataset', True) # Create the dataset

        etl = self.mapVals['etl']
        axes_enable = self.mapVals['axes_enable']
        n_rd = self.nPoints[0]
        n_ph = self.nPoints[1]
        n_sl = (((self.nPoints[2] // 2) + self.mapVals['partialAcquisition']) * axes_enable[2] + (1 - axes_enable[2]))
        ind = self.getIndex(self.etl, n_ph, self.sweepMode)
        nRep = (n_ph//etl)*n_sl
        bw = self.mapVals['bw_MHz']
        
        axesOrientation = self.axesOrientation
        axesOrientation_list = axesOrientation.tolist()

        read_dir = [0, 0, 0]
        phase_dir = [0, 0, 0]
        slice_dir = [0, 0, 0]

        read_dir[axesOrientation_list.index(0)] = 1
        phase_dir[axesOrientation_list.index(1)] = 1
        slice_dir[axesOrientation_list.index(2)] = 1
        
        # Experimental Conditions field
        exp = ismrmrd.xsd.experimentalConditionsType() 
        magneticFieldStrength = hw.larmorFreq*1e6/hw.gammaB
        exp.H1resonanceFrequency_Hz = hw.larmorFreq

        self.header.experimentalConditions = exp 

        # Acquisition System Information field
        sys = ismrmrd.xsd.acquisitionSystemInformationType() 
        sys.receiverChannels = 1 
        self.header.acquisitionSystemInformation = sys


        # Encoding field can be filled if needed
        encoding = ismrmrd.xsd.encodingType()  
        encoding.trajectory = ismrmrd.xsd.trajectoryType.CARTESIAN
        #encoding.trajectory =ismrmrd.xsd.trajectoryType[data.processing.trajectory.upper()]
        
        dset.write_xml_header(self.header.toXML()) # Write the header to the dataset
                
        
        
        new_data = np.zeros((n_ph * n_sl * self.nScans, n_rd + 2*hw.addRdPoints))
        new_data = np.reshape(self.data_fullmat, (self.nScans, n_sl, n_ph, n_rd+ 2*hw.addRdPoints))
        
        counter=0  
        for scan in range(self.nScans):
            for slice_idx in range(n_sl):
                for phase_idx in range(n_ph):
                    
                    line = new_data[scan, slice_idx, phase_idx, :]
                    line2d = np.reshape(line, (1, n_rd+2*hw.addRdPoints))
                    acq = ismrmrd.Acquisition.from_array(line2d, None)
                    
                    index_in_repetition = phase_idx % etl
                    current_repetition = (phase_idx // etl) + (slice_idx * (n_ph // etl))
                    
                    acq.clearAllFlags()
                    
                    if index_in_repetition == 0: 
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_CONTRAST)
                    elif index_in_repetition == etl - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_CONTRAST)
                    
                    if ind[phase_idx]== 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_PHASE)
                    elif ind[phase_idx] == n_ph - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_PHASE)
                    
                    if slice_idx == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                    elif slice_idx == n_sl - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                        
                    if int(current_repetition) == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_REPETITION)
                    elif int(current_repetition) == nRep - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)
                        
                    if scan == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_AVERAGE)
                    elif scan == self.nScans-1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_AVERAGE)
                    
                    
                    counter += 1 
                    
                    # +1 to start at 1 instead of 0
                    acq.idx.repetition = int(current_repetition + 1)
                    acq.idx.kspace_encode_step_1 = ind[phase_idx]+1 # phase
                    acq.idx.slice = slice_idx + 1
                    acq.idx.contrast = index_in_repetition + 1
                    acq.idx.average = scan + 1 # scan
                    
                    acq.scan_counter = counter
                    acq.discard_pre = hw.addRdPoints
                    acq.discard_post = hw.addRdPoints
                    acq.sample_time_us = 1/bw
                    self.dfov = np.array(self.dfov)
                    acq.position = (ctypes.c_float * 3)(*self.dfov.flatten())

                    
                    acq.read_dir = (ctypes.c_float * 3)(*read_dir)
                    acq.phase_dir = (ctypes.c_float * 3)(*phase_dir)
                    acq.slice_dir = (ctypes.c_float * 3)(*slice_dir)
                    
                    dset.append_acquisition(acq) # Append the acquisition to the dataset
                        
                        
        image=self.mapVals['image3D']
        image_reshaped = np.reshape(image, (self.nPoints[::-1]))
        
        for slice_idx in range (n_sl): ## image3d does not have scan dimension
            
            image_slice = image_reshaped[slice_idx, :, :]
            img = ismrmrd.Image.from_array(image_slice)
            img.transpose = False
            img.field_of_view = (ctypes.c_float * 3)(*(self.fov)*10) # mm
           
            img.position = (ctypes.c_float * 3)(*self.dfov)
            
            # img.data_type= 8 ## COMPLEX FLOAT
            img.image_type = 5 ## COMPLEX
            
            
            
            img.read_dir = (ctypes.c_float * 3)(*read_dir)
            img.phase_dir = (ctypes.c_float * 3)(*phase_dir)
            img.slice_dir = (ctypes.c_float * 3)(*slice_dir)
            
            dset.append_image(f"image_raw", img) # Append the image to the dataset
                
        
        dset.close()    


if __name__ == '__main__':
    seq = RarePyPulseq()
    seq.sequenceAtributes()
    seq.sequenceRun(plot_seq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
