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
import marge.marcos.marcos_client.experiment
import scipy.signal as sig
import marge.configs.hw_config as hw # Import the scanner hardware config
import marge.configs.units as units
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.

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

class RARE_T2prep_pp(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RARE_T2prep_pp, self).__init__()
        # Input the parameters
        self.nScans = None
        self.spoiler_amp = None
        self.rdDephTime = None
        self.spoiler_duration = None
        self.spoiler_delay = None
        self.dummyPulses = None
        self.spoilerTime = None
        self.repetitionTime = None
        self.echoTime = None
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
        self.axesOrientation = None
        self.addParameter(key='seqName', string='RAREInfo', val='RARE_T2prep_pp')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=False)
        self.addParameter(key='pypulseq', string='PyPulseq', val=False)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, units=units.kHz, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=50.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=100.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, units=units.ms, field='SEQ', tip='Echo spacing for the echo train')
        self.addParameter(key='echoTime', string='Echo Time (ms)', val=10.0, units=units.ms, field='SEQ', tip='Echo time for the preparation pulse')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=300., units=units.ms, field='SEQ', tip="0 to ommit this pulse")
        self.addParameter(key='fov', string='FOV[x,y,z] (cm)', val=[12.0, 12.0, 12.0], units=units.cm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM', tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[60, 60, 1], field='IM')
        self.addParameter(key='angle', string='Angle (º)', val=0.0, field='IM')
        self.addParameter(key='rotationAxis', string='Rotation axis', val=[0, 0, 1], field='IM')
        self.addParameter(key='etl', string='Echo train length', val=6, field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, units=units.ms, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[2, 1, 0], field='IM', tip="0=x, 1=y, 2=z")
        self.addParameter(key='axesEnable', string='Axes enable', val=[1, 1, 1], tip="Use 0 for directions with matrix size 1, use 1 otherwise.")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=5.0, units=units.ms, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='phGradTime', string='Ph gradient time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')
        self.addParameter(key='rfPhase', string='RF phase (º)', val=0.0, field='OTH')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=1, field='SEQ', tip="Use last dummy pulse to calibrate k = 0")
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='parFourierFraction', string='Partial fourier fraction', val=1.0, field='OTH', tip="Fraction of k planes aquired in slice direction")
        self.addParameter(key='echo_shift', string='Echo time shift', val=0.0, units=units.us, field='OTH', tip='Shift the gradient echo time respect to the spin echo time.')
        self.addParameter(key='unlock_orientation', string='Unlock image orientation', val=0, field='OTH', tip='0: Images oriented according to standard. 1: Image raw orientation')
        self.addParameter(key='spoiler_amp', string='Spoiler amplitude (mT/m)', val=5.0, units=units.mTm, field='SEQ')
        self.addParameter(key='spoiler_duration', string='Spoiler duration (ms)', val=3.0, units=units.ms, field='SEQ')
        self.addParameter(key='spoiler_delay', string='Spoiler delay (ms)', val=10.0, units=units.ms, field='SEQ')
        self.acq = ismrmrd.Acquisition()
        self.img = ismrmrd.Image()
        self.header = ismrmrd.xsd.ismrmrdHeader()
        
       
    def sequenceInfo(self):
        print("3D RARE sequence with T2 preparation pulse")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain \n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        nPoints = np.array(self.mapVals['nPoints'])
        etl = self.mapVals['etl']
        repetitionTime = self.mapVals['repetitionTime']
        parFourierFraction = self.mapVals['parFourierFraction']

        # check if rf amplitude is too high
        rfExFA = self.mapVals['rfExFA'] / 180 * np.pi  # rads
        rfReFA = self.mapVals['rfReFA'] / 180 * np.pi  # rads
        rfExTime = self.mapVals['rfExTime']  # us
        rfReTime = self.mapVals['rfReTime']  # us
        rfExAmp = rfExFA / (rfExTime * hw.b1Efficiency)
        rfReAmp = rfReFA / (rfReTime * hw.b1Efficiency)
        if rfExAmp>1 or rfReAmp>1:
            print("ERROR: RF amplitude is too high, try with longer RF pulse time.")
            return(0)

        seqTime = nPoints[1]/etl*nPoints[2]*repetitionTime*1e-3*nScans*parFourierFraction/60
        seqTime = np.round(seqTime, decimals=1)
        return seqTime  # minutes, scanTime

        # TODO: check for min and max values for all fields

    def sequenceRun(self, plotSeq=False, demo=False, standalone=False):
        init_gpa = False  # Starts the gpa
        self.demo = demo
        self.plotSeq = plotSeq
        self.standalone = standalone
        print('RARE_T2_prep run')

        '''
        Step 1: Define the interpreter for FloSeq/PSInterpreter.
        The interpreter is responsible for converting the high-level pulse sequence description into low-level
        instructions for the scanner hardware. You will typically update the interpreter during scanner calibration.
        '''

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

        '''
        Step 2: Define system properties using PyPulseq (pp.Opts).
        These properties define the hardware capabilities of the MRI scanner, such as maximum gradient strengths,
        slew rates, and dead times. They are typically set based on the hardware configuration file (`hw_config`).
        '''

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
        axesEnable = []
        for ii in range(3):
            if self.nPoints[ii] == 1:
                axesEnable.append(0)
            else:
                axesEnable.append(1)
        self.mapVals['axes_enable_rd_ph_sl'] = axesEnable

        # Miscellaneous
        self.freqOffset = self.freqOffset*1e6 # MHz
        gradRiseTime = hw.grad_rise_time
        resolution = self.fov/self.nPoints
        rfExAmp = self.rfExFA/(self.rfExTime*1e6*hw.b1Efficiency)*np.pi/180
        rfReAmp = self.rfReFA/(self.rfReTime*1e6*hw.b1Efficiency)*np.pi/180
        self.mapVals['rfExAmp'] = rfExAmp
        self.mapVals['rfReAmp'] = rfReAmp
        self.mapVals['resolution'] = resolution
        self.mapVals['gradRiseTime'] = gradRiseTime
        self.mapVals['addRdPoints'] = hw.addRdPoints
        self.mapVals['larmorFreq'] = hw.larmorFreq + self.freqOffset
        if rfExAmp > 1 or rfReAmp > 1:
            print("ERROR: RF amplitude is too high, try with longer RF pulse time to reduce RF amplitude.")
            return 0

        # Matrix size
        nRD = self.nPoints[0] + 2 * hw.addRdPoints
        nPH = self.nPoints[1]
        nSL = self.nPoints[2]

        # ETL if etl>nPH
        if self.etl>nPH:
            self.etl = nPH

        # Miscellaneous
        n_rd_points_per_train = self.etl * nRD

        # parAcqLines in case parAcqLines = 0
        parAcqLines = int(int(self.nPoints[2]*self.parFourierFraction)-self.nPoints[2]/2)
        self.mapVals['partialAcquisition'] = parAcqLines

        # BW
        bw = self.nPoints[0] / self.acqTime * 1e-6  # MHz
        bw_ov = bw * hw.oversamplingFactor  # MHz
        sampling_period = 1 / bw_ov  # us

        # Readout gradient time
        if self.rdGradTime<self.acqTime:
            self.rdGradTime = self.acqTime
            print("Readout gradient time set to %0.1f ms" % (self.rdGradTime * 1e3))
            self.mapVals['rdGradTime'] = self.rdGradTime * 1e3  # ms

        # Phase and slice de- and re-phasing time
        if self.phGradTime == 0 or self.phGradTime > self.echoSpacing/2-self.rfExTime/2-self.rfReTime/2-2*gradRiseTime:
            self.phGradTime = self.echoSpacing/2-self.rfExTime/2-self.rfReTime/2-2*gradRiseTime
            print("Phase and slice gradient time set to %0.1f ms" % (self.phGradTime * 1e3))
            self.mapVals['phGradTime'] = self.phGradTime*1e3  # ms

        # Max gradient amplitude
        rdGradAmplitude = self.nPoints[0]/(hw.gammaB*self.fov[0]*self.acqTime)*axesEnable[0]
        phGradAmplitude = nPH/(2*hw.gammaB*self.fov[1]*(self.phGradTime+gradRiseTime))*axesEnable[1]
        slGradAmplitude = nSL/(2*hw.gammaB*self.fov[2]*(self.phGradTime+gradRiseTime))*axesEnable[2]
        self.mapVals['rdGradAmplitude'] = rdGradAmplitude
        self.mapVals['phGradAmplitude'] = phGradAmplitude
        self.mapVals['slGradAmplitude'] = slGradAmplitude

        # Readout dephasing amplitude
        rdDephAmplitude = 0.5*rdGradAmplitude*(gradRiseTime+self.rdGradTime)/(gradRiseTime+self.rdDephTime)
        self.mapVals['rdDephAmplitude'] = rdDephAmplitude
        print("Max rd gradient amplitude: %0.1f mT/m" % (max(rdGradAmplitude, rdDephAmplitude) * 1e3))
        print("Max ph gradient amplitude: %0.1f mT/m" % (phGradAmplitude * 1e3))
        print("Max sl gradient amplitude: %0.1f mT/m" % (slGradAmplitude * 1e3))

        # Phase and slice gradient vector
        phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
        slGradients = np.linspace(-slGradAmplitude,slGradAmplitude,num=nSL,endpoint=False)

        # Now fix the number of slices to partailly acquired k-space
        nSL = (int(self.nPoints[2]/2)+parAcqLines)*axesEnable[2]+(1-axesEnable[2])
        print("Number of acquired slices: %i" % nSL)

        # Set phase vector to given sweep mode
        ind = self.getIndex(self.etl, nPH, 1)
        self.mapVals['sweepOrder'] = ind
        phGradients = phGradients[ind]
        self.mapVals['phGradients'] = phGradients.copy()
        self.mapVals['slGradients'] = slGradients.copy()

        # Normalize gradient list
        if phGradAmplitude != 0:
            phGradients /= phGradAmplitude
        if slGradAmplitude != 0:
            slGradients /= slGradAmplitude

        # Get the rotation matrix
        rot = self.getRotationMatrix()
        gradAmp = np.array([0.0, 0.0, 0.0])
        gradAmp[self.axesOrientation[0]] = 1
        gradAmp = np.reshape(gradAmp, (3, 1))
        result = np.dot(rot, gradAmp)

        # Map the axis to "x", "y", and "z" according ot axesOrientation
        axes_map = {0: "x", 1: "y", 2: "z"}
        rd_channel = axes_map.get(self.axesOrientation[0], "")
        ph_channel = axes_map.get(self.axesOrientation[1], "")
        sl_channel = axes_map.get(self.axesOrientation[2], "")

        '''
        # Step 4: Define the experiment to get the true bandwidth
        # In this step, student need to get the real bandwidth used in the experiment. To get this bandwidth, an
        # experiment must be defined and the sampling period should be obtained using get_rx_ts()[0]
        '''

        if not self.demo:
            self.expt = ex.Experiment(lo_freq=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                                 rx_t=sampling_period,  # us
                                 init_gpa=init_gpa,
                                 gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                 auto_leds=True)
            sampling_period = self.expt.get_rx_ts()[0]  # us
            bw = 1 / sampling_period / hw.oversamplingFactor  # MHz
            sampling_time = sampling_period * nRD * hw.oversamplingFactor * 1e-6  # s
            print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))
            self.expt.__del__()
        else:
            sampling_time = sampling_period * nRD * hw.oversamplingFactor * 1e-6  # s
        self.mapVals['bw_MHz'] = bw
        self.mapVals['bw_ov_MHz'] = bw * hw.oversamplingFactor
        self.mapVals['sampling_period_us'] = sampling_period
        self.mapVals['sampling_time_s'] = sampling_time

        '''
        # Step 5: Define sequence blocks.
        # In this step, you will define the building blocks of the MRI sequence, including the RF pulses and gradient pulses.
        '''

        # First delay
        delay = self.repetitionTime - self.rfExTime / 2 - self.system.rf_dead_time - self.spoiler_delay
        delay_first = pp.make_delay(delay)

        # ADC to get noise
        delay = 40e-6
        block_adc_noise = pp.make_adc(num_samples=nRD * hw.oversamplingFactor,
                                      dwell=sampling_period * 1e-6,
                                      delay=delay)

        # Preparation: plus x pi/2 pulse
        flip_angle = np.pi / 2  # rads
        delay = 0
        block_rf_plus_x_pi2 = pp.make_block_pulse(
            flip_angle=flip_angle,
            system=self.system,
            duration=self.rfExTime,
            phase_offset=0.0,
            delay=delay,
            use='preparation',
        )
        delay = self.rfExTime / 2 - self.rfReTime / 2 + self.echoTime / 2
        delay_rf_plus_x_pi2 = pp.make_delay(delay)

        # Preparation: plus y pi pulse
        flip_angle = np.pi  # rads
        delay = 0
        block_rf_plus_y_pi = pp.make_block_pulse(
            flip_angle=flip_angle,
            system=self.system,
            duration=self.rfReTime,
            phase_offset=np.pi / 2,
            delay=delay,
            use='preparation',
        )
        delay = self.rfExTime / 2 - self.rfReTime / 2 + self.echoTime / 2
        delay_rf_plus_y_pi = pp.make_delay(delay)

        # Preparation: minus x pi/2 pulse
        flip_angle = np.pi / 2
        delay = 0
        block_rf_minus_x_pi2 = pp.make_block_pulse(
            flip_angle=flip_angle,
            system=self.system,
            duration=self.rfExTime,
            phase_offset=np.pi,
            delay=delay,
            use='preparation',
        )

        # Preparation: spoiler gradient x
        block_gr_x_spoiler = pp.make_trapezoid(
            channel="x",
            system=self.system,
            amplitude=self.spoiler_amp * hw.gammaB,
            flat_time=self.spoiler_duration,
            rise_time=hw.grad_rise_time,
            delay=0,
        )

        # Preparation: spoiler gradient y
        block_gr_y_spoiler = pp.make_trapezoid(
            channel="y",
            system=self.system,
            amplitude=self.spoiler_amp * hw.gammaB,
            flat_time=self.spoiler_duration,
            rise_time=hw.grad_rise_time,
            delay=0,
        )

        # Preparation: spoiler gradient x
        block_gr_z_spoiler = pp.make_trapezoid(
            channel="z",
            system=self.system,
            amplitude=self.spoiler_amp * hw.gammaB,
            flat_time=self.spoiler_duration,
            rise_time=hw.grad_rise_time,
            delay=0,
        )

        # Preparation: delay to next excitation
        delay_prep = pp.make_delay(self.spoiler_delay)

        # Excitation pulse
        flip_ex = self.rfExFA * np.pi / 180
        block_rf_excitation = pp.make_block_pulse(
            flip_angle=flip_ex,
            system=self.system,
            duration=self.rfExTime,
            phase_offset=0.0,
            delay=0.0,
            use='excitation'
        )

        # Dephasing gradient
        delay = self.system.rf_dead_time + self.rfExTime
        block_gr_rd_preph = pp.make_trapezoid(
            channel=rd_channel,
            system=self.system,
            amplitude=rdDephAmplitude * hw.gammaB,
            flat_time=self.rdDephTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Delay to refosucing pulse
        delay_preph = pp.make_delay(self.echoSpacing / 2 + self.rfExTime / 2 - self.rfReTime / 2)

        # Refocusing pulse
        flip_re = self.rfReFA * np.pi / 180
        block_rf_refocusing = pp.make_block_pulse(
            flip_angle=flip_re,
            system=self.system,
            duration=self.rfReTime,
            phase_offset=np.pi / 2,
            delay=0,
            use='refocusing'
        )

        # Delay to next refocusing pulse
        delay_reph = pp.make_delay(self.echoSpacing)

        # Phase gradient dephasing
        delay = self.system.rf_dead_time + self.rfReTime
        block_gr_ph_deph = pp.make_trapezoid(
            channel=ph_channel,
            system=self.system,
            amplitude=phGradAmplitude * hw.gammaB + float(phGradAmplitude==0),
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Slice gradient dephasing
        delay = self.system.rf_dead_time + self.rfReTime
        block_gr_sl_deph = pp.make_trapezoid(
            channel=sl_channel,
            system=self.system,
            amplitude=slGradAmplitude * hw.gammaB + float(slGradAmplitude==0),
            flat_time=self.phGradTime,
            delay=delay,
            rise_time=hw.grad_rise_time,
        )

        # Readout gradient
        delay = self.system.rf_dead_time + self.rfReTime / 2 + self.echoSpacing / 2 - self.rdGradTime / 2 - \
                hw.grad_rise_time
        block_gr_rd_reph = pp.make_trapezoid(
            channel=rd_channel,
            system=self.system,
            amplitude=rdGradAmplitude * hw.gammaB,
            flat_time=self.rdGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # ADC to get the signal
        delay = self.system.rf_dead_time + self.rfReTime / 2 + self.echoSpacing / 2 - sampling_time / 2
        block_adc_signal = pp.make_adc(num_samples=nRD * hw.oversamplingFactor,
                          dwell=sampling_period * 1e-6,
                          delay=delay)

        # Phase gradient rephasing
        delay = self.system.rf_dead_time + self.rfReTime / 2 - self.echoSpacing / 2 + sampling_time / 2
        block_gr_ph_reph = pp.make_trapezoid(
            channel=ph_channel,
            system=self.system,
            amplitude=phGradAmplitude * hw.gammaB + float(phGradAmplitude==0),
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Slice gradient rephasing
        delay = self.system.rf_dead_time + self.rfReTime / 2 - self.echoSpacing / 2 + sampling_time / 2
        block_gr_sl_reph = pp.make_trapezoid(
            channel=sl_channel,
            system=self.system,
            amplitude=slGradAmplitude * hw.gammaB + float(slGradAmplitude==0),
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Delay TR
        delay = self.repetitionTime + self.rfReTime / 2 - self.rfExTime / 2 - (self.etl + 0.5) * self.echoSpacing - \
            self.echoTime - self.spoiler_delay

        delay_tr = pp.make_delay(delay)

        '''
        # Step 6: Define your initializeBatch according to your sequence.
        # In this step, you will create the initializeBatch method to create dummy pulses that will be initialized for
        # each new batch.
        '''

        batches = {}
        n_rd_points_dict = {}  # Dictionary to track readout points for each batch
        self.n_rd_points = 0

        def initializeBatch(name="pp_1"):
            # Set n_rd_points to 0
            self.n_rd_points = 0

            # Instantiate pypulseq sequence object and save it into the batches dictionary
            batches[name] = pp.Sequence(self.system)

            # Set slice and phase gradients to 0
            gr_ph_deph = pp.scale_grad(block_gr_ph_deph, scale=0.0)
            gr_sl_deph = pp.scale_grad(block_gr_sl_deph, scale=0.0)
            gr_ph_reph = pp.scale_grad(block_gr_ph_reph, scale=0.0)
            gr_sl_reph = pp.scale_grad(block_gr_sl_reph, scale=0.0)

            batches[name].add_block(delay_first, block_adc_noise)
            self.n_rd_points += nRD

            # Create dummy pulses
            for dummy in range(self.dummyPulses):
                # Add preparation pulses
                batches[name].add_block(block_rf_plus_x_pi2,
                                        delay_rf_plus_x_pi2)
                batches[name].add_block(block_rf_plus_y_pi,
                                        delay_rf_plus_y_pi)
                batches[name].add_block(block_rf_minus_x_pi2)
                batches[name].add_block(block_gr_x_spoiler,
                                        block_gr_y_spoiler,
                                        block_gr_z_spoiler,
                                        delay_prep)

                # Add excitation pulse and readout de-phasing gradient
                batches[name].add_block(block_gr_rd_preph,
                                        block_rf_excitation,
                                        delay_preph)

                # Add echo train
                for echo in range(self.etl):
                    if dummy == self.dummyPulses-1:
                        batches[name].add_block(block_rf_refocusing,
                                                block_gr_rd_reph,
                                                gr_ph_deph,
                                                gr_sl_deph,
                                                block_adc_signal,
                                                delay_reph)
                        batches[name].add_block(gr_ph_reph,
                                                gr_sl_reph)
                        self.n_rd_points += nRD
                    else:
                        batches[name].add_block(block_rf_refocusing,
                                                block_gr_rd_reph,
                                                gr_ph_deph,
                                                gr_sl_deph,
                                                delay_reph)
                        batches[name].add_block(gr_ph_reph,
                                                gr_sl_reph)

                # Add time delay to next repetition
                batches[name].add_block(delay_tr)

        '''
        Step 7: Define your createBatches method.
        In this step you will populate the batches adding the blocks previously defined in step 4, and accounting for
        number of acquired points to check if a new batch is required.
        '''

        def createBatches():
            seq_idx = 0  # Sequence batch index
            batch_num = "batch_0"  # Initial batch name
            waveforms = {}  # Dictionary to store generated waveforms

            # Slice sweep
            for sl_idx in range(nSL):
                ph_idx = 0
                # Phase sweep
                while ph_idx < nPH:
                    # Check if a new batch is needed (either first batch or exceeding readout points limit)
                    if seq_idx == 0 or self.n_rd_points + n_rd_points_per_train > hw.maxRdPoints:
                        # If a previous batch exists, write and interpret it
                        if seq_idx > 0:
                            batches[batch_num].write(batch_num + ".seq")
                            waveforms[batch_num], param_dict = self.flo_interpreter.interpret(batch_num + ".seq")
                            print(f"{batch_num}.seq ready!")

                        # Update to the next batch
                        seq_idx += 1
                        n_rd_points_dict[batch_num] = self.n_rd_points  # Save readout points count
                        batch_num = f"batch_{seq_idx}"
                        initializeBatch(batch_num)  # Initialize new batch
                        print(f"Creating {batch_num}.seq...")

                    # Add preparation pulses
                    batches[batch_num].add_block(block_rf_plus_x_pi2,
                                            delay_rf_plus_x_pi2)
                    batches[batch_num].add_block(block_rf_plus_y_pi,
                                            delay_rf_plus_y_pi)
                    batches[batch_num].add_block(block_rf_minus_x_pi2)
                    batches[batch_num].add_block(block_gr_x_spoiler,
                                            block_gr_y_spoiler,
                                            block_gr_z_spoiler,
                                            delay_prep)

                    # Add excitation pulse and readout de-phasing gradient
                    batches[batch_num].add_block(block_gr_rd_preph,
                                            block_rf_excitation,
                                            delay_preph)

                    # Add echo train
                    for echo in range(self.etl):
                        # Fix the phase and slice amplitude
                        gr_ph_deph = pp.scale_grad(block_gr_ph_deph, phGradients[ph_idx])
                        gr_sl_deph = pp.scale_grad(block_gr_sl_deph, slGradients[sl_idx])
                        gr_ph_reph = pp.scale_grad(block_gr_ph_reph, - phGradients[ph_idx])
                        gr_sl_reph = pp.scale_grad(block_gr_sl_reph, - slGradients[sl_idx])

                        # Add blocks
                        batches[batch_num].add_block(block_rf_refocusing,
                                                block_gr_rd_reph,
                                                gr_ph_deph,
                                                gr_sl_deph,
                                                block_adc_signal,
                                                delay_reph)
                        batches[batch_num].add_block(gr_ph_reph,
                                                gr_sl_reph)
                        self.n_rd_points += nRD
                        ph_idx += 1

                    # Add time delay to next repetition
                    batches[batch_num].add_block(delay_tr)

            # After final repetition, save and interpret the last batch
            batches[batch_num].write(batch_num + ".seq")
            waveforms[batch_num], param_dict = self.flo_interpreter.interpret(batch_num + ".seq")
            print(f"{batch_num}.seq ready!")
            print(f"{len(batches)} batches created. Sequence ready!")

            # Update the number of acquired ponits in the last batch
            n_rd_points_dict.pop('batch_0')
            n_rd_points_dict[batch_num] = self.n_rd_points

            return waveforms, n_rd_points_dict

        ''' 
        Step 8: Run the batches
        This step will handle the different batches, run it and get the resulting data. This should not be modified.
        Oversampled data will be available in self.mapVals['data_over']
        '''
        waveforms, n_readouts = createBatches()
        return self.runBatches(waveforms,
                               n_readouts,
                               frequency=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                               bandwidth=bw_ov,  # MHz
                               )

    def sequenceAnalysis(self, mode=None):
        self.mode = mode

        # Get data
        data_over = self.mapVals['data_over']
        nRD, nPH, nSL = self.nPoints
        nRD = nRD + 2 * hw.addRdPoints
        n_batches = self.mapVals['n_batches']
        n_readouts = self.mapVals['n_readouts']
        ind = self.getParameter('sweepOrder')

        # Get noise data, dummy data and signal data
        data_noise = []
        data_dummy = []
        data_signal = []
        points_per_rd = nRD * hw.oversamplingFactor
        points_per_train = points_per_rd * self.etl
        idx_0 = 0
        idx_1 = 0
        for batch in range(n_batches):
            n_rds = n_readouts[batch] * hw.oversamplingFactor
            for scan in range(self.nScans):
                idx_1 += n_rds
                data_prov = data_over[idx_0:idx_1]
                data_noise = np.concatenate((data_noise, data_prov[0:points_per_rd]), axis=0)
                if self.dummyPulses > 0:
                    data_dummy = np.concatenate((data_dummy, data_prov[points_per_rd:points_per_rd+points_per_train]), axis=0)
                data_signal = np.concatenate((data_signal, data_prov[points_per_rd+points_per_train::]), axis=0)
                idx_0 = idx_1
            n_readouts[batch] += -nRD - nRD * self.etl
        self.mapVals['data_noise'] = data_noise
        self.mapVals['data_dummy'] = data_dummy
        self.mapVals['data_signal'] = data_signal

        # Decimate data to get signal in desired bandwidth
        data_full = sig.decimate(data_signal, hw.oversamplingFactor, ftype='fir', zero_phase=True)

        # Reorganize data_full
        data_prov = np.zeros(shape=[self.nScans, nSL * nPH * nRD], dtype=complex)
        if n_batches > 1:
            data_full_a = data_full[0:sum(n_readouts[0:-1]) * self.nScans]
            data_full_b = data_full[sum(n_readouts[0:-1]) * self.nScans:]
            data_full_a = np.reshape(data_full_a, newshape=(n_batches - 1, self.nScans, -1, nRD))
            data_full_b = np.reshape(data_full_b, newshape=(1, self.nScans, -1, nRD))
            for scan in range(self.nScans):
                data_scan_a = np.reshape(data_full_a[:, scan, :, :], -1)
                data_scan_b = np.reshape(data_full_b[:, scan, :, :], -1)
                data_prov[scan, :] = np.concatenate((data_scan_a, data_scan_b), axis=0)
        else:
            data_full = np.reshape(data_full, newshape=(1, self.nScans, -1, nRD))
            for scan in range(self.nScans):
                data_prov[scan, :] = np.reshape(data_full[:, scan, :, :], -1)
        data_full = np.reshape(data_prov, -1)

        # Save data_full to save it in .h5
        self.data_fullmat = data_full

        # Get index for krd = 0
        # Average data
        data_prov = np.reshape(data_full, newshape=(self.nScans, nRD * nPH * nSL))
        data_prov = np.average(data_prov, axis=0)
        # Reorganize the data acording to sweep mode
        data_prov = np.reshape(data_prov, newshape=(nSL, nPH, nRD))
        data_temp = np.zeros_like(data_prov)
        for ii in range(nPH):
            data_temp[:, ind[ii], :] = data_prov[:, ii, :]
        data_prov = data_temp
        # Get central line
        data_prov = data_prov[int(self.nPoints[2] / 2), int(nPH / 2), :]
        indkrd0 = np.argmax(np.abs(data_prov))
        if indkrd0 < nRD / 2 - hw.addRdPoints or indkrd0 > nRD / 2 + hw.addRdPoints:
            indkrd0 = int(nRD / 2)

        # Get individual images
        data_full = np.reshape(data_full, newshape=(self.nScans, nSL, nPH, nRD))
        data_full = data_full[:, :, :, indkrd0 - int(self.nPoints[0] / 2):indkrd0 + int(self.nPoints[0] / 2)]
        data_temp = np.zeros_like(data_full)
        for ii in range(nPH):
            data_temp[:, :, ind[ii], :] = data_full[:, :, ii, :]
        data_full = data_temp
        self.mapVals['data_full'] = data_full

        # Average data
        data = np.average(data_full, axis=0)

        # Do zero padding
        data_temp = np.zeros(shape=(self.nPoints[2], self.nPoints[1], self.nPoints[0]), dtype=complex)
        data_temp[0:nSL, :, :] = data
        data = np.reshape(data_temp, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))

        # Fix the position of the sample according to dfov
        bw = self.getParameter('bw_MHz')
        time_vector = np.linspace(-self.nPoints[0] / bw / 2 + 0.5 / bw, self.nPoints[0] / bw / 2 - 0.5 / bw,
                                       self.nPoints[0]) * 1e-6  # s
        kMax = np.array(self.nPoints) / (2 * np.array(self.fov)) * np.array(self.mapVals['axesEnable'])
        kRD = time_vector * hw.gammaB * self.getParameter('rdGradAmplitude')
        kPH = np.linspace(-kMax[1], kMax[1], num=self.nPoints[1], endpoint=False)
        kSL = np.linspace(-kMax[2], kMax[2], num=self.nPoints[2], endpoint=False)
        kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
        kRD = np.reshape(kRD, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))
        kPH = np.reshape(kPH, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))
        kSL = np.reshape(kSL, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))
        dPhase = np.exp(-2 * np.pi * 1j * (self.dfov[0] * kRD + self.dfov[1] * kPH + self.dfov[2] * kSL))
        data = np.reshape(data * dPhase, newshape=(self.nPoints[2], self.nPoints[1], self.nPoints[0]))
        self.mapVals['kSpace3D'] = data
        img = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
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

        nPoints = self.mapVals['nPoints']
        axesEnable = self.mapVals['axesEnable']

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

        if (axesEnable[1] == 0 and axesEnable[2] == 0):
            bw = self.mapVals['bw']*1e-3 # kHz
            acqTime = self.mapVals['acqTime'] # ms
            tVector = np.linspace(-acqTime/2, acqTime/2, nPoints[0])
            sVector = self.mapVals['sampled'][:, 3]
            fVector = np.linspace(-bw/2, bw/2, nPoints[0])
            iVector = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(sVector)))

            # Plots to show into the GUI
            result1 = {}
            result1['widget'] = 'curve'
            result1['xData'] = tVector
            result1['yData'] = [np.abs(sVector), np.real(sVector), np.imag(sVector)]
            result1['xLabel'] = 'Time (ms)'
            result1['yLabel'] = 'Signal amplitude (mV)'
            result1['title'] = "Signal"
            result1['legend'] = ['Magnitude', 'Real', 'Imaginary']
            result1['row'] = 0
            result1['col'] = 0

            result2 = {}
            result2['widget'] = 'curve'
            result2['xData'] = fVector
            result2['yData'] = [np.abs(iVector)]
            result2['xLabel'] = 'Frequency (kHz)'
            result2['yLabel'] = "Amplitude (a.u.)"
            result2['title'] = "Spectrum"
            result2['legend'] = ['Spectrum magnitude']
            result2['row'] = 1
            result2['col'] = 0

            self.output = [result1, result2]
            
        else:
            # Plot image
            image = np.abs(self.mapVals['image3D'])
            image = image/np.max(np.reshape(image,-1))*100

            # Image orientation
            imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            if not self.unlock_orientation: # Image orientation
                if self.axesOrientation[2] == 2:  # Sagittal
                    title = "Sagittal"
                    if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 1:  #OK
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        xLabel = "(-Y) A | PHASE | P (+Y)"
                        yLabel = "(-X) I | READOUT | S (+X)"
                        imageOrientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
                    else:
                        image = np.transpose(image, (0, 2, 1))
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        xLabel = "(-Y) A | READOUT | P (+Y)"
                        yLabel = "(-X) I | PHASE | S (+X)"
                        imageOrientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
                elif self.axesOrientation[2] == 1: # Coronal
                    title = "Coronal"
                    if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 2: #OK
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        image = np.flip(image, axis=0)
                        xLabel = "(+Z) R | PHASE | L (-Z)"
                        yLabel = "(-X) I | READOUT | S (+X)"
                        imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                    else:
                        image = np.transpose(image, (0, 2, 1))
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        image = np.flip(image, axis=0)
                        xLabel = "(+Z) R | READOUT | L (-Z)"
                        yLabel = "(-X) I | PHASE | S (+X)"
                        imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                elif self.axesOrientation[2] == 0:  # Transversal
                    title = "Transversal"
                    if self.axesOrientation[0] == 1 and self.axesOrientation[1] == 2:
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        xLabel = "(+Z) R | PHASE | L (-Z)"
                        yLabel = "(+Y) P | READOUT | A (-Y)"
                        imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                    else:  #OK
                        image = np.transpose(image, (0, 2, 1))
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        xLabel = "(+Z) R | READOUT | L (-Z)"
                        yLabel = "(+Y) P | PHASE | A (-Y)"
                        imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            else:
                xLabel = "%s axis" % axesStr[1]
                yLabel = "%s axis" % axesStr[0]
                title = "Image"

            result1 = {}
            result1['widget'] = 'image'
            result1['data'] = image
            result1['xLabel'] = xLabel
            result1['yLabel'] = yLabel
            result1['title'] = title
            result1['row'] = 0
            result1['col'] = 0

            result2 = {}
            result2['widget'] = 'image'
            if self.parFourierFraction==1:
                result2['data'] = np.log10(np.abs(self.mapVals['kSpace3D']))
            else:
                result2['data'] = np.abs(self.mapVals['kSpace3D'])
            result2['xLabel'] = "k%s"%axesStr[1]
            result2['yLabel'] = "k%s"%axesStr[0]
            result2['title'] = "k-Space"
            result2['row'] = 0
            result2['col'] = 1

            # DICOM TAGS
            # Image
            imageDICOM = np.transpose(image, (0, 2, 1))
            # If it is a 3d image
            if len(imageDICOM.shape) > 2:
                # Obtener dimensiones
                slices, rows, columns = imageDICOM.shape
                self.meta_data["Columns"] = columns
                self.meta_data["Rows"] = rows
                self.meta_data["NumberOfSlices"] = slices
                self.meta_data["NumberOfFrames"] = slices
            # if it is a 2d image
            else:
                # Obtener dimensiones
                rows, columns = imageDICOM.shape
                self.meta_data["Columns"] = columns
                self.meta_data["Rows"] = rows
                self.meta_data["NumberOfSlices"] = 1
                self.meta_data["NumberOfFrames"] = 1
            imgAbs = np.abs(imageDICOM)
            imgFullAbs = np.abs(imageDICOM) * (2 ** 15 - 1) / np.amax(np.abs(imageDICOM))
            x2 = np.amax(np.abs(imageDICOM))
            imgFullInt = np.int16(np.abs(imgFullAbs))
            imgFullInt = np.reshape(imgFullInt, (slices, rows, columns))
            arr = np.zeros((slices, rows, columns), dtype=np.int16)
            arr = imgFullInt
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

            # Add results into the output attribute (result1 must be the image to save in dicom)
            self.output = [result1, result2]

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
            self.sequence_list['RARE'].mapVals['angle'] = 0.0
            self.sequence_list['RARE'].mapVals['dfov'] = [0.0, 0.0, 0.0]
        except:
            pass
        hw.dfov = [0.0, 0.0, 0.0]

        if self.mode == 'Standalone':
            self.plotResults()

        return self.output

    def myPhantom(self):
        # Reorganize the fov
        n_pixels = self.nPoints[0]*self.nPoints[1]*self.nPoints[2]

        # Get x, y and z vectors in real (x, y, z) and relative (rd, ph, sl) coordinates
        rd = np.linspace(-self.fov[0] / 2, self.fov[0] / 2, self.nPoints[0])
        ph = np.linspace(-self.fov[1] / 2, self.fov[1] / 2, self.nPoints[1])
        if self.nPoints[2]==1:
            sl = sl = np.linspace(-0, 0, 1)
            p = np.array([0.01, 0.01, 0.0])
            p = p[self.axesOrientation]
        else:
            sl = np.linspace(-self.fov[2] / 2, self.fov[2] / 2, self.nPoints[2])
            p = np.array([0.01, 0.01, 0.01])
        ph, sl, rd = np.meshgrid(ph, sl, rd)
        rd = np.reshape(rd, (1, -1))
        ph = np.reshape(ph, (1, -1))
        sl = np.reshape(sl, (1, -1))
        pos_rela = np.concatenate((rd, ph, sl), axis=0)
        pos_real = pos_rela[self.axesOrientation, :]

        # Generate the phantom
        image = np.zeros((1, n_pixels))
        image = np.concatenate((pos_real, image), axis=0)
        r = 0.01
        for ii in range(n_pixels):
            d = np.sqrt((pos_real[0,ii] - p[0])**2 + (pos_real[1,ii] - p[1])**2 + (pos_real[2,ii] - p[2])**2)
            if d <= r:
                image[3, ii] = 1
        image_3d = np.reshape(image[3, :], self.nPoints[-1::-1])
        
        # Generate k-space
        kspace_3d = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(image_3d)))
        
        kspace = np.reshape(kspace_3d, (1, -1))
        
        return kspace
        
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
        
        nScans = self.mapVals['nScans']
        nPoints = np.array(self.mapVals['nPoints'])
        etl = self.mapVals['etl']
        nRD = self.nPoints[0]
        nPH = self.nPoints[1]
        nSL = self.nPoints[2]
        ind = self.getIndex(self.etl, nPH, 1)
        nRep = (nPH//etl)*nSL
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
                
        
        
        new_data = np.zeros((nPH * nSL * nScans, nRD + 2*hw.addRdPoints))
        new_data = np.reshape(self.data_fullmat, (nScans, nSL, nPH, nRD+ 2*hw.addRdPoints))
        
        counter=0  
        for scan in range(nScans):
            for slice_idx in range(nSL):
                for phase_idx in range(nPH):
                    
                    line = new_data[scan, slice_idx, phase_idx, :]
                    line2d = np.reshape(line, (1, nRD+2*hw.addRdPoints))
                    acq = ismrmrd.Acquisition.from_array(line2d, None)
                    
                    index_in_repetition = phase_idx % etl
                    current_repetition = (phase_idx // etl) + (slice_idx * (nPH // etl))
                    
                    acq.clearAllFlags()
                    
                    if index_in_repetition == 0: 
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_CONTRAST)
                    elif index_in_repetition == etl - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_CONTRAST)
                    
                    if ind[phase_idx]== 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_PHASE)
                    elif ind[phase_idx] == nPH - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_PHASE)
                    
                    if slice_idx == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                    elif slice_idx == nSL - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                        
                    if int(current_repetition) == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_REPETITION)
                    elif int(current_repetition) == nRep - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)
                        
                    if scan == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_AVERAGE)
                    elif scan == nScans-1:
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
        image_reshaped = np.reshape(image, (nSL, nPH, nRD))
        
        for slice_idx in range (nSL): ## image3d does not have scan dimension
            
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
    seq = RARE_T2prep_pp()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=True, demo=True, standalone=True)
