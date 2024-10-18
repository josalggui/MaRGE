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
import experiment as ex
import scipy.signal as sig
from scipy.stats import linregress
import configs.hw_config as hw # Import the scanner hardware config
import configs.units as units
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.

from datetime import datetime
import ismrmrd
import ismrmrd.xsd
import datetime
import ctypes
from flocra_pulseq.interpreter import PSInterpreter
import pypulseq as pp

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class RARE_pp(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RARE_pp, self).__init__()
        # Input the parameters
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
        self.addParameter(key='seqName', string='RAREInfo', val='RARE_pp')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM') ## number of scans 
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, units=units.kHz, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=50.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=100.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, units=units.ms, field='SEQ')
        self.addParameter(key='preExTime', string='Preexitation time (ms)', val=10.0, units=units.ms, field='SEQ')
        self.addParameter(key='inversionTime', string='Inversion time (ms)', val=10.0, units=units.ms, field='SEQ', tip="0 to ommit this pulse")
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=50., units=units.ms, field='SEQ', tip="0 to ommit this pulse")
        self.addParameter(key='fov', string='FOV[x,y,z] (cm)', val=[15.0, 15.0, 15.0], units=units.cm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM', tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[40, 20, 1], field='IM')
        self.addParameter(key='angle', string='Angle (º)', val=0.0, field='IM')
        self.addParameter(key='rotationAxis', string='Rotation axis', val=[0, 0, 1], field='IM')
        self.addParameter(key='etl', string='Echo train length', val=2, field='SEQ') ## nm of peaks in 1 repetition
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=2.0, units=units.ms, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[0, 1, 2], field='IM', tip="0=x, 1=y, 2=z")
        self.addParameter(key='axesEnable', string='Axes enable', val=[1, 1, 0], tip="Use 0 for directions with matrix size 1, use 1 otherwise.")
        self.addParameter(key='sweepMode', string='Sweep mode', val=1, field='SEQ', tip="0: sweep from -kmax to kmax. 1: sweep from 0 to kmax. 2: sweep from kmax to 0")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=2.5, units=units.ms, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='phGradTime', string='Ph gradient time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')
        self.addParameter(key='rfPhase', string='RF phase (º)', val=0.0, field='OTH')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=2, field='SEQ', tip="Use last dummy pulse to calibrate k = 0")
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='parFourierFraction', string='Partial fourier fraction', val=1.0, field='OTH', tip="Fraction of k planes aquired in slice direction")
        self.addParameter(key='echo_shift', string='Echo time shift', val=0.0, units=units.us, field='OTH', tip='Shift the gradient echo time respect to the spin echo time.')
        self.addParameter(key='unlock_orientation', string='Unlock image orientation', val=0, field='OTH', tip='0: Images oriented according to standard. 1: Image raw orientation')
        self.acq = ismrmrd.Acquisition()
        self.img = ismrmrd.Image()
        self.header = ismrmrd.xsd.ismrmrdHeader()
        
       
    def sequenceInfo(self):
        print("3D RARE sequence")
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

    def sequenceAtributes(self):
        super().sequenceAtributes()

        # Conversion of variables to non-multiplied units
        self.angle = self.angle * np.pi / 180 # rads

        # Add rotation, dfov and fov to the history
        self.rotation = self.rotationAxis.tolist()
        self.rotation.append(self.angle)
        self.rotations.append(self.rotation)
        self.dfovs.append(self.dfov.tolist())
        self.fovs.append(self.fov.tolist())

    def sequenceRun(self, plotSeq=False, demo=False, standalone=False):
        init_gpa = False  # Starts the gpa
        self.demo = demo
        print('RARE run')

        # Define the interpreter. It should be updated on calibration
        self.flo_interpreter = PSInterpreter(tx_warmup=hw.blkTime,  # us
                                             rf_center=hw.larmorFreq * 1e6,  # Hz
                                             rf_amp_max=hw.b1Efficiency / (2 * np.pi) * 1e6,  # Hz
                                             gx_max=hw.gFactor[0] * hw.gammaB,  # Hz/m
                                             gy_max=hw.gFactor[1] * hw.gammaB,  # Hz/m
                                             gz_max=hw.gFactor[2] * hw.gammaB,  # Hz/m
                                             grad_max=np.max(hw.gFactor) * hw.gammaB,  # Hz/m
                                             grad_t=hw.grad_raster_time*1e6,  # us
                                             )

        # Define system properties according to hw_config file
        self.system = pp.Opts(
            rf_dead_time=(hw.blkTime) * 1e-6,  # s
            rf_ringdown_time=hw.deadTime * 1e-6,  # s
            max_grad=hw.max_grad,  # mT/m
            grad_unit='mT/m',
            max_slew=hw.max_slew_rate,  # mT/m/ms
            slew_unit='mT/m/ms',
            grad_raster_time=hw.grad_raster_time,  # s
            rise_time=hw.grad_rise_time,  # s
            rf_raster_time=1e-6,
            block_duration_raster=1e-6
        )

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
        self.mapVals['axesEnable'] = axesEnable

        # Miscellaneous
        self.freqOffset = self.freqOffset*1e6 # MHz
        gradRiseTime = hw.grad_rise_time
        gSteps = hw.grad_steps
        randFactor = 0e-3                        # Random amplitude to add to the phase gradients
        resolution = self.fov/self.nPoints
        rfExAmp = self.rfExFA/(self.rfExTime*1e6*hw.b1Efficiency)*np.pi/180
        rfReAmp = self.rfReFA/(self.rfReTime*1e6*hw.b1Efficiency)*np.pi/180
        self.mapVals['rfExAmp'] = rfExAmp
        self.mapVals['rfReAmp'] = rfReAmp
        self.mapVals['resolution'] = resolution
        self.mapVals['gradRiseTime'] = gradRiseTime
        self.mapVals['randFactor'] = randFactor
        self.mapVals['addRdPoints'] = hw.addRdPoints
        self.mapVals['larmorFreq'] = hw.larmorFreq + self.freqOffset
        if rfExAmp > 1 or rfReAmp > 1:
            print("ERROR: RF amplitude is too high, try with longer RF pulse time.")
            return 0

        # Matrix size
        nRD = self.nPoints[0] + 2 * hw.addRdPoints
        nPH = self.nPoints[1]
        nSL = self.nPoints[2]

        # ETL if etl>nPH
        if self.etl>nPH:
            self.etl = nPH

        # Miscellaneous
        grad_reference = 0.001  # Used to normalize gradient shape
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

        # Add random displacemnt to phase encoding lines
        for ii in range(nPH):
            if ii<np.ceil(nPH/2-nPH/20) or ii>np.ceil(nPH/2+nPH/20):
                phGradients[ii] = phGradients[ii]+randFactor*np.random.randn()
        kPH = hw.gammaB*phGradients*(gradRiseTime+self.phGradTime)
        self.mapVals['slGradients'] = slGradients

        # Set phase vector to given sweep mode
        ind = self.getIndex(self.etl, nPH, self.sweepMode)
        self.mapVals['sweepOrder'] = ind
        phGradients = phGradients[ind]
        self.mapVals['phGradients'] = phGradients

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

        # Get the true sampling rate
        if not self.demo:
            expt = ex.Experiment(lo_freq=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                                 rx_t=sampling_period,  # us
                                 init_gpa=init_gpa,
                                 gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                 auto_leds=True)
            sampling_period = expt.get_rx_ts()[0]  # us
            bw = 1 / sampling_period / hw.oversamplingFactor  # MHz
            sampling_time = sampling_period * nRD * hw.oversamplingFactor * 1e-6  # s
            expt.__del__()
        else:
            sampling_time = sampling_period * nRD * hw.oversamplingFactor * 1e-6  # s

        ##########################
        # Create pypulseq blocks #
        ##########################

        if self.inversionTime==0 and self.preExTime==0:
            delay = self.repetitionTime - self.rfExTime / 2 - self.system.rf_dead_time
        elif self.inversionTime>0 and self.preExTime==0:
            delay = self.repetitionTime - self.inversionTime - self.rfReTime / 2 - self.system.rf_dead_time
        elif self.inversionTime==0 and self.preExTime>0:
            delay = self.repetitionTime - self.preExTime - self.rfExTime / 2 - self.system.rf_dead_time
        else:
            delay = self.repetitionTime - self.preExTime - self.inversionTime - self.rfExTime / 2 - self.system.rf_dead_time
        delay_first = pp.make_delay(delay)

        # ADC to get noise
        delay = 20e-6
        block_adc_noise = pp.make_adc(num_samples=nRD * hw.oversamplingFactor,
                                       dwell=sampling_period * 1e-6,
                                       delay=delay)

        # Pre-excitation pulse
        if self.preExTime>0:
            flip_pre = self.rfExFA * np.pi / 180
            delay = 0
            block_rf_pre_excitation = pp.make_block_pulse(
                flip_angle=flip_pre,
                system=self.system,
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
                system=self.system,
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
            system=self.system,
            duration=self.rfExTime,
            phase_offset=0.0,
            delay=0.0,
        )

        # Dephasing gradient
        delay = self.system.rf_dead_time + self.rfExTime - hw.gradDelay * 1e-6
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
        )

        # Delay to next refocusing pulse
        delay_reph = pp.make_delay(self.echoSpacing)

        # Phase gradient dephasing
        delay = self.system.rf_dead_time + self.rfReTime - hw.gradDelay * 1e-6
        block_gr_ph_deph = pp.make_trapezoid(
            channel=ph_channel,
            system=self.system,
            amplitude=grad_reference * hw.gammaB,
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Slice gradient dephasing
        delay = self.system.rf_dead_time + self.rfReTime - hw.gradDelay * 1e-6
        block_gr_sl_deph = pp.make_trapezoid(
            channel=sl_channel,
            system=self.system,
            amplitude=grad_reference * hw.gammaB,
            flat_time=self.phGradTime,
            delay=delay,
            rise_time=hw.grad_rise_time,
        )

        # Readout gradient
        delay = self.system.rf_dead_time + self.rfReTime / 2 + self.echoSpacing / 2 - self.rdGradTime / 2 - \
                hw.grad_rise_time - hw.gradDelay * 1e-6
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
        delay = self.system.rf_dead_time + self.rfReTime / 2 - self.echoSpacing / 2 + sampling_time / 2 - \
                hw.gradDelay * 1e-6
        block_gr_ph_reph = pp.make_trapezoid(
            channel=ph_channel,
            system=self.system,
            amplitude=grad_reference * hw.gammaB,
            flat_time=self.phGradTime,
            rise_time=hw.grad_rise_time,
            delay=delay,
        )

        # Phase gradient rephasing
        delay = self.system.rf_dead_time + self.rfReTime / 2 - self.echoSpacing / 2 + sampling_time / 2 - \
                hw.gradDelay * 1e-6
        block_gr_sl_reph = pp.make_trapezoid(
            channel=sl_channel,
            system=self.system,
            amplitude=grad_reference * hw.gammaB,
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

        # Initialize batch dictionary
        batches = {}

        def initializeBatch(name="pp_1"):
            # Instantiate pypulseq sequence object and save it into the batches dictionarly
            batches[name] = pp.Sequence(self.system)

            # Set slice and phase gradients to 0
            gr_ph_deph = pp.scale_grad(block_gr_ph_deph, scale=0.0)
            gr_sl_deph = pp.scale_grad(block_gr_sl_deph, scale=0.0)
            gr_ph_reph = pp.scale_grad(block_gr_ph_reph, scale=0.0)
            gr_sl_reph = pp.scale_grad(block_gr_sl_reph, scale=0.0)

            batches[name].add_block(delay_first, block_adc_noise)

            # Create dummy pulses
            for dummy in range(self.dummyPulses):
                # Pre-excitation pulse
                if self.preExTime>0:
                    batches[name].add_block(block_rf_pre_excitation,
                                            delay_pre_excitation)

                # Inversion pulse
                if self.inversionTime>0:
                    batches[name].add_block(block_rf_inversion,
                                            delay_inversion)

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

        def createBatches():
            n_rd_points = 0  # Initialize the readout points counter
            n_rd_points_dict = {}  # Dictionary to track readout points for each batch
            seq_idx = 0  # Sequence batch index
            seq_num = "batch_0"  # Initial batch name
            waveforms = {}  # Dictionary to store generated waveforms

            # Slice sweep
            for sl_idx in range(nSL):
                # Phase sweep
                for ph_idx in range(nPH):
                    # Check if a new batch is needed (either first batch or exceeding readout points limit)
                    if seq_idx == 0 or n_rd_points + n_rd_points_per_train > hw.maxRdPoints:
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

                    # Fix the phase and slice amplitude
                    gr_ph_deph = pp.scale_grad(block_gr_ph_deph, phGradients[ph_idx] / grad_reference)
                    gr_sl_deph = pp.scale_grad(block_gr_sl_deph, slGradients[sl_idx] / grad_reference)
                    gr_ph_reph = pp.scale_grad(block_gr_ph_reph, - phGradients[ph_idx] / grad_reference)
                    gr_sl_reph = pp.scale_grad(block_gr_sl_reph, - slGradients[sl_idx] / grad_reference)

                    # # Add excitation pulse and readout de-phasing gradient
                    # batches[seq_num].add_block(rf_ex)
                    # batches[seq_num].add_block(gr_preph, delay_preph)
                    #
                    # # Add the echo train
                    # for k_echo in range(self.etl):
                    #     # Add refocusing pulse
                    #     batches[seq_num].add_block(rf_ref, delay_reph, gp_d, gs_d, gr_readout, adc)
                    #     batches[seq_num].add_block(gs_r, gp_r)
                    #     n_rd_points += nRD
                    #
                    # # Add time delay to next repetition
                    # batches[seq_num].add_block(delay_tr)

            # After final repetition, save and interpret the last batch
            batches[seq_num].write(seq_num + ".seq")
            waveforms[seq_num], param_dict = self.flo_interpreter.interpret(seq_num + ".seq")
            print(f"{seq_num}.seq ready!")
            print(f"{len(batches)} batches created. Sequence ready!")

            # Update the number of acquired ponits in the last batch
            n_rd_points_dict.pop('batch_0')
            n_rd_points_dict[seq_num] = n_rd_points

            return waveforms, n_rd_points_dict

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
                sampling_period = self.expt.get_rx_ts()[0]  # us
                bw = 1 / sampling_period / hw.oversamplingFactor  # MHz
                print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))

            # Convert the PyPulseq waveform to the Red Pitaya compatible format
            self.pypulseq2mriblankseq(waveforms=waveforms[seq_num], shimming=[0.0, 0.0, 0.0])

            # Load the waveforms into Red Pitaya if not in demo mode
            if not self.demo:
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
                            rxd, msgs = self.expt.run()  # Run the experiment and collect data
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
                data_full = sig.decimate(data_over, hw.oversamplingFactor, ftype='fir', zero_phase=True)
                self.mapVals['data_full'] = data_full

            elif plotSeq and standalone:
                # Plot the sequence if requested and return immediately
                self.sequencePlot(standalone=standalone)
        return True

    def sequenceAnalysis(self, mode=None):
        nPoints = self.mapVals['nPoints']
        axesEnable = self.mapVals['axesEnable']
        self.mode = mode

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
            self.sequenceList['RARE'].mapVals['angle'] = 0.0
            self.sequenceList['RARE'].mapVals['dfov'] = [0.0, 0.0, 0.0]
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

    def dummyAnalysis(self):
        # Get position vector
        fov = self.fov[0]
        n = self.nPoints[0]
        res = fov / n
        rd_vec = np.linspace(-fov / 2, fov / 2, n)

        # Get dummy data
        dummy_pulses = self.mapVals['dummyData'] * 1
        dummy_pulses = np.reshape(sig.decimate(np.reshape(dummy_pulses, -1),
                                               hw.oversamplingFactor,
                                               ftype='fir',
                                               zero_phase=True),
                                  (self.etl, -1))
        dummy1 = dummy_pulses[0, 10:-10]
        dummy2 = dummy_pulses[1, 10:-10]

        # Calculate 1d projections from odd and even echoes
        proj1 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dummy1)))
        proj2 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dummy2)))
        proj1 = proj1 / np.max(np.abs(proj1))
        proj2 = proj2 / np.max(np.abs(proj2))
        proj1[np.abs(proj1) < 0.1] = 0
        proj2[np.abs(proj2) < 0.1] = 0

        # Maks the results
        rd_1 = rd_vec[np.abs(proj1) != 0]
        proj1 = proj1[np.abs(proj1) != 0]
        rd_2 = rd_vec[np.abs(proj2) != 0]
        proj2 = proj2[np.abs(proj2) != 0]

        # Get phase
        phase1 = np.unwrap(np.angle(proj1))
        phase2 = np.unwrap(np.angle(proj2))

        # Do linear regression
        res1 = linregress(rd_1, phase1)
        res2 = linregress(rd_2, phase2)

        # Print info
        print('Info from dummy pulses:')
        print('Phase difference at iso-center: %0.1f º' % ((res2.intercept - res1.intercept) * 180 / np.pi))
        print('Phase slope difference %0.3f rads/m' % (res2.slope - res1.slope))
        
        
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
        ind = self.getIndex(self.etl, nPH, self.sweepMode)
        nRep = (nPH//etl)*nSL
        bw = self.mapVals['bw']
        
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
                
        
        
        new_data = np.zeros((nPH * nSL * nScans, nRD + 2*hw.addRdPoint))
        new_data = np.reshape(self.dataFullmat, (nScans, nSL, nPH, nRD+ 2*hw.addRdPoints))
        
        counter=0  
        for scan in range(nScans):
            for slice_idx in range(nSL):
                for phase_idx in range(nPH):
                    
                    line = new_data[scan, slice_idx, phase_idx, :]
                    line2d = np.reshape(line, (1, nRD+2*hw.addRdPoint))
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
                    acq.discard_pre = hw.addRdPoint
                    acq.discard_post = hw.addRdPoint
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

        
    seq = RARE_pp()
    seq.sequenceAtributes()

    # A
    seq.sequenceRun(plotSeq=True, demo=True, standalone=True)
    # seq.sequencePlot(standalone=True)

    # # B
    # seq.sequenceRun(demo=True, plotSeq=False)
    # seq.sequenceAnalysis(mode='Standalone')
