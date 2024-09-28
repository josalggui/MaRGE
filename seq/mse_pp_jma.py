"""
Created on Wen April 10 2024
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@Summary: mse sequence class
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
from scipy.optimize import curve_fit
from flocra_pulseq.interpreter import PSInterpreter
import pypulseq as pp

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class MSE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(MSE, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='MSEInfo', val='MSE_jma')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=10.0, units=units.kHz, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=60.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=120.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, units=units.ms, field='SEQ')
        self.addParameter(key='preExTime', string='Preexitation time (ms)', val=0.0, units=units.ms, field='SEQ')
        self.addParameter(key='inversionTime', string='Inversion time (ms)', val=0.0, units=units.ms, field='SEQ', tip="0 to ommit this pulse")
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=300., units=units.ms, field='SEQ', tip="0 to ommit this pulse")
        self.addParameter(key='fov', string='FOV[x,y,z] (cm)', val=[15.0, 15.0, 15.0], units=units.cm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM', tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[40, 40, 20], field='IM')
        self.addParameter(key='angle', string='Angle (º)', val=0.0, field='IM')
        self.addParameter(key='rotationAxis', string='Rotation axis', val=[0, 0, 1], field='IM')
        self.addParameter(key='etl', string='Echo train length', val=4, field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=2.0, units=units.ms, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[0, 1, 2], field='IM', tip="0=x, 1=y, 2=z")
        self.addParameter(key='axesEnable', string='Axes enable', val=[1, 1, 1], field='IM', tip="Use 0 for directions with matrix size 1, use 1 otherwise.")
        self.addParameter(key='sweepMode', string='Sweep mode', val=1, field='SEQ', tip="0: sweep from -kmax to kmax. 1: sweep from 0 to kmax. 2: sweep from kmax to 0")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=2.5, units=units.ms, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='phGradTime', string='Ph gradient time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')
        self.addParameter(key='rfPhase', string='RF phase (º)', val=0.0, field='OTH')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=1, field='SEQ', tip="Use last dummy pulse to calibrate k = 0")
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='parFourierFraction', string='Partial fourier fraction', val=1, field='OTH', tip="Fraction of k planes aquired in slice direction")
        self.addParameter(key='echo_shift', string='Echo time shift', val=0.0, units=units.us, field='OTH', tip='Shift the gradient echo time respect to the spin echo time.')
        self.addParameter(key='unlock_orientation', string='Unlock image orientation', val=0, field='OTH', tip='0: Images oriented according to standard. 1: Image raw orientation')
        # self.addParameter(key='calculateMap', string='Calculate T2 Map', val=1, field='OTH', tip='0: Do not calculate. 1: Calculate')
        self.addParameter(key='rfMode', string='RF mode', val=3, field='OTH', tip='0: CPMG. 1: APCP. 2:APCPMG. 3:CP')

    def sequenceInfo(self):
        print("3D MSE sequence")
        print("Author: Dr. J.M. Algarín")
        print("Author: Teresa Guallart Naval")
        print("Contact: tguanav@i3m.upv.es")
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
            print("RF amplitude is too high, try with longer RF pulse time.")
            return(0)

        seqTime = nPoints[1]*nPoints[2]*repetitionTime*1e-3*nScans*parFourierFraction/60
        seqTime = np.round(seqTime, decimals=1)
        return(seqTime)  # minutes, scanTime

        # TODO: check for min and max values for all fields

    def sequenceAtributes(self):
        super().sequenceAtributes()

        # Conversion of variables to non-multiplied units
        self.angle = self.angle * np.pi / 180 # rads

        # Add rotation, dfov and fov to the history
        self.rotation = self.rotationAxis.tolist()
        self.rotation.append(self.angle)
        self.rotations.append(self.rotation)
        # self.dfovs.append(self.dfov.tolist())
        self.fovs.append(self.fov.tolist())

    def sequenceRun(self, plotSeq=False, demo=False, standalone=False):
        print('MSE run')
        init_gpa=False # Starts the gpa
        self.demo = demo

        # Define the interpreter. It should be updated on calibration
        self.flo_interpreter = PSInterpreter(tx_warmup=hw.blkTime,  # us
                                             rf_center=hw.larmorFreq * 1e6,  # Hz
                                             rf_amp_max=hw.b1Efficiency / (2 * np.pi) * 1e6,  # Hz
                                             gx_max=hw.gFactor[0] * hw.gammaB,  # Hz/m
                                             gy_max=hw.gFactor[1] * hw.gammaB,  # Hz/m
                                             gz_max=hw.gFactor[2] * hw.gammaB,  # Hz/m
                                             grad_max=np.max(hw.gFactor) * hw.gammaB,  # Hz/m
                                             )

        # Define system properties according to hw_config file
        self.system = pp.Opts(
            rf_dead_time=hw.deadTime * 1e-6,  # s
            max_grad=hw.max_grad,  # mT/m
            grad_unit='mT/m',
            max_slew=hw.max_slew_rate,  # mT/m/ms
            slew_unit='mT/m/ms',
            grad_raster_time=hw.grad_raster_time,  # s
            rise_time=hw.grad_rise_time,  # s
        )

        # Set the fov
        self.dfov = self.getFovDisplacement()
        self.dfov = self.dfov[self.axesOrientation]
        self.fov = self.fov[self.axesOrientation]

        # Miscellaneous
        self.freqOffset = self.freqOffset*1e-6 # MHz
        gradRiseTime = hw.grad_rise_time
        gSteps = hw.grad_steps
        addRdPoints = hw.addRdPoints             # Initial rd points to avoid artifact at the begining of rd
        randFactor = 0e-3                        # Random amplitude to add to the phase gradients
        resolution = self.fov/self.nPoints
        rfExAmp = self.rfExFA/(self.rfExTime*1e6*hw.b1Efficiency)*np.pi/180
        rfReAmp = self.rfReFA/(self.rfReTime*1e6*hw.b1Efficiency)*np.pi/180
        self.mapVals['rfExAmp'] = rfExAmp
        self.mapVals['rfReAmp'] = rfReAmp
        self.mapVals['resolution'] = resolution
        self.mapVals['gradRiseTime'] = gradRiseTime
        self.mapVals['randFactor'] = randFactor
        self.mapVals['addRdPoints'] = addRdPoints
        self.mapVals['larmorFreq'] = hw.larmorFreq + self.freqOffset

        if rfExAmp>1 or rfReAmp>1:
            print("RF amplitude is too high, try with longer RF pulse time.")
            return(0)

        # Matrix size
        nRD = self.nPoints[0]+2*addRdPoints
        nPH = self.nPoints[1]
        nSL = self.nPoints[2]
        n_rd_points_per_train = nRD * self.etl

        # ETL if etl>nPH
        if self.etl>nPH:
            self.etl = nPH

        # parAcqLines in case parAcqLines = 0
        parAcqLines = int(int(self.nPoints[2]*self.parFourierFraction)-self.nPoints[2]/2)
        self.mapVals['partialAcquisition'] = parAcqLines

        # BW
        BW = self.nPoints[0]/self.acqTime*1e-6        # MHz
        BWov = BW*hw.oversamplingFactor     # MHz
        samplingPeriod = 1/BWov             # us

        # Readout gradient time
        if self.rdGradTime<self.acqTime:
            self.rdGradTime = self.acqTime
        self.mapVals['rdGradTime'] = self.rdGradTime * 1e3 # ms

        # Phase and slice de- and re-phasing time
        if self.phGradTime==0 or self.phGradTime>self.echoSpacing/2-self.rfExTime/2-self.rfReTime/2-2*gradRiseTime:
            self.phGradTime = self.echoSpacing/2-self.rfExTime/2-self.rfReTime/2-2*gradRiseTime
        self.mapVals['phGradTime'] = self.phGradTime*1e3 # ms

        # Max gradient amplitude
        rdGradAmplitude = self.nPoints[0]/(hw.gammaB*self.fov[0]*self.acqTime)*self.axesEnable[0]
        phGradAmplitude = nPH/(2*hw.gammaB*self.fov[1]*(self.phGradTime+gradRiseTime))*self.axesEnable[1]
        slGradAmplitude = nSL/(2*hw.gammaB*self.fov[2]*(self.phGradTime+gradRiseTime))*self.axesEnable[2]
        self.mapVals['rdGradAmplitude'] = rdGradAmplitude
        self.mapVals['phGradAmplitude'] = phGradAmplitude
        self.mapVals['slGradAmplitude'] = slGradAmplitude

        # Readout dephasing amplitude
        rdDephAmplitude = 0.5*rdGradAmplitude*(gradRiseTime+self.rdGradTime)/(gradRiseTime+self.rdDephTime)
        self.mapVals['rdDephAmplitude'] = rdDephAmplitude

        # Phase and slice gradient vector
        phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
        slGradients = np.linspace(-slGradAmplitude,slGradAmplitude,num=nSL,endpoint=False)

        # Now fix the number of slices to partailly acquired k-space
        nSL = (int(self.nPoints[2]/2)+parAcqLines)*self.axesEnable[2]+(1-self.axesEnable[2])

        # Add random displacemnt to phase encoding lines
        for ii in range(nPH):
            if ii<np.ceil(nPH/2-nPH/20) or ii>np.ceil(nPH/2+nPH/20):
                phGradients[ii] = phGradients[ii]+randFactor*np.random.randn()
        kPH = hw.gammaB*phGradients*(gradRiseTime+self.phGradTime)
        self.mapVals['phGradients'] = phGradients
        self.mapVals['slGradients'] = slGradients

        # Set phase vector to given sweep mode
        ind = self.getIndex(self.etl, nPH, self.sweepMode)
        self.mapVals['sweepOrder'] = ind
        phGradients = phGradients[ind]
        # Get the rotation matrix
        rot = self.getRotationMatrix()
        gradAmp = np.array([0.0, 0.0, 0.0])
        gradAmp[self.axesOrientation[0]] = 1
        gradAmp = np.reshape(gradAmp, (3, 1))
        result = np.dot(rot, gradAmp)

        # Initialize k-vectors
        k_ph_sl_xyz = np.ones((3, self.nPoints[0]*self.nPoints[1]*nSL))*hw.gammaB*(self.phGradTime+hw.grad_rise_time)
        k_rd_xyz = np.ones((3, self.nPoints[0]*self.nPoints[1]*nSL))*hw.gammaB

        # Map the axis to "x", "y", and "z" according ot axesOrientation
        axes_map = {0: "x", 1: "y", 2: "z"}
        rd_channel = axes_map.get(self.axesOrientation[0], "")
        ph_channel = axes_map.get(self.axesOrientation[1], "")
        sl_channel = axes_map.get(self.axesOrientation[2], "")

        ############################
        # EVENTS ###################
        ############################

        # Excitation pulse
        flip_ex = self.rfExFA * np.pi / 180
        rf_ex = pp.make_block_pulse(
            flip_angle=flip_ex,
            system=self.system,
            duration=self.rfExTime,
            delay=self.system.rf_dead_time,
            phase_offset=np.pi/2,
        )

        # Dephasing gradient
        gr_preph = pp.make_trapezoid(
            channel=rd_channel,
            system=self.system,
            amplitude=rdDephAmplitude*hw.gammaB,
            flat_time=self.rdDephTime,
            delay=self.system.rf_dead_time,
            rise_time=hw.grad_rise_time,
        )
        delay_preph = pp.make_delay(self.echoSpacing / 2 - self.rfExTime / 2 - self.rfReTime / 2 -
                                    self.system.rf_dead_time)

        # Refocusing pulse
        flip_re = self.rfReFA * np.pi / 180
        rf_ref = pp.make_block_pulse(
            flip_angle=flip_re,
            system=self.system,
            duration=self.rfReTime,
            delay=0,
            phase_offset=0,
        )
        delay_reph = pp.make_delay(self.echoSpacing)

        # Phase gradient dephasing
        gr_phase_d = pp.make_trapezoid(
            channel=ph_channel,
            system=self.system,
            amplitude=phGradAmplitude * hw.gammaB,
            flat_time=self.phGradTime,
            delay=self.rfReTime + self.system.rf_dead_time,
            rise_time=hw.grad_rise_time,
        )

        # Phase gradient dephasing
        gr_slice_d = pp.make_trapezoid(
            channel=sl_channel,
            system=self.system,
            amplitude=slGradAmplitude * hw.gammaB,
            flat_time=self.phGradTime,
            delay=self.rfReTime + self.system.rf_dead_time,
            rise_time=hw.grad_rise_time,
        )

        # Readout gradient
        delay = self.rfReTime / 2 + self.echoSpacing /2 - self.rdGradTime / 2 - hw.grad_rise_time + \
                self.system.rf_dead_time
        gr_readout = pp.make_trapezoid(
            channel=rd_channel,
            system=self.system,
            amplitude=rdGradAmplitude * hw.gammaB,
            flat_time=self.rdGradTime,
            delay=delay,
            rise_time=hw.grad_rise_time,
        )

        # ADC
        delay = self.rfReTime / 2 + self.echoSpacing / 2 - (nRD / 2 / BW) * 1e-6 + self.system.rf_dead_time
        adc = pp.make_adc(num_samples=nRD * hw.oversamplingFactor,
                          dwell = samplingPeriod * 1e-6,
                          delay=delay)

        # Phase gradient rephasing
        delay = -self.echoSpacing / 2 + (nRD / 2 / BW) * 1e-6 + self.rfReTime / 2 + self.system.rf_dead_time
        gr_phase_r = pp.make_trapezoid(
            channel=ph_channel,
            system=self.system,
            amplitude=phGradAmplitude * hw.gammaB,
            flat_time=self.phGradTime,
            delay=delay,
            rise_time=hw.grad_rise_time,
        )

        # Phase gradient rephasing
        delay = -self.echoSpacing / 2 + (nRD / 2 / BW) * 1e-6 + self.rfReTime / 2 + self.system.rf_dead_time
        gr_slice_r = pp.make_trapezoid(
            channel=sl_channel,
            system=self.system,
            amplitude=slGradAmplitude * hw.gammaB,
            flat_time=self.phGradTime,
            delay=delay,
            rise_time=hw.grad_rise_time,
        )

        # Delay to complete repetition
        delay_tr = pp.make_delay(self.repetitionTime - self.rfExTime / 2 -
                                 self.echoSpacing / 2 + self.rfReTime / 2 - self.echoSpacing * self.etl)

        batches = {}

        def initializeBatch(name="pp_1"):
            # Instantiate pypulseq sequence object and save it into the batches dictionarly
            batches[name] = pp.Sequence(self.system)

            # Set slice and phase gradients to 0
            gs_d = pp.scale_grad(gr_slice_d, 0.0)
            gp_d = pp.scale_grad(gr_phase_d, 0.0)
            gs_r = pp.scale_grad(gr_slice_r, 0.0)
            gp_r = pp.scale_grad(gr_phase_r, 0.0)

            # Create dummy pulses
            for dummy in range(self.dummyPulses):
                # Add excitation pulse and readout de-phasing gradient
                batches[name].add_block(rf_ex)
                batches[name].add_block(gr_preph, delay_preph)

                # Add echo train
                for k_echo in range(self.etl):
                    batches[name].add_block(rf_ref, delay_reph, gp_d, gs_d, gr_readout)
                    batches[name].add_block(gs_r, gp_r)

                # Add time delay to next repetition
                batches[name].add_block(delay_tr)

        def createBatches():
            n_rd_points = 0
            n_rd_points_dict = {}
            seq_idx = 0
            seq_num = "batch_0"

            # Slice sweep
            for Cz in range(nSL):

                # Get slice gradient amplitude
                Nph_range = range(nPH)

                # Phase sweep
                for Cy in Nph_range:
                    # Initialize new sequence with corresponding dummy pulses
                    if seq_idx == 0 or n_rd_points + n_rd_points_per_train > hw.maxRdPoints:
                        seq_idx += 1
                        n_rd_points_dict[seq_num] = n_rd_points
                        seq_num = "batch_%i" % seq_idx
                        initializeBatch(seq_num)
                        n_rd_points = 0

                    # Fix the phase and slice amplitude
                    sl_scale = (Cz - nSL / 2) / nSL * 2
                    pe_scale = (Cy - nPH / 2) / nPH * 2
                    gs_d = pp.scale_grad(gr_slice_d, sl_scale)
                    gp_d = pp.scale_grad(gr_phase_d, pe_scale)
                    gs_r = pp.scale_grad(gr_slice_r, -sl_scale)
                    gp_r = pp.scale_grad(gr_phase_r, -pe_scale)

                    # Add excitation pulse and readout de-phasing gradient
                    batches[seq_num].add_block(rf_ex)
                    batches[seq_num].add_block(gr_preph, delay_preph)

                    # Add the echo train
                    for k_echo in range(self.etl):
                        # Add refocusing pulse
                        batches[seq_num].add_block(rf_ref, delay_reph, gp_d, gs_d, gr_readout, adc)
                        batches[seq_num].add_block(gs_r, gp_r)
                        n_rd_points += nRD

                    # Add time delay to next repetition
                    batches[seq_num].add_block(delay_tr)

            # Get the rd point list
            n_rd_points_dict.pop('batch_0')
            n_rd_points_dict[seq_num] = n_rd_points

            # Check whether the timing of the sequence is correct
            (ok, error_report) = batches[seq_num].check_timing()
            if ok:
                print("Timing check passed successfully")
            else:
                print("Timing check failed. Error listing follows:")
                [print(e) for e in error_report]

            # Write the sequence files
            waveforms = {}
            for seq_num in batches.keys():
                batches[seq_num].write(seq_num + ".seq")
                waveforms[seq_num], param_dict = self.flo_interpreter.interpret(seq_num + ".seq")

            return waveforms, n_rd_points_dict

        # Create the batches
        waveforms, n_readouts = createBatches()
        self.mapVals['n_readouts'] = list(n_readouts.values())
        self.mapVals['n_batches'] = len(n_readouts.values())
        scan_time = (nPH * nSL + self.mapVals['n_batches'] * self.dummyPulses) * self.repetitionTime * self.nScans
        self.mapVals['Scan time (s)'] = scan_time

        # Execute the batches
        data_over = []  # To save oversampled data
        for seq_num in waveforms.keys():
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
                        data_over = np.concatenate((data_over, rxd['rx0']), axis=0)
                        acq_points = np.size([rxd['rx0']])
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



        # # Changing time parameters to us
        # self.rfExTime = self.rfExTime*1e6
        # self.rfReTime = self.rfReTime*1e6
        # self.echoSpacing = self.echoSpacing*1e6
        # self.repetitionTime = self.repetitionTime*1e6
        # gradRiseTime = gradRiseTime*1e6
        # self.phGradTime = self.phGradTime*1e6
        # self.rdGradTime = self.rdGradTime*1e6
        # self.rdDephTime = self.rdDephTime*1e6
        # self.inversionTime = self.inversionTime*1e6
        # self.preExTime = self.preExTime*1e6
        # self.echo_shift = self.echo_shift*1e6
        # nRepetitions = int(nSL*nPH)
        # scanTime = nRepetitions*self.repetitionTime
        # self.mapVals['scanTime'] = scanTime*nSL*1e-6
        # nETL = self.etl
        #
        # # Create full sequence
        # # Run the experiment
        # dataFull = []
        # dummyData = []
        # overData = []
        # noise = []
        # nBatches = 0
        # repeIndexArray = np.array([0])
        # repeIndexGlobal = repeIndexArray[0]
        # phIndex = 0
        # slIndex = 0
        # lnIndex = 0
        # acqPointsPerBatch = []
        # while repeIndexGlobal<nRepetitions:
        #     nBatches += 1
        #     # Create the experiment if it is not a demo
        #     if not self.demo:
        #         self.expt = ex.Experiment(lo_freq=hw.larmorFreq+self.freqOffset, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        #         samplingPeriod = self.expt.get_rx_ts()[0]
        #         BW = 1/samplingPeriod/hw.oversamplingFactor
        #
        #     # Time vector for main points
        #     self.time_vector = np.linspace(-self.nPoints[0]/BW/2 + 0.5/BW, self.nPoints[0]/BW/2 - 0.5/BW,
        #                                    self.nPoints[0]) * 1e-6 # s
        #
        #     # Run the createSequence method
        #     self.acqTime = self.nPoints[0]/BW        # us
        #     self.mapVals['bw'] = BW
        #     phIndex, slIndex, lnIndex, repeIndexGlobal, aa = createSequence(phIndex=phIndex,
        #                                                                     slIndex=slIndex,
        #                                                                     lnIndex=lnIndex,
        #                                                                     repeIndexGlobal=repeIndexGlobal)
        #
        #     # Save instructions into MaRCoS if not a demo
        #     if not self.demo:
        #         if self.floDict2Exp(rewrite=nBatches==1):
        #             print("Sequence waveforms loaded successfully")
        #             pass
        #         else:
        #             print("ERROR: sequence waveforms out of hardware bounds")
        #             return False
        #
        #     repeIndexArray = np.concatenate((repeIndexArray, np.array([repeIndexGlobal-1])), axis=0)
        #     acqPointsPerBatch.append(aa)
        #
        #     if not plotSeq:
        #         for ii in range(self.nScans):
        #             print("Batch %i, scan %i running..." % (nBatches, ii+1))
        #             if not self.demo:
        #                 acq_points = 0
        #                 while acq_points != (aa * hw.oversamplingFactor):
        #                     rxd, msgs = self.expt.run()
        #                     rxd['rx0'] = rxd['rx0']*hw.adcFactor   # Here I normalize to get the result in mV
        #                     acq_points = np.size(rxd['rx0'])
        #                     print("Acquired points = %i" % acq_points)
        #                     print("Expected points = %i" % (aa * hw.oversamplingFactor))
        #             else:
        #                 rxd = {}
        #                 rxd['rx0'] = np.random.randn(aa*hw.oversamplingFactor) + 1j * np.random.randn(aa*hw.oversamplingFactor)
        #             # Get noise data
        #             noise = np.concatenate((noise, rxd['rx0'][0:nRD*hw.oversamplingFactor]), axis = 0)
        #             rxd['rx0'] = rxd['rx0'][nRD*hw.oversamplingFactor::]
        #             # Get data
        #             if self.dummyPulses>0:
        #                 dummyData = np.concatenate((dummyData, rxd['rx0'][0:nRD*self.etl*hw.oversamplingFactor]), axis = 0)
        #                 overData = np.concatenate((overData, rxd['rx0'][nRD*self.etl*hw.oversamplingFactor::]), axis = 0)
        #             else:
        #                 overData = np.concatenate((overData, rxd['rx0']), axis = 0)
        #     # elif plotSeq and standalone:
        #     #     self.plotSequence()
        #
        #     if not self.demo: self.expt.__del__()
        # del aa
        #
        # if not plotSeq:
        #     acqPointsPerBatch= (np.array(acqPointsPerBatch)-self.etl*nRD*(self.dummyPulses>0)-nRD)*self.nScans
        #     print('Scans ready!')
        #     self.mapVals['noiseData'] = noise
        #     self.mapVals['overData'] = overData
        #
        #     # Fix the echo position using oversampled data
        #     if self.dummyPulses>0:
        #         dummyData = np.reshape(dummyData,  (nBatches*self.nScans, self.etl, nRD*hw.oversamplingFactor))
        #         dummyData = np.average(dummyData, axis=0)
        #         self.mapVals['dummyData'] = dummyData
        #         overData = np.reshape(overData, (-1, self.etl, nRD*hw.oversamplingFactor))
        #         #overData = self.fixEchoPosition(dummyData, overData)
        #         overData = np.reshape(overData, -1)
        #         if self.etl > 1:
        #             self.dummyAnalysis()
        #
        #     # Generate dataFull
        #     dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
        #     if nBatches>1:
        #         dataFullA = dataFull[0:sum(acqPointsPerBatch[0:-1])]
        #         dataFullB = dataFull[sum(acqPointsPerBatch[0:-1])::]
        #
        #     # Reorganize dataFull
        #     dataProv = np.zeros([self.nScans,nSL*nPH*nRD*nETL])
        #     dataProv = dataProv+1j*dataProv
        #     if nBatches>1:
        #         dataFullA = np.reshape(dataFullA, (nBatches-1, self.nScans, -1, nRD*nETL))
        #         dataFullB = np.reshape(dataFullB, (1, self.nScans, -1, nRD*nETL))
        #     else:
        #         dataFull = np.reshape(dataFull, (nBatches, self.nScans, -1, nRD*nETL))
        #     for scan in range(self.nScans):
        #         if nBatches>1:
        #             dataProv[scan, :] = np.concatenate((np.reshape(dataFullA[:,scan,:,:],-1), np.reshape(dataFullB[:,scan,:,:],-1)), axis=0)
        #         else:
        #             dataProv[scan, :] = np.reshape(dataFull[:,scan,:,:],-1)
        #     dataFull = np.reshape(dataProv,-1)
        #
        #     # Get index for krd = 0
        #     # Average data
        #     dataProv = np.reshape(dataFull, (self.nScans, nRD*nETL*nPH*nSL))
        #     dataProv = np.average(dataProv, axis=0)
        #     # Reorganize the data acording to sweep mode
        #     dataProv = np.reshape(dataProv, (nSL, nPH, nETL, nRD))
        #     dataTemp = dataProv*0
        #     for jj in range(nETL):
        #         for ii in range(nPH):
        #             dataTemp[:, ind[ii], jj, :] = dataProv[:,  ii, jj, :]
        #     dataProv = dataTemp
        #     # Check where is krd = 0
        #     dataProv = dataProv[int(self.nPoints[2]/2), int(nPH/2), 0, :]
        #     indkrd0 = np.argmax(np.abs(dataProv))
        #     if indkrd0 < nRD/2-addRdPoints or indkrd0 > nRD/2+addRdPoints:
        #         indkrd0 = int(nRD/2)
        #
        #     # Get individual images
        #     dataFull = np.reshape(dataFull, (self.nScans, nSL, nPH, nETL, nRD))
        #     dataFull = dataFull[:, :, :, :, indkrd0-int(self.nPoints[0]/2):indkrd0+int(self.nPoints[0]/2)]
        #     dataTemp = dataFull*0
        #     for ii in range(nPH):
        #         dataTemp[:, :, ind[ii], :, :] = dataFull[:, :,  ii, :, :]
        #     dataFull = dataTemp
        #     imgFull = dataFull*0
        #     for jj in range(nETL):
        #         for ii in range(self.nScans):
        #             imgFull[ii, :, :, jj, :] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataFull[ii, :, :, jj, :])))
        #     self.mapVals['dataFull'] = dataFull
        #     self.mapVals['imgFull'] = imgFull
        #
        #     # Average data
        #     dataMSE = np.average(dataFull, axis=0)
        #     # self.mapVals['kSpace3D_MSE'] = dataMSE
        #
        #     imgMSE = dataMSE*0
        #     for jj in range(nETL):
        #         imgMSE[:,:,jj,:]=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataMSE[:,:,jj,:])))
        #     # self.mapVals['image3D_MSE'] = imgMSE
        #
        #
        #     # Concatenate with k_xyz
        #     for ii in range(3):
        #         k_prov = np.reshape(k_ph_sl_xyz[ii, :], (nSL, nPH, self.nPoints[0]))
        #         k_temp = k_prov * 0
        #         for jj in range(nPH):
        #             k_temp[:, ind[jj], :] = k_prov[:, jj, :]
        #         k_ph_sl_xyz[ii, :] = np.reshape(k_temp, -1)
        #     k_xyz = k_ph_sl_xyz + k_rd_xyz
        #     data_sampled = np.transpose(dataMSE, (0,1,3,2))
        #     sampled_xyz = np.concatenate((k_xyz.T, np.reshape(data_sampled, (nSL*nPH*self.nPoints[0], nETL))), axis=1)
        #     self.mapVals['sampled_xyz'] = sampled_xyz
        #     # print(sampled_xyz.shape)
        #
        #     # Do zero padding
        #     dataAllAcq= np.zeros((nETL,self.nPoints[0]*self.nPoints[1]*self.nPoints[2]), dtype=complex)
        #     for jj in range(nETL):
        #         dataTemp = np.zeros((self.nPoints[2], self.nPoints[1],self.nPoints[0]))
        #         dataTemp = dataTemp+1j*dataTemp
        #         dataTemp[0:nSL, :, :] = dataMSE[:,:,jj,:]
        #         dataTemp = np.reshape(dataTemp, (1,self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))
        #         dataAllAcq[jj,:] = dataTemp
        #     # print(dataAllAcq.shape)
        #
        #     if self.demo:
        #         data = self.myPhantom()
        #
        #     # Fix the position of the sample according to dfov
        #     kMax = np.array(self.nPoints)/(2*np.array(self.fov))*np.array(self.axesEnable)
        #     kRD = self.time_vector*hw.gammaB*rdGradAmplitude
        #     kSL = np.linspace(-kMax[2],kMax[2],num=self.nPoints[2],endpoint=False)
        #     kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
        #     kRD = np.reshape(kRD, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))
        #     kPH = np.reshape(kPH, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))
        #     kSL = np.reshape(kSL, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))
        #     dPhase = np.exp(-2*np.pi*1j*(self.dfov[0]*kRD+self.dfov[1]*kPH+self.dfov[2]*kSL))
        #     kSpaceAll = np.zeros((self.nPoints[2], self.nPoints[1], nETL, self.nPoints[0]),dtype=complex)
        #     imageAll = np.zeros((self.nPoints[2], self.nPoints[1], nETL, self.nPoints[0]),dtype=complex)
        #     for jj in range(nETL):
        #         dataAllAcq[jj,:] = dataAllAcq[jj,:]*dPhase
        #         dataAux = np.reshape(dataAllAcq[jj,:], (self.nPoints[2], self.nPoints[1], self.nPoints[0]))
        #         kSpaceAll[:,:,jj,:] = dataAux
        #         imageAll[:,:,jj,:] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataAux)))
        #
        #     self.mapVals['kSpace3D_MSE'] = kSpaceAll
        #     self.mapVals['image3D_MSE'] = imageAll
        #     img = np.transpose(imageAll, (0,2,1,3))
        #     img = np.reshape(img, (self.nPoints[2]*nETL, self.nPoints[1], self.nPoints[0]))
        #     self.mapVals['image3D'] = img
        #     data = np.transpose(kSpaceAll, (0,2,1,3))
        #     data = np.reshape(data, (self.nPoints[2]*nETL, self.nPoints[1], self.nPoints[0]))
        #     self.mapVals['kSpace3D'] = data
        #
        #     # Create sampled data
        #     kRD = np.reshape(kRD, (self.nPoints[0]*self.nPoints[1]*self.nPoints[2], 1))
        #     kPH = np.reshape(kPH, (self.nPoints[0]*self.nPoints[1]*self.nPoints[2], 1))
        #     kSL = np.reshape(kSL, (self.nPoints[0]*self.nPoints[1]*self.nPoints[2], 1))
        #     dataAll_sampled = dataAllAcq.T
        #     self.mapVals['kMax'] = kMax
        #     self.mapVals['sampled'] = np.concatenate((kRD, kPH, kSL, dataAll_sampled), axis=1)
        #     self.mapVals['sampledCartesian'] = self.mapVals['sampled']  # To sweep
        #
        #     # if self.calculateMap == 1:
        #     #     print('Obtaining T2 Map...')
        #     #     def func1(x, m, t2):
        #     #         return m*np.exp(-x/t2)
        #     #     t2Map = np.zeros((self.nPoints[2], self.nPoints[1], self.nPoints[0],))
        #     #     t2_vector = np.linspace(self.echoSpacing, self.echoSpacing * self.etl, num=self.etl, endpoint=True)*1e3 # s
        #     #     for kk in range(self.nPoints[2]):
        #     #         for jj in range(self.nPoints[1]):
        #     #             for ii in range(self.nPoints[0]):
        #     #                 # Fitting to functions
        #     #                 fitData, xxx = curve_fit(func1, t2_vector,  np.abs(imageAll[kk,jj,:,ii]),
        #     #                                 p0=[np.abs(imageAll[kk,jj,0,ii]), 10])
        #     #                 t2Map[kk,jj,ii] = fitData[1]
        #     #     print(np.min(t2Map))
        #     #     print(np.max(t2Map))
        #     #     self.mapVals['t2Map'] = t2Map
        #
        # return True

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
            data_full_a = data_full[0:sum(n_rds[0:-1])]
            data_full_b = data_full[sum(n_rds[0:-1]):]
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
            data_ind[echo] = data[:, :, echo, :]

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
        k_space = np.abs(data_ind[:, int(nSL / 2), :, :])
        image = np.abs(image_ind[:, int(nSL / 2), :, :])
        image = image / np.max(image) * 100

        imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        if not self.unlock_orientation:  # Image orientation
            if self.axesOrientation[2] == 2:  # Sagittal
                title = "Sagittal"
                if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 1:  # OK
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    x_label = "(-Y) A | PHASE | P (+Y)"
                    y_label = "(-X) I | READOUT | S (+X)"
                    imageOrientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
                else:
                    image = np.transpose(image, (0, 2, 1))
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    x_label = "(-Y) A | READOUT | P (+Y)"
                    y_label = "(-X) I | PHASE | S (+X)"
                    imageOrientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
            elif self.axesOrientation[2] == 1:  # Coronal
                title = "Coronal"
                if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 2:  # OK
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    image = np.flip(image, axis=0)
                    x_label = "(+Z) R | PHASE | L (-Z)"
                    y_label = "(-X) I | READOUT | S (+X)"
                    imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                else:
                    image = np.transpose(image, (0, 2, 1))
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    image = np.flip(image, axis=0)
                    x_label = "(+Z) R | READOUT | L (-Z)"
                    y_label = "(-X) I | PHASE | S (+X)"
                    imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            elif self.axesOrientation[2] == 0:  # Transversal
                title = "Transversal"
                if self.axesOrientation[0] == 1 and self.axesOrientation[1] == 2:
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    x_label = "(+Z) R | PHASE | L (-Z)"
                    y_label = "(+Y) P | READOUT | A (-Y)"
                    imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                else:  # OK
                    image = np.transpose(image, (0, 2, 1))
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
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
                   'xLabel': axes_str[0],
                   'yLabel': axes_str[1],
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