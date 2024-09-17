"""
Created on Tuesday September 177 2024
@author: Prof. Dr. Maxim Zaitsev, Department of Diagnostic and Interventional Radiology, University of Freiburg, Germany
@author: Dr. J.M. Algarín, MRILab, i3M, CSIC, Valencia, Spain
@Summary: mse sequence class coded with pypulseq
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
import warnings

import numpy as np
import scipy.signal as sig
from matplotlib import pyplot as plt

import pypulseq as pp

import experiment as ex
import configs.hw_config as hw
import configs.units as units
import seq.mriBlankSeq as blankSeq

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
        self.addParameter(key='seqName', string='MSEInfo', val='MSE PyPulseq')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, units=units.kHz,
                          field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=60.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=100.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, units=units.ms, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=2000., units=units.ms, field='SEQ')
        self.addParameter(key='fov', string='FOV[x,y,z] (cm)', val=[25.6, 19.2, 12.8], units=units.cm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM',
                          tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[40, 30, 20], field='IM')
        self.addParameter(key='etl', string='Echo train length', val=10, field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, units=units.ms, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[0, 1, 2], field='IM',
                          tip="0=x, 1=y, 2=z")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=5.0, units=units.ms, field='OTH')
        self.addParameter(key='rdDephTime', string='Rd dephasing time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='phGradTime', string='Ph gradient time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='rdPreemphasis', string='Rd preemphasis', val=1.0, field='OTH')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=1, field='SEQ',
                          tip="Use last dummy pulse to calibrate k = 0")
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='unlock_orientation', string='Unlock image orientation', val=0, field='OTH',
                          tip='0: Images oriented according to standard. 1: Image raw orientation')

    def sequenceInfo(self):
        print("3D MSE sequence with PyPulseq")
        print("Author: Prof. Dr. Maxim Zaitsev")
        print("University of Freiburg, Germany")
        print("Author: Dr. José Miguel Algarín")
        print("mriLab @ i3M, CSIC, Spain \n")

    def sequenceTime(self):
        print("Sequence time not calculated...")

    def sequenceAtributes(self):
        super().sequenceAtributes()

        # self.dfovs.append(self.dfov.tolist())
        self.fovs.append(self.fov.tolist())

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        print("Run MSE powered by PyPulseq")
        init_gpa = False
        self.demo = demo

        # Set the fov
        self.dfov = self.getFovDisplacement()
        self.dfov = self.dfov[self.axesOrientation]
        self.fov = self.fov[self.axesOrientation]
        fov_mm = self.fov * 1e3
        nRD, nPH, nSL = self.nPoints  # this is actually nRd, nPh and nSl, axes given by axesOrientation
        n_echo = self.etl
        TE = self.echoSpacing
        TR = self.repetitionTime
        dG = hw.grad_rise_time
        sampling_time = self.acqTime
        if self.rdGradTime >= self.acqTime:
            ro_flattop_add = (self.rdGradTime-self.acqTime)/2
        else:
            print("ERROR: readout gradient time must be longer than acquisition time.")
            return 0
        nRD_pre = hw.addRdPoints
        nRD_post = hw.addRdPoints
        n_rd_points_per_train = n_echo * (nRD + nRD_post + nRD_pre)
        n_rd_points_total = nRD * nPH * nSL * n_echo
        os = hw.oversamplingFactor
        t_ex = self.rfExTime
        t_ref = self.rfReTime
        fsp_r = 1  # Not sure about what this parameter does.
        fsp_s = 0.5  # Not sure about what this parameter does. It is not used in the code.

        # Initialize the experiment
        bw = nRD / sampling_time * hw.oversamplingFactor  # Hz
        sampling_period = 1 / bw  # s
        self.mapVals['samplingPeriod'] = sampling_period
        if not self.demo:
            self.expt = ex.Experiment(lo_freq=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                                      rx_t=sampling_period * 1e6,  # us
                                      init_gpa=init_gpa,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                      auto_leds=True
                                      )
            sampling_period = self.expt.get_rx_ts()[0]  # us
            bw = 1 / sampling_period / hw.oversamplingFactor  # MHz
            sampling_time = nRD / bw * 1e-6  # s
            print("Sampling time fixed to: %0.3f ms" % (sampling_time * 1e-3))
            print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))
        else:
            sampling_period = sampling_period * 1e6  # us
            bw = 1 / sampling_period / hw.oversamplingFactor  # MHz
            sampling_time = nRD / bw * 1e-6  # s
        self.mapVals['bw'] = bw * 1e6  # Hz
        self.mapVals['samplingTime'] = sampling_time

        # Derived and modified parameters
        fov = np.array(fov_mm)*1e-3
        TE = round(TE/self.system.grad_raster_time/2)*self.system.grad_raster_time*2 # TE (=ESP) should be divisible to a double gradient raster, which simplifies calcuations
        ro_flattop_time = sampling_time+2*ro_flattop_add
        rf_add = math.ceil(max(self.system.rf_dead_time,
                               self.system.rf_ringdown_time) / self.system.grad_raster_time) * self.system.grad_raster_time  # round up dead times to the gradient raster time to enable correct TE & ESP calculation
        t_sp = round(
            (0.5 * (TE - ro_flattop_time - t_ref) - rf_add) / self.system.grad_raster_time) * self.system.grad_raster_time
        t_spex = round(
            (0.5 * (TE - t_ex - t_ref) - 2 * rf_add) / self.system.grad_raster_time) * self.system.grad_raster_time
        rf_ex_phase = np.pi / 2
        rf_ref_phase = 0
        rd_channel = ("x" * (self.axesOrientation[0] == 0) +
                      "y" * (self.axesOrientation[0] == 1) +
                      "z" * (self.axesOrientation[0] == 2))
        ph_channel = ("x" * (self.axesOrientation[1] == 0) +
                      "y" * (self.axesOrientation[1] == 1) +
                      "z" * (self.axesOrientation[1] == 2))
        sl_channel = ("x" * (self.axesOrientation[2] == 0) +
                      "y" * (self.axesOrientation[2] == 1) +
                      "z" * (self.axesOrientation[2] == 2))

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
        d_ref=pp.make_delay(t_ref+rf_add*2)

        delta_krd = 1 / fov[self.axesOrientation[0]]
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
            num_samples=(nRD_pre+nRD+nRD_post)*os, dwell=sampling_time/nRD/os, delay=t_sp+dG-nRD_pre*sampling_time/nRD
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
            channel=rd_channel, system=self.system, area=agr_preph, duration=t_spex, rise_time=dG
        )
        # Phase-encoding
        delta_kph = 1 / fov[self.axesOrientation[1]]
        gp_max = pp.make_trapezoid(
                        channel=ph_channel,
                        system=self.system,
                        area=delta_kph*nPH/2,
                        duration=t_sp,
                        rise_time=dG,
                    )
        delta_ksl = 1 / fov[self.axesOrientation[2]]
        gs_max = pp.make_trapezoid(
                        channel=sl_channel,
                        system=self.system,
                        area=delta_ksl*nSL/2,
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

        gr_amp = np.array([0, gr_spr.amplitude, gr_spr.amplitude, gr_acq.amplitude, gr_acq.amplitude, gr_spr.amplitude, gr_spr.amplitude, 0])
        gr = pp.make_extended_trapezoid(channel=rd_channel, times=gc_times, amplitudes=gr_amp)

        gp_amp = np.array([0, gp_max.amplitude, gp_max.amplitude, 0, 0, -gp_max.amplitude, -gp_max.amplitude, 0])
        gp_max = pp.make_extended_trapezoid(channel=ph_channel, times=gc_times, amplitudes=gp_amp)

        gs_amp = np.array([0, gs_max.amplitude, gs_max.amplitude, 0, 0, -gs_max.amplitude, -gs_max.amplitude, 0])
        gs_max = pp.make_extended_trapezoid(channel=sl_channel, times=gc_times, amplitudes=gs_amp)

        # Fill-times
        t_ex = pp.calc_duration(d_ex) + pp.calc_duration(gr_preph)
        t_ref = pp.calc_duration(d_ref) + pp.calc_duration(gr)

        t_train = t_ex + n_echo * t_ref

        TR_fill = TR - t_train
        # Round to gradient raster
        TR_fill = self.system.grad_raster_time * np.round(TR_fill / self.system.grad_raster_time)
        if TR_fill < 0:
            print("ERROR: Repetition time too short.")
            return 0
        else:
            print(f"TR fill: {1000 * TR_fill} ms")
        delay_TR = pp.make_delay(TR_fill)

        sequences = {}
        def initializeSequence(name="pp_1"):
            sequences[name] = pp.Sequence()

            # Set slice and phase gradients to 0
            gs = pp.scale_grad(gs_max, 0.0)
            gp = pp.scale_grad(gp_max, 0.0)

            for dummy in range(self.dummyPulses):
                sequences[name].add_block(rf_ex, d_ex)
                sequences[name].add_block(gr_preph)

                for k_echo in range(n_echo):
                    sequences[name].add_block(rf_ref, d_ref)
                    sequences[name].add_block(gs, gp, gr)

                sequences[name].add_block(delay_TR)

        def createSequences():
            n_rd_points = 0
            n_rd_points_dict = {}
            seq_idx = 0
            seq_num = "sequence_0"

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
                        seq_num = "sequence_%i" % seq_idx
                        initializeSequence(seq_num)
                        n_rd_points = 0

                    # Fix the phase and slice amplitude
                    sl_scale = (Cz - nSL / 2) / nSL * 2
                    pe_scale = (Cy - nPH / 2) / nPH * 2
                    gs = pp.scale_grad(gs_max, sl_scale)
                    gp = pp.scale_grad(gp_max, pe_scale)

                    # Add excitation pulse and readout de-phasing gradient
                    sequences[seq_num].add_block(rf_ex, d_ex)
                    sequences[seq_num].add_block(gr_preph)

                    # Add the echo train
                    for k_echo in range(n_echo):
                        # Add refocusing pulse
                        sequences[seq_num].add_block(rf_ref, d_ref)
                        # Add slice, phase and readout gradients
                        sequences[seq_num].add_block(gs, gp, gr, adc)
                        n_rd_points += nRD + nRD_post + nRD_pre

                    # Add time delay to next repetition
                    sequences[seq_num].add_block(delay_TR)

            # Get the rd point list
            n_rd_points_dict.pop('sequence_0')
            n_rd_points_dict[seq_num] = n_rd_points

            # Check whether the timing of the sequence is correct
            (ok, error_report) = sequences[seq_num].check_timing()
            if ok:
                print("Timing check passed successfully")
            else:
                print("Timing check failed. Error listing follows:")
                [print(e) for e in error_report]

            # Write the sequence files
            waveforms = {}
            for seq_num in sequences.keys():
                sequences[seq_num].write(seq_num+".seq")
                waveforms[seq_num], param_dict = self.flo_interpreter.interpret(seq_num+".seq")

            return waveforms, n_rd_points_dict

        # Create the sequences
        waveforms, n_readouts = createSequences()

        # Execute the sequences
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
                    print("Scan %i running..." % (scan + 1))
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

    def sequenceAnalysis(self, mode=None):
        self.mode = mode

        self.output = []
        self.saveRawData()

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

    seq = MSE()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=True, demo=True, standalone=True)
    # seq.sequencePlot(standalone=True)
    # seq.sequenceAnalysis(mode='Standalone')
