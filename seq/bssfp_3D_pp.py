"""
Created on Friday, September 20th 2024
@author: PhD. Tobias Block, NYU Langone Health, New York, USA
@author: PhD. J.M. Algarín, MRILab, i3M, CSIC, Valencia, Spain
@Summary: bSSFP created for mri4all Hackathon and adapted for MaRGE
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

from pathlib import Path
import datetime
import math
import numpy as np
from PyQt5 import uic
import matplotlib.pyplot as plt
import pickle

import pypulseq as pp  # type: ignore
import external.seq.adjustments_acq.config as cfg
from external.seq.adjustments_acq.scripts import run_pulseq
from sequences.common.get_trajectory import choose_pe_order
from sequences import PulseqSequence
from sequences.common import make_tse_3D
from common.constants import *
import common.logger as logger
from common.types import ResultItem
import common.helper as helper

log = logger.get_logger()

from common.ipc import Communicator

ipc_comm = Communicator(Communicator.ACQ)

import math

import numpy as np
import scipy.signal as sig

import pypulseq as pp

import experiment as ex
import configs.hw_config as hw
import configs.units as units
import seq.mriBlankSeq as blankSeq


class BSSFP3D(blankSeq.MRIBLANKSEQ):

    def __init__(self):
        super(BSSFP3D, self).__init__()

        # Add input parameters for the GUI
        self.axesOrientation = None
        self.dfov = None
        self.fov = None
        self.addParameter(key='seqName', string='bSSFP Info', val='bSSFP PyPulseq')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, units=units.kHz,
                          field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=60.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, units=units.ms, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=2000., units=units.ms, field='SEQ')
        self.addParameter(key='fov', string='FOV[x,y,z] (cm)', val=[25.6, 19.2, 12.8], units=units.cm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM',
                          tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[40, 30, 40], field='IM')
        self.addParameter(key='bw', string='Acquisition bandwidth (kHz)', val=32.0, units=units.kHz, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[0, 1, 2], field='IM',
                          tip="0=x, 1=y, 2=z")
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=40, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='unlock_orientation', string='Unlock image orientation', val=0, field='OTH',
                          tip='0: Images oriented according to standard. 1: Image raw orientation')

        # Sequence parameters
        param_TE: int = 50
        param_TR: int = 250
        param_NSA: int = 1
        param_orientation: str = "Axial"
        param_FOV: int = 200
        param_baseresolution: int = 64
        param_slices: int = 8
        param_BW: int = 32000
        param_trajectory: str = "Cartesian"
        param_ordering: str = "linear_up"
        param_dummy_shots: int = 40
        param_FA: int = 90

    def sequenceInfo(self):
        print("3D bSSFP sequence with PyPulseq from mri4all console")
        print("Volumetric 3D bSSFP acquisition with Cartesian sampling")
        print("Author: PhD Tobias Block")
        print("NYU Langone Health, New York, USA")
        print("Adapted by: PhD. José Miguel Algarín")
        print("mriLab @ i3M, CSIC, Spain \n")

    def sequenceTime(self):
        print("Sequence time not calculated...")

        return 0

    def sequenceAtributes(self):
        super().sequenceAtributes()

        # self.dfovs.append(self.dfov.tolist())
        self.fovs.append(self.fov.tolist())

        # Reorganize dfov and fov
        self.dfov = self.getFovDisplacement()
        self.dfov = self.dfov[self.axesOrientation]
        self.fov = self.fov[self.axesOrientation]

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        print("Run bSSFP powered by PyPulseq from mri4all")
        init_gpa = False
        self.demo = demo

        # Do some calculations
        resolution = self.fov / self.nPoints  # m
        self.mapVals['Resolution (m)'] = resolution
        n_rd, n_ph, n_sl = self.nPoints
        sampling_time = n_rd / self.bw  # s
        self.mapVals['Sampling Time (s)'] = sampling_time


    def run_sequence(self, scan_task) -> bool:

        expected_duration_sec = int(
            self.param_TR
            * (self.param_baseresolution * self.param_slices + self.param_dummy_shots)
            / 1000
        )

        rxd, rx_t = run_pulseq(
            seq_file=self.seq_file_path,
            rf_center=cfg.LARMOR_FREQ,
            tx_t=1,
            grad_t=10,
            tx_warmup=100,
            shim_x=0.0,
            shim_y=0.0,
            shim_z=0.0,
            grad_cal=False,
            save_np=True,
            save_mat=False,
            save_msgs=False,
            gui_test=False,
            case_path=self.get_working_folder(),
            raw_filename="raw",
            expected_duration_sec=expected_duration_sec,
            plot_instructions=plot_instructions,
        )

    def generate_pulseq(self) -> bool:
        alpha1 = self.param_FA
        alpha1_duration = 80e-6

        TR = self.param_TR / 1000
        TE = self.param_TE / 1000
        fovx = self.param_FOV / 1000
        fovy = self.param_FOV / 1000
        # DEBUG! TODO: Expose FOV in Z on UI
        fovz = self.param_FOV / 1000 / 2
        Nx = self.param_baseresolution
        Ny = self.param_baseresolution
        Nz = self.param_slices
        dim0 = Ny
        dim1 = Nz  # TODO: remove redundancy and bind it closer to UI - next step
        num_averages = self.param_NSA
        orientation = self.param_orientation
        BW = self.param_BW
        ordering = self.param_ordering

        adc_dwell = 1 / BW
        adc_duration = Nx * adc_dwell  # 6.4e-3

        ch0 = "x"
        ch1 = "y"
        ch2 = "z"
        if orientation == "Axial":
            ch0 = "x"
            ch1 = "y"
            ch2 = "z"
        elif orientation == "Sagittal":
            ch0 = "x"
            ch1 = "z"
            ch2 = "y"
        elif orientation == "Coronal":
            ch0 = "y"
            ch1 = "z"
            ch2 = "x"

        # ======
        # INITIATE SEQUENCE
        # ======

        seq = pp.Sequence()
        n_shots = int(Ny * Nz)

        # ======
        # SET SYSTEM CONFIG TODO --> ?
        # ======
        system = pp.Opts(
            max_grad=100,
            grad_unit="mT/m",
            max_slew=4000,
            slew_unit="T/m/s",
            rf_ringdown_time=20e-6,
            rf_dead_time=100e-6,
            rf_raster_time=1e-6,
            adc_dead_time=20e-6,
        )

        # ======
        # CREATE EVENTS
        # ======

        rf1 = pp.make_block_pulse(
            flip_angle=alpha1 * math.pi / 180,
            duration=alpha1_duration,
            delay=0e-6,
            system=system,
            use="excitation",
        )

        # Define other gradients and ADC events
        delta_kx = 1 / fovx
        delta_ky = 1 / fovy
        delta_kz = 1 / fovz
        gx = pp.make_trapezoid(
            channel=ch0, flat_area=Nx * delta_kx, flat_time=adc_duration, system=system
        )
        adc = pp.make_adc(
            num_samples=2 * Nx, duration=gx.flat_time, delay=gx.rise_time, system=system
        )

        gx_pre = pp.make_trapezoid(
            channel=ch0,
            area=gx.area / 2.0,
            system=system,
        )
        gx_pre.amplitude = -1 * gx_pre.amplitude

        pe_order = choose_pe_order(
            ndims=3,
            npe=[dim0, dim1],
            traj=ordering,
            save_pe_order=True,
            save_path=pe_order_file,
        )
        npe = pe_order.shape[0]
        phase_areas0 = pe_order[:, 0] * delta_ky
        phase_areas1 = pe_order[:, 1] * delta_kz

        # Dummy calculation to estimate required spacing
        gy_pre = pp.make_trapezoid(
            channel=ch1,
            area=1.0 * np.max(phase_areas0),
            system=system,
        )
        gz_pre = pp.make_trapezoid(
            channel=ch2,
            area=-1.0 * np.max(phase_areas1),
            system=system,
        )

        pre_duration = max(pp.calc_duration(gy_pre), pp.calc_duration(gz_pre))
        pre_duration = max(pre_duration, pp.calc_duration(gx_pre))

        # ======
        # CALCULATE DELAYS
        # ======

        if TE == 0:
            tau1 = 10 * seq.grad_raster_time
            TE = (
                tau1
                + 0.5 * pp.calc_duration(rf1)
                + pre_duration
                + 0.5 * pp.calc_duration(gx)
            )
        else:
            tau1 = (
                math.ceil(
                    (
                        TE
                        - 0.5 * pp.calc_duration(rf1)
                        - pre_duration
                        - 0.5 * pp.calc_duration(gx)
                    )
                    / seq.grad_raster_time
                )
            ) * seq.grad_raster_time

        # delay_TR = (
        #     math.ceil(
        #         (
        #             TR
        #             - 0.5 * pp.calc_duration(rf1)
        #             - TE
        #             - 0.5 * pp.calc_duration(gx)
        #             - pre_duration
        #         )
        #         / seq.grad_raster_time
        #     )
        # ) * seq.grad_raster_time

        assert np.all(tau1 >= 0)
        # assert np.all(delay_TR >= 0)

        dummyshots = self.param_dummy_shots

        # ======
        # CONSTRUCT SEQUENCE
        # ======

        adc_phase = []
        rf_phase = 0

        # Loop over phase encodes and define sequence blocks
        for avg in range(num_averages):
            for i in range(n_shots + dummyshots):
                if i < dummyshots:
                    is_dummyshot = True
                else:
                    is_dummyshot = False

                rf1.phase_offset = rf_phase / 180 * math.pi
                seq.add_block(rf1)

                if is_dummyshot:
                    pe_idx = 0
                else:
                    pe_idx = i - dummyshots

                gy_pre = pp.make_trapezoid(
                    channel=ch1,
                    area=-1.0 * phase_areas0[pe_idx],
                    duration=pre_duration,
                    system=system,
                )
                gz_pre = pp.make_trapezoid(
                    channel=ch2,
                    area=-1.0 * phase_areas1[pe_idx],
                    duration=pre_duration,
                    system=system,
                )

                seq.add_block(gx_pre, gy_pre, gz_pre)
                seq.add_block(pp.make_delay(tau1))

                if is_dummyshot:
                    seq.add_block(gx)
                else:
                    seq.add_block(gx, adc)
                    adc_phase.append(rf_phase)

                seq.add_block(pp.make_delay(tau1))
                gy_pre.amplitude = -gy_pre.amplitude
                gz_pre.amplitude = -gz_pre.amplitude
                seq.add_block(gx_pre, gy_pre, gz_pre)
                # seq.add_block(pp.make_delay(delay_TR))

                if rf_phase == 0:
                    rf_phase = 180
                else:
                    rf_phase = 0

        # Check whether the timing of the sequence is correct
        ok, error_report = seq.check_timing()
        if ok:
            log.info("Timing check passed successfully")
        else:
            log.info("Timing check failed. Error listing follows:")
            [print(e) for e in error_report]

        try:
            np.save(
                self.get_working_folder()
                + "/"
                + mri4all_taskdata.RAWDATA
                + "/"
                + mri4all_scanfiles.ADC_PHASE,
                adc_phase,
            )
        except:
            log.error("Could not write file with ADC phase")
            return False

        try:
            seq.write(output_file)
            log.debug("Seq file stored")
        except:
            log.error("Could not write sequence file")
            return False

        return True
