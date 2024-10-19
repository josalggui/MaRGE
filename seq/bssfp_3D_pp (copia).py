import os
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


class SequenceBSSFP_3D(PulseqSequence, registry_key=Path(__file__).stem):
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

    @classmethod
    def get_readable_name(self) -> str:
        return "3D bSSFP"

    @classmethod
    def get_description(self) -> str:
        return "Volumetric 3D bSSFP acquisition with Cartesian sampling"

    def setup_ui(self, widget) -> bool:
        seq_path = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(f"{seq_path}/{self.get_name()}/interface.ui", widget)

        widget.TR_SpinBox.valueChanged.connect(self.update_info)
        widget.Baseresolution_SpinBox.valueChanged.connect(self.update_info)
        widget.Slices_SpinBox.valueChanged.connect(self.update_info)
        return True

    def update_info(self):
        duration_sec = int(
            self.main_widget.TR_SpinBox.value()
            * (
                self.main_widget.Baseresolution_SpinBox.value()
                * self.main_widget.Slices_SpinBox.value()
                + self.param_dummy_shots
            )
            / 1000
        )
        duration = str(datetime.timedelta(seconds=duration_sec))

        res_slice = (
            self.main_widget.FOV_SpinBox.value()
            / self.main_widget.Slices_SpinBox.value()
            * 10
        )
        res_inplane = (
            self.main_widget.FOV_SpinBox.value()
            / self.main_widget.Baseresolution_SpinBox.value()
            * 10
        )

        self.show_ui_info_text(
            f"TA: {duration} sec       Voxel Size: {res_inplane:.2f} x {res_inplane:.2f} x {res_slice:.2f} mm"
        )

    def get_parameters(self) -> dict:
        return {
            "TE": self.param_TE,
            "TR": self.param_TR,
            "NSA": self.param_NSA,
            "orientation": self.param_orientation,
            "FOV": self.param_FOV,
            "baseresolution": self.param_baseresolution,
            "slices": self.param_slices,
            "BW": self.param_BW,
            "trajectory": self.param_trajectory,
            "ordering": self.param_ordering,
            "FA": self.param_FA,
        }

    @classmethod
    def get_default_parameters(
        self,
    ) -> dict:
        return {
            "TE": 0,
            "TR": 0,
            "NSA": 1,
            "orientation": "Axial",
            "FOV": 15,
            "baseresolution": 32,
            "slices": 8,
            "BW": 32000,
            "trajectory": "Cartesian",
            "ordering": "linear_up",
            "FA": 90,
        }

    def set_parameters(self, parameters, scan_task) -> bool:
        self.problem_list = []
        try:
            self.param_TE = parameters["TE"]
            self.param_TR = parameters["TR"]
            self.param_NSA = parameters["NSA"]
            self.param_orientation = parameters["orientation"]
            self.param_FOV = parameters["FOV"]
            self.param_baseresolution = parameters["baseresolution"]
            self.param_slices = parameters["slices"]
            self.param_BW = parameters["BW"]
            self.param_trajectory = parameters["trajectory"]
            self.param_ordering = parameters["ordering"]
            self.param_FA = parameters["FA"]
        except:
            self.problem_list.append("Invalid parameters provided")
            return False
        return self.validate_parameters(scan_task)

    def write_parameters_to_ui(self, widget) -> bool:
        widget.TE_SpinBox.setValue(self.param_TE)
        widget.TR_SpinBox.setValue(self.param_TR)
        widget.FA_SpinBox.setValue(self.param_FA)
        widget.NSA_SpinBox.setValue(self.param_NSA)
        widget.Orientation_ComboBox.setCurrentText(self.param_orientation)
        widget.FOV_SpinBox.setValue(self.param_FOV)
        widget.Baseresolution_SpinBox.setValue(self.param_baseresolution)
        widget.Slices_SpinBox.setValue(self.param_slices)
        widget.BW_SpinBox.setValue(self.param_BW)
        widget.Trajectory_ComboBox.setCurrentText(self.param_trajectory)
        widget.Ordering_ComboBox.setCurrentText(self.param_ordering)
        return True

    def read_parameters_from_ui(self, widget, scan_task) -> bool:
        self.problem_list = []
        self.param_TE = widget.TE_SpinBox.value()
        self.param_TR = widget.TR_SpinBox.value()
        self.param_NSA = widget.NSA_SpinBox.value()
        self.param_orientation = widget.Orientation_ComboBox.currentText()
        self.param_FOV = widget.FOV_SpinBox.value()
        self.param_baseresolution = widget.Baseresolution_SpinBox.value()
        self.param_slices = widget.Slices_SpinBox.value()
        self.param_BW = widget.BW_SpinBox.value()
        self.param_trajectory = widget.Trajectory_ComboBox.currentText()
        self.param_ordering = widget.Ordering_ComboBox.currentText()
        self.param_FA = widget.FA_SpinBox.value()
        self.validate_parameters(scan_task)
        return self.is_valid()

    def validate_parameters(self, scan_task) -> bool:
        if self.param_TE > self.param_TR:
            self.problem_list.append("TE cannot be longer than TR")
        return self.is_valid()

    def calculate_sequence(self, scan_task) -> bool:
        log.info("Calculating sequence " + self.get_name())
        ipc_comm.send_status(f"Calculating sequence...")

        scan_task.processing.recon_mode = "basic3d"
        scan_task.processing.dim = 3
        scan_task.processing.dim_size = f"{self.param_slices},{self.param_baseresolution},{2*self.param_baseresolution}"
        scan_task.processing.oversampling_read = 2
        self.seq_file_path = self.get_working_folder() + "/seq/acq0.seq"

        if not self.generate_pulseq():
            log.error("Unable to calculate sequence " + self.get_name())
            return False

        log.info("Done calculating sequence " + self.get_name())
        return True

    def run_sequence(self, scan_task) -> bool:
        log.info("Running sequence " + self.get_name())
        ipc_comm.send_status(f"Preparing scan...")

        expected_duration_sec = int(
            self.param_TR
            * (self.param_baseresolution * self.param_slices + self.param_dummy_shots)
            / 1000
        )

        plot_instructions = True

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

        if plot_instructions:
            file = open(self.get_working_folder() + "/other/seq.plot", "wb")
            fig = plt.gcf()
            pickle.dump(fig, file)
            file.close()

            result = ResultItem()
            result.name = "seq_plot"
            result.description = "Timing diagram of sequence"
            result.type = "plot"
            result.file_path = "other/seq.plot"
            result.autoload_viewer = 4
            scan_task.results.append(result)

        log.info("Done running sequence " + self.get_name())
        return True

    def generate_pulseq(self) -> bool:
        output_file = self.seq_file_path
        pe_order_file = self.get_working_folder() + "/rawdata/pe_order.npy"

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
