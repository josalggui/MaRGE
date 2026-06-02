"""Conversion of double-echo RARE local-denoising .mat files to MRD binary streams."""

import sys
import io
import os
import argparse
import numpy as np
from pathlib import Path
from typing import Generator

import mrd
import scipy.io as sio


def matToMRD(input, output_file, input_field_raw: str = "sampled_odd"):
    """
    Convert a double-echo RARE local-denoising .mat file to an MRD binary stream.

    Reads the selected echo's k-space (sampled_odd or sampled_eve), physical-space
    trajectory (kx, ky, kz), noise acquisitions, partial Fourier fraction, and
    geometry metadata. Noise std and parFourierFraction are embedded in the MRD header
    as user parameters. Noise acquisitions are written first, followed by all k-space lines.

    Args:
        input (str): Path to the input .mat file.
        output_file (str | os.PathLike | file-like | None): Destination MRD file path,
            writable binary stream, or None to write to stdout.
        input_field_raw (str, optional): .mat field name of the k-space array to use.
            Defaults to 'sampled_odd'.
    """
    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    if output_file is None:
        output = sys.stdout.buffer
        must_close_out = False
    elif isinstance(output_file, (str, os.PathLike)):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        output = open(output_file, "wb")
        must_close_out = True
    else:
        # file-like (io.BytesIO, etc.)
        output = output_file
        must_close_out = False

    # ------------------------------------------------------------------
    # .mat
    # ------------------------------------------------------------------
    mat_data = sio.loadmat(input)

    axesOrientation = mat_data["axesOrientation"][0]
    nPoints = mat_data["nPoints"][0]  # rd, ph, sl
    nPoints_sig = nPoints[[2, 1, 0]]  # sl, ph, rd
    inverse_axesOrientation = np.argsort(axesOrientation)
    nXYZ = nPoints[inverse_axesOrientation]  # x, y, z

    nPoints = [int(x) for x in nPoints]
    nXYZ = [int(x) for x in nXYZ]
    axesOrientation = [int(x) for x in axesOrientation]
    nPoints_sig = [int(x) for x in nPoints_sig]

    try:
        rdGradAmplitude = mat_data["rd_grad_amplitude"]
    except KeyError:
        rdGradAmplitude = mat_data["rdGradAmplitude"]

    fov = mat_data["fov"][0] * 1e1                    # cm -> mm
    fov_adq = fov[axesOrientation]                    # rd, ph, sl
    fov = fov.astype(int)
    fov = [int(x) for x in fov]
    fov_adq = fov_adq.astype(np.float32)
    fov_adq = [int(x) for x in fov_adq]

    dfov = (mat_data["dfov"][0] * 1e-3).astype(np.float32)  # mm
    acqTime = mat_data["acqTime"][0] * 1e-3                  # s
    bw = mat_data["bw_MHz"][0][0] * 1e6                      # Hz
    dwell = 1 / bw * 1e9                                     # ns

    # ------------------------------------------------------------------
    # Partial Fourier + noise std
    # ------------------------------------------------------------------
    if "parFourierFraction" in mat_data:
        parFourierFraction = float(np.squeeze(mat_data["parFourierFraction"]))
    elif "partialFourierFraction" in mat_data:
        parFourierFraction = float(np.squeeze(mat_data["partialFourierFraction"]))
    else:
        parFourierFraction = 1.0  # valor por defecto si no existe

    # ------------------------------------------------------------------
    # k-space (selected by input_field_raw)
    # ------------------------------------------------------------------
    sampledCartesian = mat_data[input_field_raw]
    signal = sampledCartesian[:, 3]
    kSpace = np.reshape(signal, nPoints_sig)                           # sl, ph, rd
    kSpace = kSpace.reshape(1, kSpace.shape[0], kSpace.shape[1], kSpace.shape[2])

    # k vectors
    kTrajec = np.real(sampledCartesian[:, 0:3]).astype(np.float32)    # rd, ph, sl
    kTrajec = kTrajec[:, inverse_axesOrientation]                     # x, y, z

    # Reshape function
    def _reshape4(arr):
        """Reshape a 1-D array to (1, sl, ph, rd) using nPoints_sig."""
        arr = np.reshape(arr, nPoints_sig)
        return arr.reshape(1, arr.shape[0], arr.shape[1], arr.shape[2])

    kx = _reshape4(kTrajec[:, 0])
    ky = _reshape4(kTrajec[:, 1])
    kz = _reshape4(kTrajec[:, 2])

    rdTimes = np.linspace(-acqTime / 2, acqTime / 2, num=nPoints[0])
    rdTimes = rdTimes.reshape(1, 1, 1, nPoints[0])

    # Position vectors
    rd_pos = np.linspace(-fov_adq[0] / 2, fov_adq[0] / 2, nPoints[0], endpoint=False)
    ph_pos = np.linspace(-fov_adq[1] / 2, fov_adq[1] / 2, nPoints[1], endpoint=False)
    sl_pos = np.linspace(-fov_adq[2] / 2, fov_adq[2] / 2, nPoints[2], endpoint=False)

    ph_posFull, sl_posFull, rd_posFull = np.meshgrid(ph_pos, sl_pos, rd_pos)
    N = nPoints[0] * nPoints[1] * nPoints[2]
    xyz_matrix = np.concatenate(
        [rd_posFull.reshape(N, 1), ph_posFull.reshape(N, 1), sl_posFull.reshape(N, 1)],
        axis=1,
    )
    xyz_matrix = xyz_matrix[:, inverse_axesOrientation]  # x, y, z

    x_esp = _reshape4(xyz_matrix[:, 0])
    y_esp = _reshape4(xyz_matrix[:, 1])
    z_esp = _reshape4(xyz_matrix[:, 2])

    # ------------------------------------------------------------------
    # Noise acquisitions
    # ------------------------------------------------------------------
    data_noise = mat_data['data_noise']
    nNoise = mat_data['nNoise'][0][0].item()
    noise_std = float(np.std(data_noise)) #¿?

    # ------------------------------------------------------------------
    # MRD Header
    # ------------------------------------------------------------------
    h = mrd.Header()

    sys_info = mrd.AcquisitionSystemInformationType()
    sys_info.receiver_channels = 1
    h.acquisition_system_information = sys_info

    e = mrd.EncodingSpaceType()
    e.matrix_size = mrd.MatrixSizeType(x=nXYZ[0], y=nXYZ[1], z=nXYZ[2])
    e.field_of_view_mm = mrd.FieldOfViewMm(x=fov[0], y=fov[1], z=fov[2])

    r = mrd.EncodingSpaceType()
    r.matrix_size = mrd.MatrixSizeType(x=nXYZ[0], y=nXYZ[1], z=nXYZ[2])
    r.field_of_view_mm = mrd.FieldOfViewMm(x=fov[0], y=fov[1], z=fov[2])

    enc = mrd.EncodingType()
    enc.trajectory = mrd.Trajectory.CARTESIAN
    enc.encoded_space = e
    enc.recon_space = r

    enc.encoding_limits = mrd.EncodingLimitsType()
    enc.encoding_limits.kspace_encoding_step_0 = mrd.LimitType(minimum=0, maximum=nPoints_sig[0] - 1, center=nPoints_sig[0] // 2)
    enc.encoding_limits.kspace_encoding_step_1 = mrd.LimitType(minimum=0, maximum=nPoints_sig[1] - 1, center=nPoints_sig[1] // 2)
    enc.encoding_limits.kspace_encoding_step_2 = mrd.LimitType(minimum=0, maximum=nPoints_sig[2] - 1, center=nPoints_sig[2] // 2)
    enc.encoding_limits.average    = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.slice      = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.contrast   = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.phase      = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.repetition = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.set        = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.segment    = mrd.LimitType(minimum=0, maximum=0, center=0)

    h.encoding.append(enc)

    if h.user_parameters is None:
        h.user_parameters = mrd.UserParametersType()

    # doubles
    readout_gradient = mrd.UserParameterDoubleType()
    readout_gradient.name = "readout_gradient_intensity"
    readout_gradient.value = float(np.squeeze(rdGradAmplitude))

    pf_param = mrd.UserParameterDoubleType()
    pf_param.name = "parFourierFraction"
    pf_param.value = parFourierFraction

    noise_param = mrd.UserParameterDoubleType()
    noise_param.name = "noise_std"
    noise_param.value = noise_std

    h.user_parameters.user_parameter_double.append(readout_gradient)
    h.user_parameters.user_parameter_double.append(pf_param)
    h.user_parameters.user_parameter_double.append(noise_param)

    # strings
    p_ax = mrd.UserParameterStringType()
    p_ax.name = "axesOrientation"
    p_ax.value = ",".join(map(str, axesOrientation))

    p_df = mrd.UserParameterStringType()
    p_df.name = "dfov"
    p_df.value = ",".join(map(str, dfov))

    h.user_parameters.user_parameter_string.append(p_ax)
    h.user_parameters.user_parameter_string.append(p_df)

    # ------------------------------------------------------------------
    # Stream aquisitions generator
    # ------------------------------------------------------------------
    def generate_data() -> Generator[mrd.StreamItem, None, None]:
        """
        Yield MRD StreamItems for all noise scans followed by all k-space acquisitions.

        Noise acquisitions are yielded first, each flagged as IS_NOISE_MEASUREMENT.
        Then all (slice, phase-encode line) combinations are yielded in order,
        with trajectory vectors in physical space (kx, ky, kz, rdTimes, x_esp, y_esp, z_esp).

        Yields:
            mrd.StreamItem.Acquisition: One item per noise scan and per k-space line.
        """
        # --- Noise acquisitions ---
        for n in range(nNoise):
            noise = mrd.Acquisition()
            noise.data.resize((1, nPoints[0]))
            noise.trajectory.resize((0, 0))
            noise.head.center_sample = round(nPoints[0] / 2)
            noise.head.scan_counter = n
            noise.head.sample_time_ns = int(dwell)
            noise.head.acquisition_time_stamp_ns = int(n * 2.5 * 1e3)
            noise.head.physiology_time_stamp_ns = [int(2.5 * n * 1e3), 0, 0]
            noise.head.channel_order = [0]
            noise.head.flags = mrd.AcquisitionFlags(0)
            noise.head.flags |= mrd.AcquisitionFlags.IS_NOISE_MEASUREMENT
            noise.head.idx.kspace_encode_step_1 = n
            noise.head.idx.kspace_encode_step_2 = 0
            noise.head.idx.slice = 0
            noise.head.idx.repetition = 0
            noise.head.idx.average = 0
            noise.head.idx.phase = 0
            noise.head.idx.set = 0
            noise.head.idx.contrast = 0
            noise.head.idx.segment = 0
            noise.data[:] = data_noise[n, :]
            yield mrd.StreamItem.Acquisition(noise)

        # --- Signal acquisitions ---
        acq = mrd.Acquisition()
        acq.data.resize((1, nPoints[0]))
        acq.trajectory.resize((7, nPoints[0]))
        acq.head.center_sample = round(nPoints[0] / 2)

        for s in range(nPoints[2]):
            for line in range(nPoints[1]):
                num = line + s * nPoints[1]

                acq.head.flags = mrd.AcquisitionFlags(0)
                if line == 0:
                    acq.head.flags |= mrd.AcquisitionFlags.FIRST_IN_ENCODE_STEP_1
                    acq.head.flags |= mrd.AcquisitionFlags.FIRST_IN_SLICE
                    acq.head.flags |= mrd.AcquisitionFlags.FIRST_IN_REPETITION
                if line == nPoints[1] - 1:
                    acq.head.flags |= mrd.AcquisitionFlags.LAST_IN_ENCODE_STEP_1
                    acq.head.flags |= mrd.AcquisitionFlags.LAST_IN_SLICE
                    acq.head.flags |= mrd.AcquisitionFlags.LAST_IN_REPETITION

                acq.head.scan_counter = num + nNoise
                acq.head.acquisition_time_stamp_ns = int(num * 2 * 1e9)
                acq.head.physiology_time_stamp_ns = [int(2.5 * num * 1e9), 0, 0]
                acq.head.channel_order = [0]
                acq.head.discard_pre = 0
                acq.head.discard_post = 0
                acq.head.center_sample = round(nPoints[0] / 2)
                acq.head.sample_time_ns = int(dwell)
                acq.head.idx.kspace_encode_step_1 = line
                acq.head.idx.kspace_encode_step_2 = s
                acq.head.idx.slice = 0
                acq.head.idx.repetition = 0
                acq.head.idx.average = 0
                acq.head.idx.phase = 0
                acq.head.idx.set = 0
                acq.head.idx.contrast = 0
                acq.head.idx.segment = 0

                acq.data[:] = kSpace[:, s, line, :]
                acq.trajectory[0, :] = kx[:, s, line, :]
                acq.trajectory[1, :] = ky[:, s, line, :]
                acq.trajectory[2, :] = kz[:, s, line, :]
                acq.trajectory[3, :] = rdTimes[:, :, :, :]
                acq.trajectory[4, :] = x_esp[:, s, line, :]
                acq.trajectory[5, :] = y_esp[:, s, line, :]
                acq.trajectory[6, :] = z_esp[:, s, line, :]

                yield mrd.StreamItem.Acquisition(acq)

    with mrd.BinaryMrdWriter(output) as w:
        w.write_header(h)
        w.write_data(generate_data())

    if must_close_out:
        output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert rare_double_image MAT to MRD (local/BytesIO mode)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input .mat file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output .mrd file")
    parser.add_argument("-f", "--input_field", type=str, default="sampled_odd",
                        help="k-space field in .mat (sampled_odd / sampled_eve)")
    args = parser.parse_args()

    with open(args.output, "wb") as f:
        matToMRD(args.input, f, input_field_raw=args.input_field)