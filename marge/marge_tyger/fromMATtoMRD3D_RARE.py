"""Conversion of RARE .mat files to MRD binary streams for Tyger."""

import sys
import argparse
import numpy as np
from typing import Generator
import mrd
import scipy.io as sio
import sys
import os
from pathlib import Path

def matToMRD(input, output_file, input_field=None):
    """
    Convert a RARE acquisition .mat file to an MRD binary stream.

    Reads k-space data, trajectory, noise, and geometry metadata from the .mat file,
    builds a complete MRD header (encoding limits, axesOrientation, dfov, readout
    gradient), and writes all acquisitions — noise scans first, then k-space lines —
    to the output MRD file or stream.

    Args:
        input (str): Path to the input .mat file.
        output_file (str | os.PathLike | file-like): Destination MRD file path or
            writable binary stream.
        input_field (str, optional): .mat field name of the k-space array to use.
            If None or empty, the k-space is reconstructed from sampledCartesian.
    """
    # print('From MAT to MRD...')

    if output_file is None:
        raise ValueError("'output_file' needed.")

    must_close_out = False
    if isinstance(output_file, (str, os.PathLike)):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        output = open(output_file, "wb")
        must_close_out = True
    else:
        # file-like con .write() binario
        output = output_file

    # INPUT - Read .mat
    mat_data = sio.loadmat(input)

    # Head info
    axesOrientation = mat_data['axesOrientation'][
        0]  # indicate the order of the dimensions in the data (rd, ph, sl) and how they are oriented in space (x, y, z)
    nPoints = mat_data['nPoints'][0]  # rd, ph, sl
    nPoints = [int(x) for x in nPoints]

    try:  # RAREpp and RARE_double_image
        rdGradAmplitude = mat_data['rd_grad_amplitude']
    except:  # RAREprotocols
        rdGradAmplitude = mat_data['rdGradAmplitude']

    fov = mat_data['fov'][
              0] * 1e1;  # fov is in x, y, z order in the .mat file, but we need to reorder it to match the axes orientation (rd, ph, sl)
    fov_adq = [fov[axesOrientation[k]] for k in range(3)]  # rd, ph, sl

    dfov = mat_data['dfov'][0] * 1e-3;
    dfov = dfov.astype(np.float32)  # mm; x, y, z
    dfov_adq = [dfov[axesOrientation[k]] for k in range(3)]  # rd, ph, sl

    acqTime = mat_data['acqTime'][0] * 1e-3  # s
    bw = mat_data['bw_MHz'][0][0] * 1e6  # Hz
    dwell = 1 / bw * 1e9  # ns

    # parFourierFraction: Partial fourier fraction. Fraction of k planes aquired in slice direction
    parFourierFraction = mat_data['parFourierFraction'][0][0].item()

    # partialAcquisition: While doing partial acquisition, this is the number of extra slices acquired next to half nSlices / 2
    partialAcquisition = mat_data['partialAcquisition'][0][0].item()

    print(
        f"axesOrientation: {axesOrientation}, all data are in the order of rd, ph, sl, but the orientation of these dimensions in space is given by axesOrientation, where 0, 1, 2 correspond to x, y, z respectively")
    print(f"axesOrientation: {axesOrientation}")
    print(f"nPoints: {nPoints}")
    print(f"fov_adq: {fov_adq}, dfov_adq: {dfov_adq}")
    print(f"acqTime: {acqTime}, bw: {bw}, dwell: {dwell}")
    print(f"parFourierFraction: {parFourierFraction}, partialAcquisition: {partialAcquisition}")

    # Signal vector
    # sampledCartesian is a 4-D array with the following columns: kx, ky, kz, signal. The rows are ordered according to the acquisition order (rd, ph, sl)
    sampledCartesian = mat_data['sampledCartesian']
    signal = sampledCartesian[:, 3]
    if input_field is not None and input_field != '':
        kSpace = mat_data[input_field]
        if kSpace.ndim == 3:
            kSpace = kSpace[np.newaxis, ...]  # add channel dim: (1, sl, ph, rd)
    else:
        kSpace = np.reshape(signal, (1, nPoints[2], nPoints[1], nPoints[0]))  # sl, ph, rd

    # k vectors, sampled kspace trajectory in rd, ph, sl order
    kTrajec = np.real(sampledCartesian[:, 0:3]).astype(np.float32)  # rd, ph, sl

    krd = kTrajec[:, 0]
    krd = np.reshape(krd, (1, nPoints[2], nPoints[1], nPoints[0]))  # sl, ph, rd

    kph = kTrajec[:, 1]
    kph = np.reshape(kph, (1, nPoints[2], nPoints[1], nPoints[0]))  # sl, ph, rd

    ksl = kTrajec[:, 2]
    ksl = np.reshape(ksl, (1, nPoints[2], nPoints[1], nPoints[0]))  # sl, ph, rd

    rdTimes = np.linspace(-acqTime / 2, acqTime / 2, num=nPoints[0])
    rdTimes = np.reshape(rdTimes, (1, 1, 1, rdTimes.shape[0]))

    # Position vectors
    rd_pos = np.linspace(-fov_adq[0] / 2, fov_adq[0] / 2, nPoints[0], endpoint=False)
    ph_pos = np.linspace(-fov_adq[1] / 2, fov_adq[1] / 2, nPoints[1], endpoint=False)
    sl_pos = np.linspace(-fov_adq[2] / 2, fov_adq[2] / 2, nPoints[2], endpoint=False)
    ph_posFull, sl_posFull, rd_posFull = np.meshgrid(ph_pos, sl_pos, rd_pos)
    rd_posFull = np.reshape(rd_posFull, (nPoints[0] * nPoints[1] * nPoints[2], 1))
    ph_posFull = np.reshape(ph_posFull, (nPoints[0] * nPoints[1] * nPoints[2], 1))
    sl_posFull = np.reshape(sl_posFull, (nPoints[0] * nPoints[1] * nPoints[2], 1))
    rd_ph_sl_matri = np.concatenate((rd_posFull, ph_posFull, sl_posFull), axis=1)  # rd, ph, sl

    rd_esp = rd_ph_sl_matri[:, 0]  # sl, ph, rd
    rd_esp = np.reshape(rd_esp, (1, nPoints[2], nPoints[1], nPoints[0]))

    ph_esp = rd_ph_sl_matri[:, 1]  # sl, ph, rd
    ph_esp = np.reshape(ph_esp, (1, nPoints[2], nPoints[1], nPoints[0]))

    sl_esp = rd_ph_sl_matri[:, 2]  # sl, ph, rd
    sl_esp = np.reshape(sl_esp, (1, nPoints[2], nPoints[1], nPoints[0]))

    ## Noise acq
    data_noise = mat_data['data_noise']
    nNoise = mat_data['nNoise'][0][0].item()

    print(f"nNoise: {nNoise}")
    print(f"sampledCartesian shape: {sampledCartesian.shape}")
    print(f"kSpace shape: {kSpace.shape}, krd shape: {krd.shape}, kph shape: {kph.shape}, ksl shape: {ksl.shape}")

    # OUTPUT - write .mrd
    # MRD Format
    h = mrd.Header()

    sys_info = mrd.AcquisitionSystemInformationType()
    sys_info.receiver_channels = 1
    sys_info.system_field_strength_t = 0.088
    sys_info.system_vendor = "PhysioMRI"
    sys_info.system_model = "odin"
    sys_info.relative_receiver_noise_bandwidth = 0.72
    sys_info.receiver_channels = 1
    sys_info.coil_label = [mrd.CoilLabelType(coil_number=0, coil_name="coil_1")]
    sys_info.institution_name = "i3m"
    sys_info.station_name = "S1"
    sys_info.device_id = "001"
    sys_info.device_serial_number = "001"

    h.acquisition_system_information = sys_info

    e = mrd.EncodingSpaceType()
    e.matrix_size = mrd.MatrixSizeType(x=nPoints[0], y=nPoints[1], z=nPoints[2])
    e.field_of_view_mm = mrd.FieldOfViewMm(x=fov_adq[0], y=fov_adq[1], z=fov_adq[2])

    r = mrd.EncodingSpaceType()
    r.matrix_size = mrd.MatrixSizeType(x=nPoints[0], y=nPoints[1], z=nPoints[2])
    r.field_of_view_mm = mrd.FieldOfViewMm(x=fov_adq[0], y=fov_adq[1], z=fov_adq[2])

    enc = mrd.EncodingType()
    enc.trajectory = mrd.Trajectory.CARTESIAN
    enc.encoded_space = e
    enc.recon_space = r

    enc.encoding_limits = mrd.EncodingLimitsType()
    enc.encoding_limits.kspace_encoding_step_0 = mrd.LimitType(minimum=0, maximum=nPoints[0] - 1,
                                                               center=(nPoints[0]) // 2)
    enc.encoding_limits.kspace_encoding_step_1 = mrd.LimitType(minimum=0, maximum=nPoints[1] - 1,
                                                               center=(nPoints[1]) // 2)

    if nPoints[2] > 1 and parFourierFraction < 1.0 and partialAcquisition > 0:
        # partial fourier acquisition in slice direction
        acquired_e2 = nPoints[2] // 2 + partialAcquisition
        enc.encoding_limits.kspace_encoding_step_2 = mrd.LimitType(minimum=0, maximum=acquired_e2, center=(nPoints[
            2]) // 2)  # post-zero type partial fourier, we acquire more than half of the k-space lines in slice direction, so the center is still in the middle of the acquired lines
        print(
            f"Partial Fourier acquisition in slice direction is detected: acquired_e2={acquired_e2} out of {nPoints[2]}")
    else:
        enc.encoding_limits.kspace_encoding_step_2 = mrd.LimitType(minimum=0, maximum=nPoints[2] - 1,
                                                                   center=(nPoints[2]) // 2)
        print(f"Partial Fourier acquisition in slice direction is not detected")

    enc.encoding_limits.average = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.slice = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.contrast = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.phase = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.repetition = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.set = mrd.LimitType(minimum=0, maximum=0, center=0)
    enc.encoding_limits.segment = mrd.LimitType(minimum=0, maximum=0, center=0)

    h.encoding.append(enc)

    readout_gradient = mrd.UserParameterDoubleType()
    readout_gradient.name = "readout_gradient_intensity"
    #readout_gradient.value = rdGradAmplitude
    readout_gradient.value = float(np.squeeze(rdGradAmplitude).item())


    axes_param = mrd.UserParameterStringType()
    axes_param.name = "axesOrientation"
    axes_param.value = ",".join(map(str, axesOrientation))

    d_fov = mrd.UserParameterStringType()
    d_fov.name = "dfov"
    d_fov.value = ",".join(map(str, dfov_adq))

    if h.user_parameters is None:
        h.user_parameters = mrd.UserParametersType()
    h.user_parameters.user_parameter_double.append(readout_gradient)
    h.user_parameters.user_parameter_string.append(axes_param)
    h.user_parameters.user_parameter_string.append(d_fov)

    print(f"mrd header: {h}")

    def generate_data() -> Generator[mrd.StreamItem, None, None]:
        """
        Yield MRD StreamItems for all noise scans followed by all k-space acquisitions.

        Noise acquisitions are yielded first, each flagged as IS_NOISE_MEASUREMENT.
        Then all (slice, phase-encode line) combinations are yielded in order,
        with trajectory vectors (krd, kph, ksl, rdTimes, rd_esp, ph_esp, sl_esp)
        attached to each acquisition.

        Yields:
            mrd.StreamItem.Acquisition: One item per noise scan and per k-space line.
        """
        acq = mrd.Acquisition()

        acq.data.resize((1, nPoints[0]))
        acq.trajectory.resize((7, nPoints[0]))
        acq.head.center_sample = round(nPoints[0] / 2)

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

        for s in range(nPoints[2]):
            for line in range(nPoints[1]):

                num = (line + s * nPoints[1])

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
                acq.trajectory[0, :] = krd[:, s, line, :]
                acq.trajectory[1, :] = kph[:, s, line, :]
                acq.trajectory[2, :] = ksl[:, s, line, :]
                acq.trajectory[3, :] = rdTimes[:, :, :, :]
                acq.trajectory[4, :] = rd_esp[:, s, line, :]
                acq.trajectory[5, :] = ph_esp[:, s, line, :]
                acq.trajectory[6, :] = sl_esp[:, s, line, :]

                yield mrd.StreamItem.Acquisition(acq)

    with mrd.BinaryMrdWriter(output) as w:
        w.write_header(h)
        w.write_data(generate_data())

    if 'must_close_out' in locals() and must_close_out:
        output.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mat to MRD")
    parser.add_argument('-i', '--input', type=str, required=False, help="Input file path")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output MRD file")

    # parser.set_defaults(
    #     input = '/home/teresa/marcos_tyger/Brain_Images/brainIR.mat',
    #     output= '/home/teresa/marcos_tyger/Brain_Images/brainIR_23.10.25.bin',
    # )

    args = parser.parse_args()
    matToMRD(args.input, args.output)