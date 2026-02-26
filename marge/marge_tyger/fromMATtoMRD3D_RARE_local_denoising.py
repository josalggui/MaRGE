import sys
import argparse
import numpy as np
from typing import Generator
import mrd
import scipy.io as sio


def matToMRD(input, output_file, input_field: str = ""):
    output = sys.stdout.buffer if output_file is None else output_file
    mat_data = sio.loadmat(input)

    # ----------------------------
    # Header info
    # ----------------------------
    axesOrientation = mat_data["axesOrientation"][0]
    nPoints = mat_data["nPoints"][0]          # rd, ph, sl
    nPoints_sig = nPoints[[2, 1, 0]]          # sl, ph, rd
    inverse_axesOrientation = np.argsort(axesOrientation)
    nXYZ = nPoints[inverse_axesOrientation]   # x, y, z

    nPoints = [int(x) for x in nPoints]
    nXYZ = [int(x) for x in nXYZ]
    axesOrientation = [int(x) for x in axesOrientation]
    nPoints_sig = [int(x) for x in nPoints_sig]

    try:
        rdGradAmplitude = mat_data["rd_grad_amplitude"]
    except Exception:
        rdGradAmplitude = mat_data["rdGradAmplitude"]

    fov = mat_data["fov"][0] * 1e1
    fov_adq = fov[np.array(axesOrientation)]
    fov = [int(x) for x in fov.astype(int)]
    fov_adq = [int(x) for x in fov_adq.astype(np.float32)]

    dfov = (mat_data["dfov"][0] * 1e-3).astype(np.float32)
    acqTime = mat_data["acqTime"][0] * 1e-3  # s

    # ----------------------------
    # PF + noise std
    # ----------------------------
    if "parFourierFraction" in mat_data:
        parFourierFraction = float(np.squeeze(mat_data["parFourierFraction"]))
    elif "partialFourierFraction" in mat_data:
        parFourierFraction = float(np.squeeze(mat_data["partialFourierFraction"]))
    else:
        raise KeyError("El MAT no contiene 'parFourierFraction' (ni 'partialFourierFraction').")

    if "data_noise" not in mat_data:
        raise KeyError("El MAT no contiene 'data_noise' (necesario para std(data_noise)).")

    noise_std = float(np.std(mat_data["data_noise"]))  # EXACTO: std(data_noise)

    # ----------------------------
    # k-space
    # ----------------------------
    sampledCartesian = mat_data["sampledCartesian"]
    signal = sampledCartesian[:, 3]

    if input_field:
        kSpace = mat_data[input_field]
    else:
        kSpace = np.reshape(signal, nPoints_sig)  # sl, ph, rd

    kSpace = np.reshape(kSpace, (1, kSpace.shape[0], kSpace.shape[1], kSpace.shape[2]))

    # k vectors
    kTrajec = np.real(sampledCartesian[:, 0:3]).astype(np.float32)
    kTrajec = kTrajec[:, inverse_axesOrientation]  # x,y,z

    kx = np.reshape(kTrajec[:, 0], nPoints_sig)
    ky = np.reshape(kTrajec[:, 1], nPoints_sig)
    kz = np.reshape(kTrajec[:, 2], nPoints_sig)

    kx = np.reshape(kx, (1, kx.shape[0], kx.shape[1], kx.shape[2]))
    ky = np.reshape(ky, (1, ky.shape[0], ky.shape[1], ky.shape[2]))
    kz = np.reshape(kz, (1, kz.shape[0], kz.shape[1], kz.shape[2]))

    rdTimes = np.linspace(-acqTime / 2, acqTime / 2, num=nPoints[0]).reshape((1, 1, 1, nPoints[0]))

    # Position vectors
    rd_pos = np.linspace(-fov_adq[0] / 2, fov_adq[0] / 2, nPoints[0], endpoint=False)
    ph_pos = np.linspace(-fov_adq[1] / 2, fov_adq[1] / 2, nPoints[1], endpoint=False)
    sl_pos = np.linspace(-fov_adq[2] / 2, fov_adq[2] / 2, nPoints[2], endpoint=False)

    ph_posFull, sl_posFull, rd_posFull = np.meshgrid(ph_pos, sl_pos, rd_pos)
    N = nPoints[0] * nPoints[1] * nPoints[2]
    rd_posFull = rd_posFull.reshape((N, 1))
    ph_posFull = ph_posFull.reshape((N, 1))
    sl_posFull = sl_posFull.reshape((N, 1))

    xyz_matrix = np.concatenate((rd_posFull, ph_posFull, sl_posFull), axis=1)
    xyz_matrix = xyz_matrix[:, inverse_axesOrientation]

    x_esp = np.reshape(xyz_matrix[:, 0], nPoints_sig)
    y_esp = np.reshape(xyz_matrix[:, 1], nPoints_sig)
    z_esp = np.reshape(xyz_matrix[:, 2], nPoints_sig)

    x_esp = x_esp.reshape((1, x_esp.shape[0], x_esp.shape[1], x_esp.shape[2]))
    y_esp = y_esp.reshape((1, y_esp.shape[0], y_esp.shape[1], y_esp.shape[2]))
    z_esp = z_esp.reshape((1, z_esp.shape[0], z_esp.shape[1], z_esp.shape[2]))

    # ----------------------------
    # MRD Header
    # ----------------------------
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
    h.encoding.append(enc)

    if h.user_parameters is None:
        h.user_parameters = mrd.UserParametersType()

    # doubles
    readout_gradient = mrd.UserParameterDoubleType()
    readout_gradient.name = "readout_gradient_intensity"
    readout_gradient.value = float(np.squeeze(rdGradAmplitude))

    pf_param = mrd.UserParameterDoubleType()
    pf_param.name = "parFourierFraction"
    pf_param.value = float(parFourierFraction)

    noise_param = mrd.UserParameterDoubleType()
    noise_param.name = "noise_std"
    noise_param.value = float(noise_std)

    h.user_parameters.user_parameter_double.append(readout_gradient)
    h.user_parameters.user_parameter_double.append(pf_param)
    h.user_parameters.user_parameter_double.append(noise_param)

    # strings
    axes_param = mrd.UserParameterStringType()
    axes_param.name = "axesOrientation"
    axes_param.value = ",".join(map(str, axesOrientation))

    d_fov = mrd.UserParameterStringType()
    d_fov.name = "dfov"
    d_fov.value = ",".join(map(str, dfov))

    h.user_parameters.user_parameter_string.append(axes_param)
    h.user_parameters.user_parameter_string.append(d_fov)

    # ----------------------------
    # Stream acquisitions
    # ----------------------------
    def generate_data() -> Generator[mrd.StreamItem, None, None]:
        acq = mrd.Acquisition()
        acq.data.resize((1, nPoints[0]))
        acq.trajectory.resize((7, nPoints[0]))
        acq.center_sample = round(nPoints[0] / 2)

        for s in range(nPoints[2]):
            for line in range(nPoints[1]):
                # flags + idx VAN EN head en tu API
                acq.head.flags = mrd.AcquisitionFlags(0)
                if line == 0:
                    acq.head.flags |= mrd.AcquisitionFlags.FIRST_IN_ENCODE_STEP_1
                    acq.head.flags |= mrd.AcquisitionFlags.FIRST_IN_SLICE
                    acq.head.flags |= mrd.AcquisitionFlags.FIRST_IN_REPETITION
                if line == nPoints[1] - 1:
                    acq.head.flags |= mrd.AcquisitionFlags.LAST_IN_ENCODE_STEP_1
                    acq.head.flags |= mrd.AcquisitionFlags.LAST_IN_SLICE
                    acq.head.flags |= mrd.AcquisitionFlags.LAST_IN_REPETITION

                acq.head.idx.kspace_encode_step_1 = line
                acq.head.idx.kspace_encode_step_2 = s
                acq.head.idx.slice = s
                acq.head.idx.repetition = 0

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MAT to MRD")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-f", "--input_field", type=str, default="", help="Optional kSpace field in MAT")
    args = parser.parse_args()

    with open(args.output, "wb") as f:
        matToMRD(args.input, f, input_field=args.input_field)