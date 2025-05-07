import sys
import argparse
import numpy as np
from typing import Generator
import mrd
import scipy.io as sio
import matplotlib.pyplot as plt

def matToMRD(input, output_file):
    
    # OUTPUT
    output = sys.stdout.buffer
    if output_file is not None:
        output = output_file

    # INPUT - Read .mat
    mat_data = sio.loadmat(input)
    axesOrientation = mat_data['axesOrientation'][0]
    nPoints = mat_data['nPoints'][0]    # rd, ph, sl
    nXYZ = nPoints[axesOrientation]     # x, y, z
    nPoints = [int(x) for x in nPoints]; nXYZ = [int(x) for x in nXYZ]; axesOrientation = [int(x) for x in axesOrientation]
    rdGradAmplitude = mat_data['rd_grad_amplitude']
    
    fov = mat_data['fov'][0]*1e1; fov = fov.astype(int); fov = [int(x) for x in fov] # mm; x, y, z
    dfov = mat_data['dfov'][0]; dfov = dfov.astype(np.float32)  # mm; x, y, z
    acqTime = mat_data['acqTime'][0]*1e-3 #s
    
    sampledCartesian = mat_data['sampledCartesian']
    signal = sampledCartesian[:,3]          # x, y, z
    kSpace = np.reshape(signal, nXYZ)       
    kSpace = np.transpose(kSpace, axesOrientation)  # rd, ph, sl
    kSpace = np.transpose(kSpace, [2,1,0])          # sl, ph, rd
    kSpace = np.reshape(kSpace, (1,kSpace.shape[0],kSpace.shape[1], kSpace.shape[2])) # Expand to MRD requisites

    kTrajec = np.real(sampledCartesian[:,0:3]).astype(np.float32)       # rd, ph, sl
    kTrajec = kTrajec[:,axesOrientation]    # x, y, z

    kx = kTrajec[:,0]; kx = np.reshape(kx, nXYZ)       
    kx = np.transpose(kx, axesOrientation)  # rd, ph, sl
    kx = np.transpose(kx, [2,1,0])          # sl, ph, rd
    kx = np.reshape(kx, (1,kx.shape[0],kx.shape[1], kx.shape[2]))

    ky = kTrajec[:,1]; ky = np.reshape(ky, nXYZ)       
    ky = np.transpose(ky, axesOrientation)  # rd, ph, sl
    ky = np.transpose(ky, [2,1,0])          # sl, ph, rd
    ky = np.reshape(ky, (1,ky.shape[0],ky.shape[1], ky.shape[2]))

    kz = kTrajec[:,2]; kz = np.reshape(kz, nXYZ)       
    kz = np.transpose(kz, axesOrientation)  # rd, ph, sl
    kz = np.transpose(kz, [2,1,0])          # sl, ph, rd
    kz = np.reshape(kz, (1,kz.shape[0],kz.shape[1], kz.shape[2]))

    rdTimes = np.linspace(-acqTime / 2, acqTime / 2, num=nPoints[0])
    rdTimes = np.reshape(rdTimes, (1,1,1, rdTimes.shape[0]))

    # MRD Format
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
    
    readout_gradient = mrd.UserParameterDoubleType()
    readout_gradient.name = "readout_gradient_intensity"
    readout_gradient.value = rdGradAmplitude
    if h.user_parameters is None:
        h.user_parameters = mrd.UserParametersType()
    h.user_parameters.user_parameter_double.append(readout_gradient)


    def generate_data() -> Generator[mrd.StreamItem, None, None]:
        acq = mrd.Acquisition()

        acq.data.resize((1, nPoints[0]))
        acq.trajectory.resize((4, nPoints[0]))
        acq.channel_order = axesOrientation
        acq.center_sample = round(nPoints[0] / 2)
        acq.position = dfov

        for s in range(nPoints[2]):
            for line in range(nPoints[1]):

                acq.flags = mrd.AcquisitionFlags(0)
                if line == 0:
                    acq.flags |= mrd.AcquisitionFlags.FIRST_IN_ENCODE_STEP_1
                    acq.flags |= mrd.AcquisitionFlags.FIRST_IN_SLICE
                    acq.flags |= mrd.AcquisitionFlags.FIRST_IN_REPETITION
                if line == nPoints[1] - 1:
                    acq.flags |= mrd.AcquisitionFlags.LAST_IN_ENCODE_STEP_1
                    acq.flags |= mrd.AcquisitionFlags.LAST_IN_SLICE
                    acq.flags |= mrd.AcquisitionFlags.LAST_IN_REPETITION

                acq.idx.kspace_encode_step_1 = line
                acq.idx.kspace_encode_step_2 = s
                acq.idx.slice = s
                acq.idx.repetition = 0
                acq.data[:] = kSpace[:, s, line, :]
                acq.trajectory[0,:] = kx[:, s, line, :]
                acq.trajectory[1,:] = ky[:, s, line, :]
                acq.trajectory[2,:] = kz[:, s, line, :]
                acq.trajectory[3,:] = rdTimes[:, :, :, :]
                yield mrd.StreamItem.Acquisition(acq)

    with mrd.BinaryMrdWriter(output) as w:
        w.write_header(h)
        w.write_data(generate_data())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mat to MRD")
    parser.add_argument('-i', '--input', type=str, required=False, help="Input file path")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output MRD file")

    # parser.set_defaults(
    #     input = '/home/tyger/tyger_repo_may/brainIR.mat',
    #     output= '/home/tyger/tyger_repo_may/Tyger_MRIlab/input_rdGrad.bin',

    # )
    
    args = parser.parse_args()
    matToMRD(args.input, args.output)