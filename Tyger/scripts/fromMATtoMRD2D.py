import sys
import argparse
import numpy as np
from typing import Generator
import mrd
import scipy.io as sio
import matplotlib.pyplot as plt

def matToMRD(input, output_file,nSlice):
    # OUTPUT
    output = sys.stdout.buffer
    if output_file is not None:
        output = output_file

    # INPUT - Read .mat
    mat_data = sio.loadmat(input)
    kSpace3D, fov, nPoints, axesOrientation = mat_data['kSpace3D'], mat_data['fov'], mat_data['nPoints'], mat_data['axesOrientation']
    img = np.fft.ifftshift(np.fft.ifftn((kSpace3D)))
    img2D = img[int(nSlice), :, :]
    kSpace = np.fft.fftn(np.fft.fftshift((img2D)))
    kSpace3D = np.reshape(kSpace, (1,1,kSpace3D.shape[1], kSpace3D.shape[2]))
    fov = fov.T; fov = fov[axesOrientation]; fov = np.reshape(fov, (1, 3)); fov = fov[0]*1e1; fov = fov.astype(int); fov = [int(x) for x in fov] # FOV mm [rd,ph,sl]
    nPoints = nPoints[0]; nPoints = [int(x) for x in nPoints] # nPoints [rd,ph,sl]
    # Only to check
    # print(kSpace3D.shape)
    # nSlice = 10
    # plt.figure()
    # plt.imshow(np.abs(kSpace3D[0,nSlice,:,:]))
    # plt.figure()
    # img = np.fft.ifftshift(np.fft.ifftn((kSpace3D)))
    # plt.imshow(np.abs(img[0,nSlice,:,:]))
    # plt.show()

    h = mrd.Header()

    sys_info = mrd.AcquisitionSystemInformationType()
    sys_info.receiver_channels = 1
    h.acquisition_system_information = sys_info

    e = mrd.EncodingSpaceType()
    e.matrix_size = mrd.MatrixSizeType(x=nPoints[0], y=nPoints[1], z=nPoints[2])
    e.field_of_view_mm = mrd.FieldOfViewMm(x=fov[0], y=fov[1], z=fov[2])

    r = mrd.EncodingSpaceType()
    r.matrix_size = mrd.MatrixSizeType(x=nPoints[0], y=nPoints[1], z=nPoints[2])
    r.field_of_view_mm = mrd.FieldOfViewMm(x=fov[0], y=fov[1], z=fov[2])

    enc = mrd.EncodingType()
    enc.trajectory = mrd.Trajectory.CARTESIAN
    enc.encoded_space = e
    enc.recon_space = r
    h.encoding.append(enc)

    def generate_data() -> Generator[mrd.StreamItem, None, None]:
        # We'll reuse this Acquisition object
        acq = mrd.Acquisition()

        acq.data.resize((1, nPoints[0]))
        acq.channel_order = list(range(1))
        acq.center_sample = round(nPoints[0] / 2)
        acq.read_dir[0] = 1.0
        acq.phase_dir[1] = 1.0
        acq.slice_dir[2] = 1.0

        scan_counter = 0

        for s in range(1):
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
                acq.data[:] = kSpace3D[:, s, line, :]
                yield mrd.StreamItem.Acquisition(acq)

    with mrd.BinaryMrdWriter(output) as w:
        w.write_header(h)
        w.write_data(generate_data())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mat to MRD")
    parser.add_argument('-i', '--input', type=str, required=False, help="Input file path")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output MRD file")
    parser.add_argument('-s', '--slice', type=int, required=False, help="Slice to 2D conversion")
    # parser.set_defaults(
    #     input = '/home/teresa/Documentos/Tyger/pythonMRDi3m/knee44L.mat',
    #     output= 'test.bin',
    #     slice = 10
    # )
    args = parser.parse_args()
    matToMRD(args.input, args.output,args.slice)