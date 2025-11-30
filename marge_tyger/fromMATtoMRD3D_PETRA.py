import sys
import argparse
import numpy as np
from typing import Generator
import mrd
import scipy.io as sio
import sys

def matToMRD(input, output_file):
    print('From MAT to MRD...')
    
    # OUTPUT - write .mrd
    output = sys.stdout.buffer
    if output_file is not None:
        output = output_file
        
    # INPUT - Read .mat
    mat_data = sio.loadmat(input)
    
    # Head info
    nPoints = mat_data['nPoints'][0]    # x,y,z (?)
    nPoints_recon  = np.sort(nPoints)
    nPoints = [int(x) for x in nPoints]
    nPoints_recon = [int(x) for x in nPoints_recon]
    fov = mat_data['fov'][0]*1e1; 
    fov = fov.astype(int); fov = [int(x) for x in fov] # mm; x, y, z (?)
    
    # print('nPoints',  nPoints)
    # print('fov:,', fov)
    
    # Signal vector
    sampledCartesian = mat_data['kSpaceRaw']
    lenT = len(sampledCartesian)
    signal = sampledCartesian[:,3]         
    kSpace = np.reshape(signal, (1,lenT,1,1)) # Expand to MRD requisites

    # k vectors
    kTrajec = np.real(sampledCartesian[:,0:3]).astype(np.float32)    # x,y,z (?)
    
    kx = kTrajec[:,0]
    kx = np.reshape(kx, (1,lenT,1,1))

    ky = kTrajec[:,1]
    ky = np.reshape(ky, (1,lenT,1,1))
    
    kz = kTrajec[:,2]
    kz = np.reshape(kz, (1,lenT,1,1))
    
    # OUTPUT - write .mrd
    # MRD Format
    h = mrd.Header()

    sys_info = mrd.AcquisitionSystemInformationType()
    sys_info.receiver_channels = 1
    h.acquisition_system_information = sys_info

    e = mrd.EncodingSpaceType()
    e.matrix_size = mrd.MatrixSizeType(x=nPoints_recon[0], y=nPoints_recon[1], z=nPoints_recon[2])
    e.field_of_view_mm = mrd.FieldOfViewMm(x=fov[0], y=fov[1], z=fov[2])

    r = mrd.EncodingSpaceType()
    r.matrix_size = mrd.MatrixSizeType(x=nPoints[0], y=nPoints[1], z=nPoints[2])
    r.field_of_view_mm = mrd.FieldOfViewMm(x=fov[0], y=fov[1], z=fov[2])

    enc = mrd.EncodingType()
    enc.trajectory = mrd.Trajectory.RADIAL
    enc.encoded_space = e
    enc.recon_space = r
    h.encoding.append(enc)

    def generate_data() -> Generator[mrd.StreamItem, None, None]:
        acq = mrd.Acquisition()

        acq.data.resize((1, nPoints[0]))
        acq.trajectory.resize((7, nPoints[0]))
        acq.head.center_sample = round(nPoints[0] / 2)

        for s in range(lenT):
            acq.head.idx.kspace_encode_step_1 = 1
            acq.head.idx.kspace_encode_step_2 = s
            acq.head.idx.slice = s
            acq.head.idx.repetition = 0
            acq.data[:] = kSpace[:, s, :, :]
            acq.trajectory[0,:] = kx[:, s, :, :]
            acq.trajectory[1,:] = ky[:, s, :, :]
            acq.trajectory[2,:] = kz[:, s, :, :]
            
            yield mrd.StreamItem.Acquisition(acq)
    
    def generate_data_batches(batch_size: int) -> Generator[mrd.StreamItem, None, None]:
        acq = mrd.Acquisition()

        for batch_start in range(0, lenT, batch_size):
            batch_end = min(batch_start + batch_size, lenT)
            batch_len = batch_end - batch_start

            data_batch = kSpace[:, batch_start:batch_end, :, :]  
            kx_batch = kx[:, batch_start:batch_end, :, :]
            ky_batch = ky[:, batch_start:batch_end, :, :]
            kz_batch = kz[:, batch_start:batch_end, :, :]

            n_points_batch = np.prod(data_batch.shape)
            
            acq.data.resize((1, n_points_batch))
            acq.trajectory.resize((7, n_points_batch))
            acq.head.center_sample = round(n_points_batch / 2)
            
            acq.data[:] = data_batch.reshape(-1)
            acq.trajectory[0,:] = kx_batch.reshape(-1)
            acq.trajectory[1,:] = ky_batch.reshape(-1)
            acq.trajectory[2,:] = kz_batch.reshape(-1)
            
            acq.head.idx.kspace_encode_step_1 = 1
            acq.head.idx.kspace_encode_step_2 = batch_start  
            acq.head.idx.slice = batch_start
            acq.head.idx.repetition = 0

            yield mrd.StreamItem.Acquisition(acq)

    with mrd.BinaryMrdWriter(output) as w:
        w.write_header(h)
        # w.write_data(generate_data())
        w.write_data(generate_data_batches(1000))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mat to MRD")
    parser.add_argument('-i', '--input', type=str, required=False, help="Input file path")
    parser.add_argument('-o', '--output', type=str, required=False, help="Output MRD file")

    # parser.set_defaults(
    #     input = '/home/tyger/Tyger_MRIlab/toTest/Petra_tyger/PETRA.2025.07.23.20.23.09.100.mat',
    #     output= '/home/tyger/Tyger_MRIlab/toTest/Petra_tyger/testPERA.bin',
    # )
    
    args = parser.parse_args()
    matToMRD(args.input, args.output)