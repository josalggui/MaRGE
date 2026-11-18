import sys
import argparse
import numpy as np
from typing import Generator
import mrd
import scipy.io as sio
import sys

def matToMRD(input, output_file, input_field = ''):
    # print('From MAT to MRD...')
    
    # OUTPUT - write .mrd
    output = sys.stdout.buffer
    if output_file is not None:
        output = output_file
        
    # INPUT - Read .mat
    mat_data = sio.loadmat(input)
    
    # Head info
    axesOrientation = mat_data['axesOrientation'][0]
    nPoints = mat_data['nPoints'][0]    # rd, ph, sl
    nPoints_sig = nPoints[[2,1,0]] # sl, ph, rd (signal is shorted like this)
    inverse_axesOrientation = np.argsort(axesOrientation)   
    nXYZ = nPoints[inverse_axesOrientation]  # x, y, z
    nPoints = [int(x) for x in nPoints]; nXYZ = [int(x) for x in nXYZ]; axesOrientation = [int(x) for x in axesOrientation]
    nPoints_sig = [int(x) for x in nPoints_sig]
    
    try: # RAREpp and RARE_double_image
        rdGradAmplitude = mat_data['rd_grad_amplitude']
    except: # RAREprotocols
        rdGradAmplitude = mat_data['rdGradAmplitude']
        
    fov = mat_data['fov'][0]*1e1; 
    fov_adq = fov[axesOrientation] # rd, ph, sl
    fov = fov.astype(int); fov = [int(x) for x in fov] # mm; x, y, z
    fov_adq = fov_adq
    fov_adq = fov_adq.astype(np.float32); fov_adq = [int(x) for x in fov_adq] # mm; x, y, z
    dfov = mat_data['dfov'][0]*1e-3; dfov = dfov.astype(np.float32)  # mm; x, y, z
    acqTime = mat_data['acqTime'][0]*1e-3 # s
    
    # Signal vector
    sampledCartesian = mat_data['sampledCartesian']
    signal = sampledCartesian[:,3]         
    if input_field:
        kSpace = mat_data[input_field]
    else:
        kSpace = np.reshape(signal, nPoints_sig) # sl, ph, rd
    kSpace = np.reshape(kSpace, (1,kSpace.shape[0],kSpace.shape[1], kSpace.shape[2])) # Expand to MRD requisites

    # k vectors
    kTrajec = np.real(sampledCartesian[:,0:3]).astype(np.float32)    # rd, ph, sl
    kTrajec = kTrajec[:,inverse_axesOrientation]    # x, y, z

    kx = kTrajec[:,0]; kx = np.reshape(kx, nPoints_sig)   # sl, ph, rd    
    kx = np.reshape(kx, (1,kx.shape[0],kx.shape[1], kx.shape[2]))

    ky = kTrajec[:,1]; ky = np.reshape(ky, nPoints_sig)   # sl, ph, rd  
    ky = np.reshape(ky, (1,ky.shape[0],ky.shape[1], ky.shape[2]))

    kz = kTrajec[:,2]; kz = np.reshape(kz, nPoints_sig)   # sl, ph, rd       
    kz = np.reshape(kz, (1,kz.shape[0],kz.shape[1], kz.shape[2]))

    rdTimes = np.linspace(-acqTime / 2, acqTime / 2, num=nPoints[0])
    rdTimes = np.reshape(rdTimes, (1,1,1, rdTimes.shape[0]))

    # Position vectors
    rd_pos = np.linspace(-fov_adq[0] / 2 , fov_adq[0] / 2 , nPoints[0], endpoint=False)
    ph_pos = np.linspace(-fov_adq[1] / 2 , fov_adq[1] / 2 , nPoints[1], endpoint=False)
    sl_pos = np.linspace(-fov_adq[2] / 2 , fov_adq[2] / 2 , nPoints[2], endpoint=False)
    ph_posFull, sl_posFull, rd_posFull = np.meshgrid(ph_pos, sl_pos, rd_pos)
    rd_posFull = np.reshape(rd_posFull, newshape=(nPoints[0] * nPoints[1] * nPoints[2], 1))
    ph_posFull = np.reshape(ph_posFull, newshape=(nPoints[0] * nPoints[1] * nPoints[2], 1))
    sl_posFull = np.reshape(sl_posFull, newshape=(nPoints[0] * nPoints[1] * nPoints[2], 1))
    xyz_matrix = np.concatenate((rd_posFull, ph_posFull, sl_posFull), axis=1) # rd, ph, sl
    xyz_matrix = xyz_matrix[:,inverse_axesOrientation]   # x, y, z
    
    x_esp = xyz_matrix[:,0]; x_esp = np.reshape(x_esp, nPoints_sig)   # sl, ph, rd    
    x_esp = np.reshape(x_esp, (1,x_esp.shape[0],x_esp.shape[1], x_esp.shape[2]))

    y_esp = xyz_matrix[:,1]; y_esp = np.reshape(y_esp, nPoints_sig)   # sl, ph, rd  
    y_esp = np.reshape(y_esp, (1,y_esp.shape[0],y_esp.shape[1], y_esp.shape[2]))

    z_esp = xyz_matrix[:,2]; z_esp = np.reshape(z_esp, nPoints_sig)   # sl, ph, rd       
    z_esp = np.reshape(z_esp, (1,z_esp.shape[0],z_esp.shape[1], z_esp.shape[2]))
    
    # OUTPUT - write .mrd
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
    
    axes_param = mrd.UserParameterStringType()
    axes_param.name = "axesOrientation"
    axes_param.value = ",".join(map(str, axesOrientation))  
    
    d_fov = mrd.UserParameterStringType()
    d_fov.name = "dfov"
    d_fov.value = ",".join(map(str, dfov))  
    
    if h.user_parameters is None:
        h.user_parameters = mrd.UserParametersType()
    h.user_parameters.user_parameter_double.append(readout_gradient)
    h.user_parameters.user_parameter_string.append(axes_param)
    h.user_parameters.user_parameter_string.append(d_fov)

    def generate_data() -> Generator[mrd.StreamItem, None, None]:
        acq = mrd.Acquisition()

        acq.data.resize((1, nPoints[0]))
        acq.trajectory.resize((7, nPoints[0]))
        acq.center_sample = round(nPoints[0] / 2)

        for s in range(nPoints[2]):
            for line in range(nPoints[1]):

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
                acq.trajectory[0,:] = kx[:, s, line, :]
                acq.trajectory[1,:] = ky[:, s, line, :]
                acq.trajectory[2,:] = kz[:, s, line, :]
                acq.trajectory[3,:] = rdTimes[:, :, :, :]
                acq.trajectory[4,:] = x_esp[:, s, line, :]
                acq.trajectory[5,:] = y_esp[:, s, line, :]
                acq.trajectory[6,:] = z_esp[:, s, line, :]
                
                yield mrd.StreamItem.Acquisition(acq)

    with mrd.BinaryMrdWriter(output) as w:
        w.write_header(h)
        w.write_data(generate_data())

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