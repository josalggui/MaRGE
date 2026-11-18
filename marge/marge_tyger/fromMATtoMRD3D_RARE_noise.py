import sys
import argparse
import numpy as np
from typing import Generator
import mrd
import scipy.io as sio
import sys
import os
from pathlib import Path

def matToMRD(input, output_file):
    # OUTPUT
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
       
    fov = mat_data['fov'][0]*1e1
    fov_adq = fov[axesOrientation] # rd, ph, sl
    fov = fov.astype(int); fov = [int(x) for x in fov] # mm; x, y, z
    fov_adq = fov_adq
    fov_adq = fov_adq.astype(np.float32); fov_adq = [int(x) for x in fov_adq] # mm; x, y, z
    dfov = mat_data['dfov'][0]*1e-3; dfov = dfov.astype(np.float32)  # mm; x, y, z
    acqTime = mat_data['acqTime'][0]*1e-3 # s
    bw = mat_data['bw_MHz'][0][0]*1e6 # Hz
    dwell = 1/bw* 1e9 # ns
   
    # print('axesOrientation',  axesOrientation)
    # print('nPoints',  nPoints)
    # print('nXYZ',  nXYZ)
    # print('nPoints_sig', nPoints_sig)
    # print('fov:,', fov)
    # print('fov_adq: ',fov_adq)
    # print('dfov: ', dfov)
   
    # Signal vector
    sampledCartesian = mat_data['sampledCartesian']
    signal = sampledCartesian[:,3]        
    kSpace = np.reshape(signal, nPoints_sig) # sl, ph, rd
    kSpace = np.reshape(kSpace, (1,kSpace.shape[0],kSpace.shape[1], kSpace.shape[2])) # Expand to MRD requisites
    # kSpace[0,0:10,:,:] = kSpace[0,0:10,:,:]*0 ## Zero padding simulation
    # kSpace[0,:,:,350:] = kSpace[0,:,:,350:]*0 ## Removing artifacts
    
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
    # rd_pos = np.linspace(-fov_adq[0] / 2 + fov_adq[0] / (2 * nPoints[0]) , fov_adq[0] / 2 + fov_adq[0] / (2 * nPoints[0]), nPoints[0], endpoint=False)
    # ph_pos = np.linspace(-fov_adq[1] / 2 + fov_adq[1] / (2 * nPoints[1]) , fov_adq[1] / 2 + fov_adq[1] / (2 * nPoints[1]), nPoints[1], endpoint=False)
    # sl_pos = np.linspace(-fov_adq[2] / 2 + fov_adq[2] / (2 * nPoints[2]) , fov_adq[2] / 2 + fov_adq[2] / (2 * nPoints[2]), nPoints[2], endpoint=False)
    rd_pos = np.linspace(-fov_adq[0] / 2 , fov_adq[0] / 2 , nPoints[0], endpoint=False)
    ph_pos = np.linspace(-fov_adq[1] / 2 , fov_adq[1] / 2 , nPoints[1], endpoint=False)
    sl_pos = np.linspace(-fov_adq[2] / 2 , fov_adq[2] / 2 , nPoints[2], endpoint=False)
    ph_posFull, sl_posFull, rd_posFull = np.meshgrid(ph_pos, sl_pos, rd_pos)
    rd_posFull = np.reshape(rd_posFull, shape=(nPoints[0] * nPoints[1] * nPoints[2], 1))
    ph_posFull = np.reshape(ph_posFull, shape=(nPoints[0] * nPoints[1] * nPoints[2], 1))
    sl_posFull = np.reshape(sl_posFull, shape=(nPoints[0] * nPoints[1] * nPoints[2], 1))
    xyz_matrix = np.concatenate((rd_posFull, ph_posFull, sl_posFull), axis=1) # rd, ph, sl
    xyz_matrix = xyz_matrix[:,inverse_axesOrientation]   # x, y, z
   
    x_esp = xyz_matrix[:,0]; x_esp = np.reshape(x_esp, nPoints_sig)   # sl, ph, rd    
    x_esp = np.reshape(x_esp, (1,x_esp.shape[0],x_esp.shape[1], x_esp.shape[2]))

    y_esp = xyz_matrix[:,1]; y_esp = np.reshape(y_esp, nPoints_sig)   # sl, ph, rd  
    y_esp = np.reshape(y_esp, (1,y_esp.shape[0],y_esp.shape[1], y_esp.shape[2]))

    z_esp = xyz_matrix[:,2]; z_esp = np.reshape(z_esp, nPoints_sig)   # sl, ph, rd      
    z_esp = np.reshape(z_esp, (1,z_esp.shape[0],z_esp.shape[1], z_esp.shape[2]))
   
    ## Noise acq
    data_noise = mat_data['data_noise']
    nNoise = mat_data['nNoise'][0][0].item()
    # print('Num of noise acq:')
    # print(nNoise)
    
    # OUTPUT - write .mrd
    # MRD Format
    h = mrd.Header()

    sys_info = mrd.AcquisitionSystemInformationType()
    sys_info.receiver_channels = 1
    sys_info.system_field_strength_t = 0.097
    sys_info.system_vendor = "i3m"
    sys_info.system_model = "i3m_model"
    sys_info.relative_receiver_noise_bandwidth = 0.72
    sys_info.receiver_channels = 1
    sys_info.coil_label = [mrd.CoilLabelType(coil_number=0, coil_name="coil_1")]
    sys_info.institution_name = "PhysioMRI"
    sys_info.station_name = "i3m_station"
    sys_info.device_id = "i3m_device"
    sys_info.device_serial_number = "i3m_serial"

    h.acquisition_system_information = sys_info

    e = mrd.EncodingSpaceType()
    e.matrix_size = mrd.MatrixSizeType(x=nPoints_sig[2], y=nPoints_sig[1], z=nPoints_sig[0])
    e.field_of_view_mm = mrd.FieldOfViewMm(x=fov[2], y=fov[1], z=fov[0])

    r = mrd.EncodingSpaceType()
    r.matrix_size = mrd.MatrixSizeType(x=nPoints_sig[2], y=nPoints_sig[1], z=nPoints_sig[0])
    r.field_of_view_mm = mrd.FieldOfViewMm(x=fov[2], y=fov[1], z=fov[0])

    enc = mrd.EncodingType()
    enc.trajectory = mrd.Trajectory.CARTESIAN
    enc.encoded_space = e
    enc.recon_space = r

    enc.encoding_limits = mrd.EncodingLimitsType()
    enc.encoding_limits.kspace_encoding_step_0 = mrd.LimitType(minimum=0, maximum=nPoints_sig[0]-1, center=(nPoints_sig[0])//2)
    enc.encoding_limits.kspace_encoding_step_1 = mrd.LimitType(minimum=0, maximum=nPoints_sig[1]-1, center=(nPoints_sig[1])//2)
    enc.encoding_limits.kspace_encoding_step_2 = mrd.LimitType(minimum=0, maximum=nPoints_sig[2]-1, center=(nPoints_sig[2])//2)
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
                acq.head.physiology_time_stamp_ns = [int(2.5*num * 1e9), 0, 0]

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
    
    if 'must_close_out' in locals() and must_close_out:
        output.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Convert mat to MRD")
#     parser.add_argument('-i', '--input', type=str, required=False, help="Input file path")
#     parser.add_argument('-o', '--output', type=str, required=False, help="Output MRD file")

#     parser.set_defaults(
#         input = '/data/raw_data/i3m/RarePyPulseq.2025.10.24.14.18.10.422.mat',
#         output= '/data/raw_data/i3m/RarePyPulseq.2025.10.24.14.18.10.422.mrd',
#     )
   
#     args = parser.parse_args()
#     matToMRD(args.input, args.output)