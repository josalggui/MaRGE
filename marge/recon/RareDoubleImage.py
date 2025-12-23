import marge.configs.hw_config as hw
import numpy as np
import scipy as sp
from marge.marge_utils import utils
from marge.marge_utils.utils import fix_echo_position


def RareDoubleImage(raw_data_path=None):
    if raw_data_path is None:
        return None

    # Load rawdata and prepare the output dictionary and dicom metadata
    mat_data = sp.io.loadmat(raw_data_path)
    output_dict = {}
    dicom_meta_data = {}

    # Print inputs
    try:
        keys = mat_data['input_keys']
        strings = mat_data['input_strings']
        string = ""
        print("****Inputs****")
        for ii, key in enumerate(keys):
            string = string + f"{str(strings[ii]).strip()}: {np.squeeze(mat_data[str(key).strip()])}, "
        print(string)
    except:
        pass
    print("****Outputs****")

    # Get data
    full_plot = mat_data['full_plot'].item()
    par_fourier_fraction = mat_data['parFourierFraction'].item()
    axes_orientation = np.squeeze(mat_data['axesOrientation'])
    fov = np.squeeze(mat_data['fov']) * 1e-2
    dfov = np.squeeze(mat_data['dfov']) * 1e-3
    fov = fov[axes_orientation]
    dfov = dfov[axes_orientation]
    etl = mat_data['etl'].item()
    n_scans = mat_data['nScans'].item()
    axes_enable = np.squeeze(mat_data['axes_enable'])
    axes_orientation = np.squeeze(mat_data['axesOrientation'])
    n_points = np.squeeze(mat_data['nPoints'])
    partial_acquisition = mat_data['partialAcquisition'].item()
    n_batches = mat_data['n_batches'].item()
    n_readouts = mat_data['n_readouts'][0]
    n_rd, n_ph, n_sl = n_points
    ind = np.squeeze(mat_data['sweepOrder'])
    add_rd_points = mat_data['add_rd_points'].item()
    n_noise = mat_data['nNoise'].item()
    k_fill = mat_data['k_fill'].item()
    oversampling_factor = mat_data['oversampling_factor'].item()
    decimation_factor = mat_data['decimation_factor'].item()
    dummy_pulses = mat_data['dummyPulses'].item()
    rd_direction = mat_data['rd_direction'].item()
    data_decimated = np.squeeze(mat_data['data_decimated'])

    # Correct values
    n_rd_0 = n_points[0]
    n_readouts = n_readouts * oversampling_factor // decimation_factor
    n_points[0] = int(n_points[0] * oversampling_factor / decimation_factor)
    n_rd = int((n_rd + 2 * add_rd_points) * oversampling_factor / decimation_factor)
    n_sl = (n_sl // 2 + partial_acquisition * axes_enable[2] + (1 - axes_enable[2]))
    fov[0] = fov[0] * oversampling_factor / decimation_factor
    add_rd_points = int(add_rd_points * oversampling_factor / decimation_factor)

    # Get noise data, dummy data and signal data
    data_noise = []
    data_dummy = []
    data_signal = []
    points_per_rd = n_rd
    points_per_train = points_per_rd * etl
    idx_0 = 0
    idx_1 = 0
    for batch in range(n_batches):
        n_rds = n_readouts[batch]
        for scan in range(n_scans):
            idx_1 += n_rds
            data_prov = data_decimated[idx_0:idx_1]
            if batch == 0:
                data_noise = np.concatenate((data_noise, data_prov[0:points_per_rd * n_noise]), axis=0)
                if dummy_pulses > 0:
                    data_dummy = np.concatenate(
                        (data_dummy, data_prov[points_per_rd * n_noise:points_per_rd + points_per_train]), axis=0)
                data_signal = np.concatenate((data_signal, data_prov[points_per_rd * n_noise + points_per_train::]),
                                             axis=0)
            else:
                data_noise = np.concatenate((data_noise, data_prov[0:points_per_rd]), axis=0)
                if dummy_pulses > 0:
                    data_dummy = np.concatenate((data_dummy, data_prov[points_per_rd:points_per_rd + points_per_train]),
                                                axis=0)
                data_signal = np.concatenate((data_signal, data_prov[points_per_rd + points_per_train::]), axis=0)
            idx_0 = idx_1
        n_readouts[batch] += -n_rd - n_rd * etl
    data_noise = np.reshape(data_noise, (-1, n_rd))
    data_noise = data_noise[:, add_rd_points:-add_rd_points]
    output_dict['data_noise'] = data_noise
    output_dict['data_dummy'] = data_dummy
    output_dict['data_signal'] = data_signal

    # Decimate data to get signal in desired bandwidth
    data_full = data_signal

    # Reorganize data_full
    data_prov = np.zeros(shape=[n_scans, n_sl * n_ph * n_rd * 2], dtype=complex)
    if n_batches == 2:
        data_full_a = data_full[0:sum(n_readouts[0:-1]) * n_scans]
        data_full_b = data_full[sum(n_readouts[0:-1]) * n_scans:]
        data_full_a = np.reshape(data_full_a, shape=(n_batches - 1, n_scans, -1, n_rd))
        data_full_b = np.reshape(data_full_b, shape=(1, n_scans, -1, n_rd))
        for scan in range(n_scans):
            data_scan_a = np.reshape(data_full_a[:, scan, :, :], -1)
            data_scan_b = np.reshape(data_full_b[:, scan, :, :], -1)
            data_prov[scan, :] = np.concatenate((data_scan_a, data_scan_b), axis=0)
    elif n_batches > 2:
        data_full_ini = data_full[0:n_readouts[0] * n_scans]
        data_full_a = data_full[
            n_readouts[0] * n_scans:n_readouts[0] * n_scans + n_readouts[1] * (n_batches - 2) * n_scans]
        data_full_b = data_full[n_readouts[0] * n_scans + n_readouts[1] * (n_batches - 2) * n_scans:]
        data_full_ini = np.reshape(data_full_ini, shape=(1, n_scans, -1, n_rd))
        data_full_a = np.reshape(data_full_a, shape=(n_batches - 2, n_scans, -1, n_rd))
        data_full_b = np.reshape(data_full_b, shape=(1, n_scans, -1, n_rd))
        for scan in range(n_scans):
            data_scan_ini = np.reshape(data_full_ini[:, scan, :, :], -1)
            data_scan_a = np.reshape(data_full_a[:, scan, :, :], -1)
            data_scan_b = np.reshape(data_full_b[:, scan, :, :], -1)
            data_prov[scan, :] = np.concatenate((data_scan_ini, data_scan_a, data_scan_b), axis=0)
    else:
        data_full = np.reshape(data_full, shape=(1, n_scans, -1, n_rd))
        for scan in range(n_scans):
            data_prov[scan, :] = np.reshape(data_full[:, scan, :, :], -1)
    data_full = np.reshape(data_prov, -1)

    # Get index for krd = 0
    # Average data
    data_prov = np.reshape(data_full, shape=(n_scans, n_rd * n_ph * n_sl * 2))
    data_prov = np.average(data_prov, axis=0)
    # Reorganize the data according to sweep mode
    data_prov = np.reshape(data_prov, shape=(n_sl, n_ph, 2 * n_rd))
    data_temp = np.zeros_like(data_prov)
    for ii in range(n_ph):
        data_temp[:, ind[ii], :] = data_prov[:, ii, :]
    data_prov = data_temp
    # Get central line
    data_prov = data_prov[int(n_points[2] / 2), int(n_ph / 2), 0:n_rd]
    ind_krd_0 = np.argmax(np.abs(data_prov))
    if ind_krd_0 < n_rd / 2 - add_rd_points or ind_krd_0 > n_rd / 2 + add_rd_points:
        ind_krd_0 = int(n_rd / 2)

    # Get individual images
    data_full = np.reshape(data_full, shape=(n_scans, n_sl, n_ph, 2 * n_rd))
    data_full_odd = data_full[:, :, :, ind_krd_0 - int(n_points[0] / 2):ind_krd_0 + int(n_points[0] / 2)]
    data_full_eve = data_full[:, :, :,
                    n_rd + ind_krd_0 - int(n_points[0] / 2):n_rd + ind_krd_0 + int(n_points[0] / 2)]
    data_temp_odd = np.zeros_like(data_full_odd)
    data_temp_eve = np.zeros_like(data_full_eve)
    for ii in range(n_ph):
        data_temp_odd[:, :, ind[ii], :] = data_full_odd[:, :, ii, :]
        data_temp_eve[:, :, ind[ii], :] = data_full_eve[:, :, ii, :]

    data_full_odd = data_temp_odd
    data_full_eve = data_temp_eve
    output_dict['data_full_odd_echoes'] = data_full_odd
    output_dict['data_full_even_echoes'] = data_full_eve

    # Average data
    data_odd = np.average(data_full_odd, axis=0)
    data_eve = np.average(data_full_eve, axis=0)

    # Do zero padding
    data_temp_odd = np.zeros(shape=(n_points[2], n_points[1], n_points[0]), dtype=complex)
    data_temp_eve = np.zeros(shape=(n_points[2], n_points[1], n_points[0]), dtype=complex)
    data_temp_odd[0:n_sl, :, :] = data_odd
    data_temp_eve[0:n_sl, :, :] = data_eve
    if k_fill == 'POCS' and n_points[2] > n_sl:
        data_temp_odd = utils.run_pocs_reconstruction(n_points=n_points[::-1], factors=[par_fourier_fraction, 1, 1], k_space_ref=data_temp_odd)
        data_temp_eve = utils.run_pocs_reconstruction(n_points=n_points[::-1], factors=[par_fourier_fraction, 1, 1], k_space_ref=data_temp_eve)

    # METHOD 1: Get decimated k-space by FFT
    subdecimation_method = 'FFT'
    if subdecimation_method=='FFT' and (full_plot==False or full_plot=='False'):
        image_temp_odd = utils.run_ifft(data_temp_odd)
        image_temp_eve = utils.run_ifft(data_temp_eve)
        idx_0 = n_points[0] // 2 - n_points[0] // 2 * decimation_factor // oversampling_factor
        idx_1 = n_points[0] // 2 + n_points[0] // 2 * decimation_factor // oversampling_factor
        image_temp_odd = image_temp_odd[:, :, idx_0:idx_1]
        image_temp_eve = image_temp_eve[:, :, idx_0:idx_1]
        data_temp_odd = utils.run_dfft(image_temp_odd)
        data_temp_eve = utils.run_dfft(image_temp_eve)
        n_points[0] = n_rd_0
        fov[0] = fov[0] * decimation_factor / oversampling_factor

    # METHO 2: Get image by decimating k-space with average of consecutive samples.
    if subdecimation_method=='AVG' and (full_plot==False or full_plot=='False'):
        n_points[0] = n_rd_0
        fov[0] = fov[0] * decimation_factor / oversampling_factor
        data_prov_odd = np.zeros((n_points[2], n_points[1], n_points[0]), dtype=complex)
        data_prov_eve = np.zeros((n_points[2], n_points[1], n_points[0]), dtype=complex)
        dii = oversampling_factor // decimation_factor
        for ii in range(n_rd_0):
            data_prov_odd[:, :, ii] = np.mean(data_temp_odd[:, :, dii*ii:dii*(ii+1)], axis=2)
            data_prov_eve[:, :, ii] = np.mean(data_temp_eve[:, :, dii*ii:dii*(ii+1)], axis=2)
        data_temp_odd = data_prov_odd
        data_temp_eve = data_prov_eve
        image_temp_odd = utils.run_ifft(data_temp_odd)
        image_temp_eve = utils.run_ifft(data_temp_eve)

    # Fix the position of the sample according to dfov
    data_odd = np.reshape(data_temp_odd, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    data_eve = np.reshape(data_temp_eve, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    bw = mat_data['bw_MHz'].item()
    time_vector = np.linspace(-n_points[0] / bw / 2 + 0.5 / bw, n_points[0] / bw / 2 - 0.5 / bw,
                              n_points[0]) * 1e-6 * decimation_factor / oversampling_factor # s
    kMax = np.array(n_points) / (2 * np.array(fov)) * np.array(axes_enable)
    kRD = time_vector * hw.gammaB * mat_data['rd_grad_amplitude'].item()
    kPH = np.linspace(-kMax[1], kMax[1], num=n_points[1], endpoint=False)
    kSL = np.linspace(-kMax[2], kMax[2], num=n_points[2], endpoint=False)
    kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
    kRD = np.reshape(kRD, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    kPH = np.reshape(kPH, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    kSL = np.reshape(kSL, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    dPhase = np.exp(2 * np.pi * 1j * (dfov[0] * kRD + dfov[1] * kPH + dfov[2] * kSL))
    data_odd = np.reshape(data_odd * dPhase, shape=(n_points[2], n_points[1], n_points[0]))
    data_eve = np.reshape(data_eve * dPhase, shape=(n_points[2], n_points[1], n_points[0]))

    output_dict['kSpace3D_odd_echoes'] = data_odd
    output_dict['kSpace3D_even_echoes'] = data_eve
    img_odd = utils.run_ifft(data_odd)
    img_eve = utils.run_ifft(data_eve)
    img = (np.abs(img_odd) + np.abs(img_eve)) / 2
    data = utils.run_dfft(img)
    output_dict['image3D_odd_echoes'] = img_odd
    output_dict['image3D_even_echoes'] = img_eve
    output_dict['image3D'] = img
    output_dict['kSpace3D'] = data
    data_odd = np.reshape(data_odd, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    data_eve = np.reshape(data_eve, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    data = np.reshape(data, shape=(1, n_points[0] * n_points[1] * n_points[2]))

    # Create sampled data
    kRD = np.reshape(kRD, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    kPH = np.reshape(kPH, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    kSL = np.reshape(kSL, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    data_odd = np.reshape(data_odd, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    data_eve = np.reshape(data_eve, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    data = np.reshape(data, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    output_dict['kMax_1/m'] = kMax
    output_dict['sampled_odd'] = np.concatenate((kRD, kPH, kSL, data_odd), axis=1)
    output_dict['sampled_eve'] = np.concatenate((kRD, kPH, kSL, data_eve), axis=1)
    output_dict['sampled'] = np.concatenate((kRD, kPH, kSL, data), axis=1)
    output_dict['sampledCartesian'] = output_dict['sampled']

    # Get axes in strings
    axesDict = {'x': 0, 'y': 1, 'z': 2}
    axesKeys = list(axesDict.keys())
    axesVals = list(axesDict.values())
    axesStr = ['', '', '']
    n = 0
    for val in axes_orientation:
        index = axesVals.index(val)
        axesStr[n] = axesKeys[index]
        n += 1

    # Plot image
    img_mag_odd = np.abs(img_odd)
    img_mag_eve = np.abs(img_eve)
    img_pha_odd = np.angle(img_odd)
    mask_odd = img_mag_odd > 0.3 * np.max(img_mag_odd)
    mean_phase_odd = np.mean(img_pha_odd[mask_odd])
    img_pha_odd[~mask_odd] = mean_phase_odd
    img_pha_eve = np.angle(img_eve)
    mask_eve = img_mag_eve > 0.3 * np.max(img_mag_eve)
    mean_phase_eve = np.mean(img_pha_eve[mask_eve])
    img_pha_eve[~mask_eve] = mean_phase_eve
    if par_fourier_fraction == 1:
        k_odd = np.log10(np.abs(output_dict['kSpace3D_odd_echoes']))
        k_even = np.log10(np.abs(output_dict['kSpace3D_even_echoes']))
    else:
        if k_fill == 'ZP':
            k_odd = np.zeros_like(output_dict['kSpace3D_odd_echoes'], dtype=float)
            k_odd[0:n_sl, :, :] = np.log10(np.abs(output_dict['kSpace3D_odd_echoes'][0:n_sl, :, :]))
            k_even = np.zeros_like(output_dict['kSpace3D_even_echoes'], dtype=float)
            k_even[0:n_sl, :, :] = np.log10(np.abs(output_dict['kSpace3D_even_echoes'][0:n_sl, :, :]))
        else:
            k_odd = np.log10(np.abs(output_dict['kSpace3D_odd_echoes']))
            k_even = np.log10(np.abs(output_dict['kSpace3D_even_echoes']))
    result_k_odd, k_odd, _ = utils.fix_image_orientation(k_odd, axes=axes_orientation, rd_direction=rd_direction)
    result_k_even, k_even, _ = utils.fix_image_orientation(k_even, axes=axes_orientation, rd_direction=rd_direction)

    if full_plot is True or full_plot == 'True' or full_plot == 1:
        # Image plot
        result_mag_odd, img_mag_odd, _ = utils.fix_image_orientation(img_mag_odd, axes=axes_orientation, rd_direction=rd_direction)
        result_mag_odd['row'] = 0
        result_mag_odd['col'] = 0

        result_mag_eve, img_mag_eve, _ = utils.fix_image_orientation(img_mag_eve, axes=axes_orientation, rd_direction=rd_direction)
        result_mag_eve['row'] = 1
        result_mag_eve['col'] = 0

        result_pha_odd, img_pha_odd, _ = utils.fix_image_orientation(img_pha_odd, axes=axes_orientation, rd_direction=rd_direction)
        result_pha_odd['row'] = 0
        result_pha_odd['col'] = 1

        result_pha_eve, img_pha_eve, _ = utils.fix_image_orientation(img_pha_eve, axes=axes_orientation, rd_direction=rd_direction)
        result_pha_eve['row'] = 1
        result_pha_eve['col'] = 1

        # k-space plot
        result_k_odd['title'] = "k-Space odd echoes"
        result_k_odd['row'] = 0
        result_k_odd['col'] = 2

        # k-space plot
        result_k_even['title'] = "k-Space even echoes"
        result_k_even['row'] = 1
        result_k_even['col'] = 2

        # Add results into the output attribute (result_1 must be the image to save in dicom)
        output = [result_mag_odd, result_pha_odd, result_k_odd, result_mag_eve, result_pha_eve, result_k_even]
    else:
        # Image plot
        img = img[:, :, n_points[0]//2 - n_rd_0//2:n_points[0]//2 + n_rd_0//2]
        result_1, img, _ = utils.fix_image_orientation(img, axes=axes_orientation, rd_direction=rd_direction)
        result_1['row'] = 0
        result_1['col'] = 0

        # k-space plot
        k_space = utils.run_dfft(img)
        result_2, k_space, _ = utils.fix_image_orientation(np.abs(k_space), axes=axes_orientation, rd_direction=rd_direction)
        result_2['row'] = 0
        result_2['col'] = 1

        # Dicom parameters
        dicom_meta_data["RepetitionTime"] = mat_data['repetitionTime']
        dicom_meta_data["EchoTime"] = mat_data['echoSpacing']
        dicom_meta_data["EchoTrainLength"] = mat_data['etl']

        # Add results into the output attribute (result_1 must be the image to save in dicom)
        output = [result_1, result_2]

    return output_dict, output