import marge.configs.hw_config as hw
import numpy as np
import scipy as sp
from marge.marge_utils import utils


def RarePyPulseq(raw_data_path=None):
    """
    Analyzes the sequence data and performs several steps including data extraction, processing,
    noise estimation, dummy pulse separation, signal decimation, data reshaping, Fourier transforms,
    and image reconstruction.

    Parameters:
    mode (str, optional): A string indicating the mode of operation. If set to 'Standalone',
                           additional plotting will be performed. Default is None.

    The method performs the following key operations:
    1. Extracts relevant data from `self.mapVals`, including the data for readouts, signal,
       noise, and dummy pulses.
    2. Decimates the signal data to match the desired bandwidth and reorganizes the data for
       further analysis.
    3. Performs averaging on the data and reorganizes it according to sweep order.
    4. Computes the central line and adjusts for any drift in the k-space data.
    5. Applies zero-padding to the data to match the expected resolution.
    6. Computes the k-space trajectory (kRD, kPH, kSL) and applies the phase correction.
    7. Performs inverse Fourier transforms to reconstruct the 3D image data.
    8. Saves the processed data and produces plots for visualization based on the mode of operation.
    9. Optionally outputs sampled data and performs DICOM formatting for medical imaging storage.

    The method also handles the creation of various output results that can be plotted in the GUI,
    including signal curves, frequency spectra, and 3D images. It also updates the metadata for
    DICOM storage.

    The sequence of operations ensures the data is processed correctly according to the
    hardware setup and scan parameters.

    Results are saved in `self.mapVals` and visualized depending on the provided mode. The method
    also ensures proper handling of rotation angles and field-of-view (dfov) values, resetting
    them as necessary.
    """

    if raw_data_path is None:
        return None

    # load .mat
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
    data_decimated = np.squeeze(mat_data['data_decimated'])
    n_points = np.squeeze(mat_data['nPoints'])
    partial_acquisition = mat_data['partialAcquisition'].item()
    n_rd, n_ph, n_sl = n_points
    n_batches = mat_data['n_batches'].item()
    n_readouts = mat_data['n_readouts'][0]
    ind = np.squeeze(mat_data['sweepOrder'])
    add_rd_points = mat_data['add_rd_points'].item()
    k_fill = mat_data['k_fill'].item()
    oversampling_factor = mat_data['oversampling_factor'].item()
    decimation_factor = mat_data['decimation_factor'].item()
    dummy_pulses = mat_data['dummyPulses'].item()
    rd_direction = mat_data['rd_direction'].item()
    n_noise = mat_data['nNoise'].item()
    full_plot = mat_data['full_plot'].item()

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
                    data_dummy = np.concatenate((data_dummy, data_prov[
                        points_per_rd * n_noise:points_per_rd * n_noise + points_per_train]), axis=0)
                data_signal = np.concatenate((data_signal, data_prov[points_per_rd * n_noise + points_per_train::]),
                                             axis=0)
            else:
                data_noise = np.concatenate((data_noise, data_prov[0:points_per_rd]), axis=0)
                if mat_data['dummyPulses'].item() > 0:
                    data_dummy = np.concatenate((data_dummy, data_prov[points_per_rd:points_per_rd + points_per_train]),
                                                axis=0)
                data_signal = np.concatenate((data_signal, data_prov[points_per_rd + points_per_train::]), axis=0)
            idx_0 = idx_1
        n_readouts[batch] += -n_rd - n_rd * mat_data['etl'].item()
    data_noise = np.reshape(data_noise, (-1, n_points[0] + add_rd_points * 2))
    data_noise = data_noise[:, add_rd_points: -add_rd_points]
    output_dict['data_noise'] = data_noise
    output_dict['data_dummy'] = data_dummy
    output_dict['data_signal'] = data_signal

    # Decimate data to get signal in desired bandwidth
    data_full = data_signal

    # Reorganize data_full
    data_prov = np.zeros(shape=[n_scans, n_sl * n_ph * n_rd], dtype=complex)
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

    # # Save data_full to save it in .h5
    # self.data_fullmat = data_full

    # Get index for krd = 0
    # Average data
    data_prov = np.reshape(data_full, shape=(n_scans, n_rd * n_ph * n_sl))
    data_prov = np.average(data_prov, axis=0)
    # Reorganize the data according to sweep mode
    data_prov = np.reshape(data_prov, shape=(n_sl, n_ph, n_rd))
    data_temp = np.zeros_like(data_prov)
    for ii in range(n_ph):
        data_temp[:, ind[ii], :] = data_prov[:, ii, :]
    data_prov = data_temp
    # Get central line
    data_prov = data_prov[int(n_points[2] / 2), int(n_ph / 2), :]
    ind_krd_0 = np.argmax(np.abs(data_prov))
    if ind_krd_0 < n_rd / 2 - add_rd_points or ind_krd_0 > n_rd / 2 + add_rd_points:
        ind_krd_0 = int(n_rd / 2)

    # Get individual images
    data_full = np.reshape(data_full, shape=(n_scans, n_sl, n_ph, n_rd))
    data_full = data_full[:, :, :, ind_krd_0 - int(n_points[0] / 2):ind_krd_0 + int(n_points[0] / 2)]
    data_temp = np.zeros_like(data_full)
    for ii in range(n_ph):
        data_temp[:, :, ind[ii], :] = data_full[:, :, ii, :]
    data_full = data_temp
    output_dict['data_full'] = data_full

    # Average data
    data = np.average(data_full, axis=0)

    # Do zero padding or POCS
    data_temp = np.zeros(shape=(n_points[2], n_points[1], n_points[0]), dtype=complex)
    data_temp[0:n_sl, :, :] = data
    if k_fill == 'POCS':
        data_temp = utils.run_pocs_reconstruction(n_points=n_points[::-1], factors=[par_fourier_fraction, 1, 1], k_space_ref=data_temp)

    # #############################################
    # print("Raw image statistics:")
    # noise_2 = int(np.std(data_temp) * 100)
    # print(f"Std noise: {noise_2}")
    # image_2 = int(np.std(image_temp) * 1e6)
    # print(f"Std image: {image_2}")
    # #############################################

    # METHOD 1: Get decimated k-space by FFT
    subdecimation_method = 'FFT'
    if subdecimation_method=='FFT' and (full_plot==False or full_plot=='False'):
        image_temp = utils.run_ifft(data_temp)
        idx_0 = n_points[0] // 2 - n_points[0] // 2 * decimation_factor // oversampling_factor
        idx_1 = n_points[0] // 2 + n_points[0] // 2 * decimation_factor // oversampling_factor
        image_temp = image_temp[:, :, idx_0:idx_1]
        data_temp = utils.run_dfft(image_temp)
        n_points[0] = n_rd_0
        fov[0] = fov[0] * decimation_factor / oversampling_factor

    # METHOD 2: Get image by decimating k-space with average of consecutive samples.
    if subdecimation_method=='AVG' and (full_plot==False or full_plot=='False'):
        n_points[0] = n_rd_0
        fov[0] = fov[0] * decimation_factor / oversampling_factor
        data_prov = np.zeros((n_points[2], n_points[1], n_points[0]), dtype=complex)
        dii = oversampling_factor // decimation_factor
        for ii in range(n_rd_0):
            data_prov[:, :, ii] = np.mean(data_temp[:, :, dii*ii:dii*(ii+1)], axis=2)
        data_temp = data_prov
        image_temp = utils.run_ifft(data_temp)

    # #############################################
    # print("Target image statistics:")
    # noise_2 = int(np.std(data) * 100)
    # print(f"Std noise: {noise_2}")
    # image_2 = int(np.std(image_temp) * 1e6)
    # print(f"Std image: {image_2}")
    # #############################################

    # Fix the position of the sample according to dfov
    data = np.reshape(data_temp, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    bw = mat_data['bw_MHz'].item()
    time_vector = np.linspace(-n_points[0] / bw / 2 + 0.5 / bw, n_points[0] / bw / 2 - 0.5 / bw, n_points[0]) * 1e-6 # s
    kMax = np.squeeze(np.array(n_points) / (2 * np.array(fov)) * np.array(mat_data['axes_enable']))
    kRD = time_vector * hw.gammaB * mat_data['rd_grad_amplitude'].item()
    kPH = np.linspace(-kMax[1], kMax[1], num=n_points[1], endpoint=False)
    kSL = np.linspace(-kMax[2], kMax[2], num=n_points[2], endpoint=False)
    kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
    kRD = np.reshape(kRD, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    kPH = np.reshape(kPH, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    kSL = np.reshape(kSL, shape=(1, n_points[0] * n_points[1] * n_points[2]))
    dPhase = np.exp(2 * np.pi * 1j * (dfov[0] * kRD + dfov[1] * kPH + dfov[2] * kSL))
    data = np.reshape(data * dPhase, shape=(n_points[2], n_points[1], n_points[0]))
    output_dict['kSpace3D'] = data
    output_dict['image3D'] = utils.run_ifft(data)
    data = np.reshape(data, shape=(1, n_points[0] * n_points[1] * n_points[2]))

    # Create sampled data
    kRD = np.reshape(kRD, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    kPH = np.reshape(kPH, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    kSL = np.reshape(kSL, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    data = np.reshape(data, shape=(n_points[0] * n_points[1] * n_points[2], 1))
    output_dict['kMax_1/m'] = kMax
    output_dict['sampled'] = np.concatenate((kRD, kPH, kSL, data), axis=1)
    output_dict['sampledCartesian'] = output_dict['sampled']  # To sweep
    data = np.reshape(data, shape=(n_points[2], n_points[1], n_points[0]))

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

    if axes_enable[1] == 0 and axes_enable[2] == 0:
        bw = mat_data['bw_MHz'] * 1e-3  # kHz
        acqTime = mat_data['acqTime']  # ms
        tVector = np.linspace(-acqTime / 2, acqTime / 2, n_points[0])
        sVector = mat_data['sampled'][:, 3]
        fVector = np.linspace(-bw / 4, bw / 4, n_rd_0)
        iVector = utils.run_ifft(sVector)
        iVector = iVector[n_points[0]//2 - n_rd_0//2:n_points[0]//2 + n_rd_0//2]

        # Plots to show into the GUI
        result_1 = {}
        result_1['widget'] = 'curve'
        result_1['xData'] = tVector
        result_1['yData'] = [np.abs(sVector), np.real(sVector), np.imag(sVector)]
        result_1['xLabel'] = 'Time (ms)'
        result_1['yLabel'] = 'Signal amplitude (mV)'
        result_1['title'] = "Signal"
        result_1['legend'] = ['Magnitude', 'Real', 'Imaginary']
        result_1['row'] = 0
        result_1['col'] = 0

        result_2 = {}
        result_2['widget'] = 'curve'
        result_2['xData'] = fVector
        result_2['yData'] = [np.abs(iVector)]
        result_2['xLabel'] = 'Frequency (kHz)'
        result_2['yLabel'] = "Amplitude (a.u.)"
        result_2['title'] = "Spectrum"
        result_2['legend'] = ['Spectrum magnitude']
        result_2['row'] = 1
        result_2['col'] = 0

        output = [result_1, result_2]

    else:
        # Plot image
        image = np.abs(output_dict['image3D'])

        # Image plot
        if mat_data['unlock_orientation'] == 0:
            result_1, _, _ = utils.fix_image_orientation(image, axes=axes_orientation, rd_direction=rd_direction)
            result_1['row'] = 0
            result_1['col'] = 0
        else:
            result_1 = {'widget': 'image',
                        'data': image,
                        'xLabel': "%s" % axesStr[1],
                        'yLabel': "%s" % axesStr[0],
                        'title': "i-Space",
                        'row': 0,
                        'col': 0}

        # k-space plot
        if par_fourier_fraction == 1:
                data = np.log10(np.abs(output_dict['kSpace3D']))
                data = np.abs(output_dict['kSpace3D'])
        else:
            if k_fill == 'ZP':
                data = np.zeros_like(output_dict['kSpace3D'], dtype=float)
                data[0:n_sl, :, :] = np.log10(np.abs(output_dict['kSpace3D'][0:n_sl, :, :]))
            elif k_fill == 'POCS':
                data = np.log10(np.abs(output_dict['kSpace3D']))
        
        if mat_data['unlock_orientation'] == 0:
            result_2, _, _ = utils.fix_image_orientation(data, axes=axes_orientation, rd_direction=rd_direction)
            result_2['row'] = 0
            result_2['col'] = 1
        elif mat_data['unlock_orientation'] == 1:
            result_2 = {'widget': 'image',
                        'data': data,
                        'xLabel': "%s" % axesStr[1],
                        'yLabel': "%s" % axesStr[0],
                        'title': "i-Space",
                        'row': 0,
                        'col': 1}
        
        # Dicom parameters
        dicom_meta_data["RepetitionTime"] = mat_data['repetitionTime']
        dicom_meta_data["EchoTime"] = mat_data['echoSpacing']
        dicom_meta_data["EchoTrainLength"] = mat_data['etl']

        # Add results into the output attribute (result_1 must be the image to save in dicom)
        output = [result_1, result_2]

    # Save results
    # self.save_ismrmrd()

    return output_dict, output

if __name__ == '__main__':
    RarePyPulseq(raw_data_path="C:\CSIC\REPOSITORIOS\MaRCoS\MaRGE\RareDoubleImage.2025.08.18.17.29.50.502.mat")