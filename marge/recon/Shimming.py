import numpy as np
import scipy as sp
from marge.configs import units as units
from marge.configs import hw_config as hw

def Shimming(raw_data_path=None):
    if raw_data_path is None:
        return None

    # load .mat
    mat_data = sp.io.loadmat(raw_data_path)
    output_dict = {}

    # Load data
    data = mat_data['data'][0]
    n_points = mat_data['nPoints'][0][0]
    n_shimming = mat_data['nShimming'][0][0]

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
    data = np.reshape(data, shape=(3, n_shimming, -1))

    def getFWHM(s=None):
        bw = mat_data['bw'] * 1e-3
        f_vector = np.linspace(-bw / 2, bw / 2, n_points)
        target = np.max(s) / 2
        p0 = np.argmax(s)
        f0 = f_vector[p0]
        s1 = np.abs(s[0:p0] - target)
        f1 = f_vector[np.argmin(s1)]
        s2 = np.abs(s[p0::] - target)
        f2 = f_vector[np.argmin(s2) + p0]
        return f2 - f1

    # Get FFT
    dataFFT = np.zeros((3, n_shimming))
    dataFWHM = np.zeros((3, n_shimming))
    for ii in range(3):
        for jj in range(n_shimming):
            spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data[ii, jj, :]))))
            dataFFT[ii, jj] = np.max(spectrum)
            dataFWHM[ii, jj] = getFWHM(spectrum)
    output_dict['amplitudeVSshimming'] = dataFFT

    # Get max signal for each excitation
    sxVector = np.squeeze(mat_data['sxVector'])
    syVector = np.squeeze(mat_data['syVector'])
    szVector = np.squeeze(mat_data['szVector'])

    # Get the shimming values
    sx = sxVector[np.argmax(dataFFT[0, :])]
    sy = syVector[np.argmax(dataFFT[1, :])]
    sz = szVector[np.argmax(dataFFT[2, :])]
    fwhm = dataFWHM[2, np.argmax(dataFFT[2, :])]
    print("Shimming X = %0.1f" % (sx / units.sh))
    print("Shimming Y = %0.1f" % (sy / units.sh))
    print("Shimming Z = %0.1f" % (sz / units.sh))
    print("FWHM = %0.0f Hz" % (fwhm * 1e3))
    print("Homogeneity = %0.0f ppm" % (fwhm * 1e3 / hw.larmorFreq))
    print("Shimming loaded into the sequences.")

    # Shimming plot
    result1 = {'widget': 'curve',
               'xData': [sxVector / units.sh, syVector / units.sh, szVector / units.sh],
               'yData': [np.abs(dataFFT[0, :]), np.abs(dataFFT[1, :]), np.abs(dataFFT[2, :])],
               'xLabel': 'Shimming',
               'yLabel': 'a.u.',
               'title': 'Spectrum amplitude',
               'legend': ['X', 'Y', 'Z'],
               'row': 0,
               'col': 0}

    result2 = {'widget': 'curve',
               'xData': [sxVector / units.sh, syVector / units.sh, szVector / units.sh],
               'yData': [dataFWHM[0, :], dataFWHM[1, :], dataFWHM[2, :]],
               'xLabel': 'Shimming',
               'yLabel': 'FWHM (kHz)',
               'title': 'FWHM',
               'legend': ['X', 'Y', 'Z'],
               'row': 1,
               'col': 0}

    shimming = [np.round(sx / units.sh, decimals=1),
                np.round(sy / units.sh, decimals=1),
                np.round(sz / units.sh, decimals=1)]
    output_dict['shimming0'] = shimming

    return output_dict, [result1, result2]