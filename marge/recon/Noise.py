import marge.configs.hw_config as hw
import numpy as np
import scipy as sp


def Noise(raw_data_path=None):
    if raw_data_path is None:
        return None

    # load .mat
    mat_data = sp.io.loadmat(raw_data_path)
    output_dict = {}

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

    # Get data and time vector
    data = mat_data['data']
    data = np.squeeze(data)
    bw = mat_data['bw'][0][0]  # kHz
    acq_time = mat_data['nPoints'][0][0] / bw
    t_vector = np.linspace(0, acq_time, np.size(data))

    # Get spectrum a frequency vector
    spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(data)))
    spectrum = np.squeeze(spectrum)
    f_vector = np.linspace(-bw / 2, bw / 2, num=np.size(data), endpoint=False)

    # Get rms noise
    noiserms = np.std(data)
    noiserms = noiserms * 1e3  # uV
    print('rms noise: %0.1f uV @ %0.1f kHz' % (noiserms, bw))
    johnson = np.sqrt(2 * 50 * 300 * bw * 1e3 * 1.38e-23) * 10 ** (hw.lnaGain / 20) * 1e6 / 2  # uV
    print('Thermal limit: %0.1f uV @ %0.1f kHz' % (johnson, bw))
    print('Noise factor: %0.1f' % (noiserms / johnson))
    if noiserms / johnson > 2:
        print("WARNING: Noise is too high")

    # Plot signal versus time
    result1 = {'widget': 'curve',
               'xData': t_vector,
               'yData': [np.abs(data), np.real(data), np.imag(data)],
               'xLabel': 'Time (ms)',
               'yLabel': 'Signal amplitude (mV)',
               'title': 'Noise vs time',
               'legend': ['abs', 'real', 'imag'],
               'row': 0,
               'col': 0}

    # Plot spectrum
    result2 = {'widget': 'curve',
               'xData': f_vector,
               'yData': [np.abs(spectrum)],
               'xLabel': 'Frequency (kHz)',
               'yLabel': 'Mag FFT (a.u.)',
               'title': 'Noise spectrum',
               'legend': [''],
               'row': 1,
               'col': 0}

    outputs = [result1, result2]

    return output_dict, outputs