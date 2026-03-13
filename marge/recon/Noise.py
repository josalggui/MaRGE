import marge.configs.hw_config as hw
import numpy as np
import scipy as sp


def Noise(raw_data_path=None):
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

    # Get data and time vector
    data = np.squeeze(mat_data['data_decimated'])
    bw = mat_data['bw'][0][0]  # kHz
    acq_time = mat_data['nPoints'][0][0] / bw
    channels = np.squeeze(mat_data['channels'])

    # If data comes from MIMO, it retrieves data from channel 1 of master RP
    if len(data.shape) == 1:
        data = [data]
        channels = [channels]

    # Get spectrum a frequency vector
    spectrums = []
    datas = []
    legends = []
    n = 0
    for single_data in data:
        spectrums.append(np.abs(np.squeeze(np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(single_data))))))
        datas.append(np.abs(single_data))
        legends.append(f"Channel {channels[n]}")
        n += 1
    f_vector = np.linspace(-bw / 2, bw / 2, num=np.size(single_data), endpoint=False)
    t_vector = np.linspace(0, acq_time, np.size(single_data))


    # Get rms noise
    noiserms = np.std(data[0])
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
               'yData': datas,
               'xLabel': 'Time (ms)',
               'yLabel': 'Signal amplitude (mV)',
               'title': 'Noise vs time',
               'legend': legends,
               'row': 0,
               'col': 0}

    # Plot spectrum
    result2 = {'widget': 'curve',
               'xData': f_vector,
               'yData': spectrums,
               'xLabel': 'Frequency (kHz)',
               'yLabel': 'Mag FFT (a.u.)',
               'title': 'Noise spectrum',
               'legend': legends,
               'row': 1,
               'col': 0}

    outputs = [result1, result2]

    return output_dict, outputs, dicom_meta_data