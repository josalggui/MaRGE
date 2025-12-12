import numpy as np
import scipy as sp
import marge.configs.hw_config as hw


def Larmor(raw_data_path=None):
    """
    Analyzes the acquired data from the Larmor sequence to determine the Larmor frequency
    and compute the corresponding time-domain signal and frequency spectrum.

    This method processes the acquired data by generating time and frequency vectors,
    performing a Fourier transform to obtain the signal spectrum, and determining the
    central frequency. It updates the Larmor frequency and provides the results in both
    time and frequency domains. The data is then optionally plotted, and the results are
    saved for further analysis.

    Args:
        mode (str, optional): The mode of execution. If set to 'Standalone', the results are plotted
                               in a standalone manner. Defaults to None.

    Returns:
        list: A list containing the time-domain signal and frequency spectrum for visualization.

    Notes:
        - The Larmor frequency is recalculated based on the signal's central frequency from the spectrum.
        - The time-domain signal and frequency spectrum are both included in the output layout for visualization.
        - The results are saved as raw data and can be accessed later.
        - If the mode is not 'Standalone', the Larmor frequency is updated in all sequences in the sequence list.
    """
    
    if raw_data_path is None:
        return None

    # load .mat
    mat_data = sp.io.loadmat(raw_data_path)

    # Create new dictionary to save new outputs
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

    # Load data
    signal = mat_data['data'][0]
    acq_time = mat_data['acqTime'][0][0] * 1e3  # ms
    n_points = mat_data['nPoints'][0][0]  # kHz
    bw = mat_data['bw'][0][0]

    # Generate time and frequency vectors and calculate the signal spectrum
    tVector = np.linspace(-acq_time / 2, acq_time / 2, n_points)
    fVector = np.linspace(-bw / 2, bw / 2, n_points)
    spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal)))

    # Get the central frequency
    idf = np.argmax(np.abs(spectrum))
    fCentral = fVector[idf] * 1e-3  # MHz
    hw.larmorFreq = mat_data['larmorFreq'][0][0] + fCentral
    print('Larmor frequency: %1.5f MHz' % hw.larmorFreq)
    output_dict['fCentral'] = fCentral
    output_dict['larmorFreq0'] = hw.larmorFreq
    output_dict['signalVStime'] = [tVector, signal]
    output_dict['spectrum'] = [fVector, spectrum]

    # Add time signal to the layout
    result1 = {'widget': 'curve',
               'xData': tVector,
               'yData': [np.abs(signal), np.real(signal), np.imag(signal)],
               'xLabel': 'Time (ms)',
               'yLabel': 'Signal amplitude (mV)',
               'title': 'Echo',
               'legend': ['abs', 'real', 'imag'],
               'row': 0,
               'col': 0}

    # Add frequency spectrum to the layout
    result2 = {'widget': 'curve',
               'xData': fVector,
               'yData': [np.abs(spectrum)],
               'xLabel': 'Frequency (kHz)',
               'yLabel': 'Spectrum amplitude (a.u.)',
               'title': 'Spectrum',
               'legend': [''],
               'row': 1,
               'col': 0}

    outputs = [result1, result2]

    return output_dict, outputs
