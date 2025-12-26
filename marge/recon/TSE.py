import marge.configs.hw_config as hw
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit


def TSE(raw_data_path=None):
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
    
    # Get data
    data = mat_data['data'][0]
    etl = mat_data['etl'].item()
    echo_spacing = mat_data['echoSpacing'].item()
    acq_time = mat_data['acqTime'].item()
    n_points = mat_data['nPoints'].item()

    # Prepare data for full data plot
    data_echoes = np.reshape(data, (etl, -1))
    data_echoes[:, 0] = 0.
    data_echoes[:, -1] = 0.
    data_echoes = np.reshape(data, -1)
    t0 = np.linspace(echo_spacing - acq_time / 2, echo_spacing + acq_time / 2,
                     n_points)  # ms
    t1_vector = t0
    for echo_index in range(etl - 1):
        t1_vector = np.concatenate((t1_vector, t0 + echo_spacing * (echo_index + 1)), axis=0)

    # Prepare data for echo amplitude vs echo time
    data = np.reshape(data, (etl, -1))
    data = data[:, int(n_points / 2)]
    t2_vector = np.linspace(echo_spacing, echo_spacing * etl, num=etl, endpoint=True)  # ms

    # Save point here to sweep class
    output_dict['sampledPoint'] = data[0]

    # Functions for fitting
    def func1(x, m, t2):
        return m * np.exp(-x / t2)

    # Fitting to functions
    fitData1, xxx = curve_fit(func1, t2_vector, np.abs(data), p0=[np.abs(data[0]), 10])
    fitting1 = func1(t2_vector, fitData1[0], fitData1[1])
    corr_coef1 = np.corrcoef(np.abs(data), fitting1)
    print('For one component:')
    print('rho: %0.1f' % round(fitData1[0], 1))
    print('T2 (ms): %0.1f ms' % round(fitData1[1], 1))
    print('Correlation: %0.3f' % corr_coef1[0, 1])
    output_dict['T21'] = fitData1[1]
    output_dict['M1'] = fitData1[0]

    # Signal vs rf time
    result1 = {'widget': 'curve',
               'xData': t2_vector,
               'yData': [np.abs(data),
                         func1(t2_vector, fitData1[0], fitData1[1])],
               'xLabel': 'Echo time (ms)',
               'yLabel': 'Echo amplitude (mV)',
               'title': 'Echo amplitude VS Echo time',
               'legend': ['Experimental at echo time',
                          'Fitting to monoexponential'],
               'row': 0,
               'col': 0}

    result2 = {'widget': 'curve',
               'xData': t1_vector,
               'yData': [np.abs(data_echoes)],
               'xLabel': 'Echo time (ms)',
               'yLabel': 'Echo amplitude (mV)',
               'title': 'Echo train',
               'legend': ['Experimental measurement'],
               'row': 1,
               'col': 0}

    # Save results into rawData
    t1_vector = np.reshape(t1_vector, (-1, 1))
    data_echoes = np.reshape(data_echoes, (-1, 1))
    output_dict['signal_vs_time'] = np.concatenate((t1_vector, data_echoes), axis=1)
    t2_vector = np.reshape(t2_vector, (-1, 1))
    data = np.reshape(data, (-1, 1))
    output_dict['echo_amplitude_vs_time'] = np.concatenate((t2_vector, data), axis=1)

    # Create self.output to run in iterative mode
    results = [result1, result2]

    return output_dict, results