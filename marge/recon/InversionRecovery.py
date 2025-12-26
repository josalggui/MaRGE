import numpy as np
import scipy as sp
from scipy.optimize import curve_fit


def InversionRecovery(raw_data_path=None):
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
    data_full = mat_data['data'][0]
    n_scans = mat_data['nScans'].item()
    n_steps = mat_data['nSteps'].item()
    n_points = mat_data['nPoints'].item()
    acq_time = mat_data['acqTime'].item()
    ir_time_vector = mat_data['irTimeVector'][0] * 1e-3

    # Functions for fitting
    def func1(x, m, t1):
        return np.abs(m * 1 - 2 * np.exp(-x / t1))

    # Process data to be plotted
    data_full = np.reshape(data_full, (n_scans, n_steps, -1))
    data_full = np.average(data_full, axis=0)
    data = data_full[:, int(n_points / 2)]
    t0 = np.linspace(0, acq_time, n_points, endpoint=False)
    for ii in range(n_steps-1):
        t0_prov = t0 + acq_time
        np.concatenate((t0, t0_prov), axis=0)

    # Fitting to functions
    fitData1, _ = curve_fit(func1, ir_time_vector, np.abs(data), p0=[np.abs(data[0]), 10])
    fitting1 = func1(ir_time_vector, fitData1[0], fitData1[1])
    corr_coef1 = np.corrcoef(np.abs(data), fitting1)
    print('For one component:')
    print('rho: %0.1f' % round(fitData1[0], 1))
    print('T1 (ms): %0.1f ms' % round(fitData1[1], 1))
    print('Correlation: %0.3f' % corr_coef1[0, 1])
    output_dict['T1'] = fitData1[1]
    output_dict['M1'] = fitData1[0]

    # Time vector for full signal
    t_readout = np.linspace(0, acq_time, n_points, endpoint=False)
    t = np.array([])
    for ii in range(n_steps):
        t = np.concatenate((t, t_readout + ii * acq_time), axis=0)

    # Signal vs inverion time
    data = [ir_time_vector, data]
    result1 = {'widget': 'curve',
               'xData': data[0],
               'yData': [np.abs(data[1]),
                         func1(ir_time_vector, fitData1[0], fitData1[1])],
               'xLabel': 'Time (ms)',
               'yLabel': 'Signal amplitude (mV)',
               'title': '',
               'legend': ['Experimental at echo time',
                          'Fitting to monoexponential'],
               'row': 0,
               'col': 0}

    data_full = np.reshape(data_full, -1)
    result2 = {'widget': 'curve',
               'xData': t,
               'yData': [np.abs(data_full),
                         np.real(data_full),
                         np.imag(data_full)],
               'xLabel': 'Echo time (ms)',
               'yLabel': 'Echo amplitude (mV)',
               'title': 'Echo train',
               'legend': ['abs', 'real', 'imag'],
               'row': 1,
               'col': 0}
    
    # create self.out to run in iterative mode
    results = [result1, result2]
    
    return output_dict, results