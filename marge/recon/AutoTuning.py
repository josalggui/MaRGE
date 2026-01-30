import numpy as np
import scipy as sp
from scipy.interpolate import interp1d


def AutoTuning(raw_data_path=None):
    # load .mat
    mat_data = sp.io.loadmat(raw_data_path)

    # Create new dictionary to save new outputs
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

    # Get results
    s11 = np.array(mat_data['s11_hist'])
    s11_opt = mat_data['s11'].item()
    f_vec = np.squeeze(mat_data['f_vec'])
    s_vec = np.squeeze(mat_data['s_vec'])
    frequency = mat_data['frequency'].item()

    # Interpolate s_vec
    interp_func = interp1d(f_vec, s_vec, kind='cubic')
    f_vec_t = np.linspace(np.min(f_vec), np.max(f_vec), 1000)
    s_vec_t = interp_func(f_vec_t)

    # Insert s11 into s_vec
    index = np.searchsorted(f_vec_t, frequency)
    f_vec_t = np.insert(f_vec_t, index, frequency)
    s_vec_t = np.insert(s_vec_t, index, s11_opt)

    # Get s in dB
    s_vec_db = 20 * np.log10(np.abs(s_vec_t))

    # Get quality factor
    try:
        idx = np.argmin(s_vec_db)
        f0 = f_vec_t[idx]
        f1 = f_vec_t[np.argmin(np.abs(s_vec_db[0:idx] + 3))]
        f2 = f_vec_t[idx + np.argmin(np.abs(s_vec_db[idx::] + 3))]
        q = f0 / (f2 - f1)
        print("Q = %0.0f" % q)
        print("BW @ -3 dB = %0.0f kHz" % ((f2 - f1) * 1e3))
        output_dict['Q'] = q
    except:
        pass

    # Create data array in case single point is acquired
    if mat_data['test'].item() == 'manual':
        s11 = np.squeeze(np.concatenate((s11, s11), axis=0))

    # Plot smith chart
    result1 = {'widget': 'smith',
               'xData': [np.real(s11), np.real(s_vec_t)],
               'yData': [np.imag(s11), np.imag(s_vec_t)],
               'xLabel': 'Real(S11)',
               'yLabel': 'Imag(S11)',
               'title': 'Smith chart',
               'legend': ['', ''],
               'row': 0,
               'col': 0}

    # Plot reflection coefficient
    result2 = {'widget': 'curve',
               'xData': (f_vec_t - frequency) * 1e3,
               'yData': [s_vec_db],
               'xLabel': 'Frequency (kHz)',
               'yLabel': 'S11 (dB)',
               'title': 'Reflection coefficient',
               'legend': [''],
               'row': 0,
               'col': 1}

    outputs = [result1, result2]

    return output_dict, outputs, dicom_meta_data