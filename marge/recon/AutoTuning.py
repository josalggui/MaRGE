import numpy as np
import scipy as sp
from scipy.interpolate import interp1d


def AutoTuning(raw_data_path=None):
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

    # Get results
    print(f"S11 = {mat_data['s11_db'].item()} dB")
    print(f"Serie: {mat_data['series'].item()}")
    print(f"Tuning: {mat_data['tuning'].item()}")
    print(f"Matching: {mat_data['matching'].item()}")


    outputs = []

    return output_dict, outputs