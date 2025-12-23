"""
Sequence Analysis Template
--------------------------
Use this file as a starting point for writing new sequence analysis functions.

Expected .mat fields (customize as needed):
- data_decimated
- nPoints
- bw_MHz
- nRepetitions
- input_keys (optional)
- input_strings (optional)
"""

import numpy as np
import scipy.io as sio


def Default_SeqName(raw_data_path=None):
    """
    Template function for sequence analysis.

    Parameters
    ----------
    raw_data_path : str
        Path to the .mat file containing raw sequence data

    Returns
    -------
    output_dict : dict
        Dictionary to be stored back into raw data (metrics, parameters, etc.)
    outputs : list
        List of plotting dictionaries
    """

    # ------------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------------
    if raw_data_path is None:
        return None

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    mat_data = sio.loadmat(raw_data_path)
    output_dict = {}  # Store numerical results or metadata here

    # ------------------------------------------------------------------
    # Print input parameters (optional)
    # ------------------------------------------------------------------
    try:
        keys = mat_data["input_keys"]
        strings = mat_data["input_strings"]

        print("**** Inputs ****")
        input_string = ""
        for ii, key in enumerate(keys):
            key_name = str(key).strip()
            label = str(strings[ii]).strip()
            value = np.squeeze(mat_data[key_name])
            input_string += f"{label}: {value}, "
        print(input_string)

    except Exception:
        # Inputs are optional
        pass

    print("**** Outputs ****")

    # ------------------------------------------------------------------
    # Load required variables (CUSTOMIZE THIS SECTION)
    # ------------------------------------------------------------------
    data = mat_data["data_decimated"][0]
    n_points = mat_data["nPoints"].item()
    bw = mat_data["bw_MHz"].item()
    n_repetitions = mat_data["nRepetitions"].item()

    # ------------------------------------------------------------------
    # Perform analysis (CUSTOMIZE THIS SECTION)
    # ------------------------------------------------------------------
    t_vector = np.linspace(
        0,
        n_points * n_repetitions / bw,
        n_points * n_repetitions
    )

    # Example: store derived values
    output_dict["max_signal"] = np.max(np.abs(data))
    output_dict["mean_signal"] = np.mean(np.abs(data))

    # ------------------------------------------------------------------
    # Build plotting outputs (CUSTOMIZE THIS SECTION)
    # ------------------------------------------------------------------
    result_1 = {
        "widget": "curve",
        "xData": t_vector,
        "yData": [
            np.abs(data),
            np.real(data),
            np.imag(data)
        ],
        "xLabel": "Time (ms)",
        "yLabel": "Signal amplitude (mV)",
        "title": "Sequence Analysis Template",
        "legend": ["abs", "real", "imag"],
        "row": 0,
        "col": 0
    }

    outputs = [result_1]

    # ------------------------------------------------------------------
    # Return results
    # ------------------------------------------------------------------
    return output_dict, outputs
