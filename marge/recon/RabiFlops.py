import scipy as sp
import numpy as np
import scipy.signal as sig
from scipy.interpolate import make_interp_spline
from marge.configs import hw_config as hw


def RabiFlops(raw_data_path=None):
    if raw_data_path is None:
        return None

    # load .mat
    mat_data = sp.io.loadmat(raw_data_path)
    output_dict = {}

    # Print inputs
    keys = mat_data['input_keys']
    strings = mat_data['input_strings']
    string = ""
    print("****Inputs****")
    for ii, key in enumerate(keys):
        string = string + f"{str(strings[ii]).strip()}: {np.squeeze(mat_data[str(key).strip()])}, "
    print(string)
    print("****Outputs****")

    # Get data and time vector
    nScans = mat_data['nScans'][0][0]
    nSteps = mat_data['nSteps'][0][0]
    nPoints = mat_data['nPoints'][0][0]
    dataOversampled = mat_data['dataOversampled'][0]
    timeVector = mat_data['rfTime'][0]

    # Get FID and Echo
    dataFull = sig.decimate(dataOversampled, hw.oversamplingFactor, ftype='fir', zero_phase=True)
    output_dict['dataFull'] = dataFull
    dataFull = np.reshape(dataFull, (nScans, nSteps, 2, -1))
    dataFID = dataFull[:, :, 0, :]
    output_dict['dataFID'] = dataFID
    dataEcho = dataFull[:, :, 1, :]
    output_dict['dataEcho'] = dataEcho
    dataFIDAvg = np.mean(dataFID, axis=0)
    output_dict['dataFIDAvg'] = dataFIDAvg
    dataEchoAvg = np.mean(dataEcho, axis=0)
    output_dict['dataEchoAvg'] = dataEchoAvg

    rabiFID = dataFIDAvg[:, 10]
    output_dict['rabiFID'] = rabiFID
    rabiEcho = dataEchoAvg[:, int(nPoints / 2)]
    output_dict['rabiEcho'] = rabiEcho

    # Analyze the curve
    piHalfTime, interpolations = analyze_rabi_curve(data=[timeVector, rabiFID, rabiEcho],
                                                    method=mat_data['cal_method'][0],
                                                    discriminator=mat_data['discriminator'][0])

    output_dict['piHalfTime'] = piHalfTime
    print("pi/2 pulse with RF amp = %0.2f a.u. and pulse time = %0.1f us" % (mat_data['rfExAmp'][0][0],
                                                                               piHalfTime))
    hw.b1Efficiency = np.pi / 2 / (mat_data['rfExAmp'][0][0] * piHalfTime)

    # Signal vs rf time
    result1 = {'widget': 'curve',
               'xData': [timeVector * 1e6, timeVector * 1e6, timeVector * 1e6,
                         interpolations[0] * 1e6, interpolations[0] * 1e6, interpolations[0] * 1e6],
               'yData': [np.abs(rabiFID), np.real(rabiFID), np.imag(rabiFID),
                         np.abs(interpolations[1]), np.real(interpolations[1]), np.imag(interpolations[1])],
               'xLabel': 'Time (us)',
               'yLabel': 'Signal amplitude (mV)',
               'title': 'Rabi Flops with FID',
               'legend': ['abs', 'real', 'imag', 'abs spline', 'real spline', 'imag spline'],
               'row': 0,
               'col': 0}

    result2 = {'widget': 'curve',
               'xData': [timeVector * 1e6, timeVector * 1e6, timeVector * 1e6,
                         interpolations[0] * 1e6, interpolations[0] * 1e6, interpolations[0] * 1e6],
               'yData': [np.abs(rabiEcho), np.real(rabiEcho), np.imag(rabiEcho),
                         np.abs(interpolations[2]), np.real(interpolations[2]), np.imag(interpolations[2])],
               'xLabel': 'Time (us)',
               'yLabel': 'Signal amplitude (mV)',
               'title': 'Rabi Flops with Spin Echo',
               'legend': ['abs', 'real', 'imag', 'abs spline', 'real spline', 'imag spline'],
               'row': 1,
               'col': 0}

    outputs = [result1, result2]

    return output_dict, outputs

def analyze_rabi_curve(data=None, method='ECHO', discriminator='min'):
    """
    Analyze a Rabi oscillation curve and estimate the pi/2 pulse time.

    This function takes a time-domain Rabi dataset (FID and ECHO signals) from rabiFlops.py,
    interpolates it with a cubic spline, and estimates the pi/2 pulse length
    by finding the first minimum or maximum in the smoothed envelope.

    Parameters
    ----------
    data : list or None, optional
        A list or tuple containing:
            - time : array-like
                Time values in seconds.
            - rabiFID : array-like
                Free Induction Decay (FID) signal amplitude for each.
            - rabiEcho : array-like
                Echo signal values (can be complex).
        If None, example data is generated for testing.

    method : str, optional
        Which signal to analyze. Options:
            - 'ECHO': Use the echo signal (default).
            - 'FID' : Use the FID signal.

    discriminator : str, optional
        How to estimate the pi/2 time:
            - 'min' : Find the first minimum in the oscillation envelope
                      and divide by 2 (default).
            - 'max' : Find the first maximum in the oscillation envelope.

    Returns
    -------
    pi_half_time : float
        Estimated pi/2 pulse time in microseconds.

    spline_data : list
        A list containing:
            - time_new : np.ndarray
                Interpolated time axis.
            - spline_fid : np.ndarray
                Interpolated FID signal.
            - spline_echo : np.ndarray
                Interpolated echo signal.

    Notes
    -----
    - The signals can be complex; real and imaginary parts are interpolated separately.
    - If the curve does not show a clear minimum/maximum,
      a warning is printed and the result may be inaccurate.
    - The returned pi_half_time is always in microseconds.
    """

    if data is None:
        time = np.linspace(0, 100, 10) * 1e-6
        signal = np.sin(2 * time / 100 * np.pi)
        data = [time, signal, signal]
    time = data[0]
    rabiFID = data[1]
    rabiEcho = data[2]

    # Interpolate with spline
    time_new = np.linspace(time.min(), time.max(), 300)
    spline_fid_real = make_interp_spline(time, np.real(rabiFID), k=3)
    spline_fid_imag = make_interp_spline(time, np.imag(rabiFID), k=3)
    spline_fid = spline_fid_real(time_new) + 1j * spline_fid_imag(time_new)
    spline_echo_real = make_interp_spline(time, np.real(rabiEcho), k=3)
    spline_echo_imag = make_interp_spline(time, np.imag(rabiEcho), k=3)
    spline_echo = spline_echo_real(time_new) + 1j * spline_echo_imag(time_new)
    if method == 'ECHO':
        amplitude_smooth = spline_echo
    elif method == 'FID':
        amplitude_smooth = spline_fid
    else:
        print("WARNING: unknown method '%s'" % method)
        return

    # Analyze curve
    n_steps = np.size(time_new)
    test = True
    n = 1
    while test:
        if n >= n_steps:
            print("WARNING: Rabi may be not properly calibrated")
            break
        d = np.abs(amplitude_smooth[n]) - np.abs(amplitude_smooth[n - 1])
        n += 1
        if d < 0:
            test = False
    if discriminator == 'min':
        test = True
        while test:
            if n >= n_steps:
                print("WARNING: Rabi may be not properly calibrated")
                break
            d = np.abs(amplitude_smooth[n]) - np.abs(amplitude_smooth[n - 1])
            n += 1
            if d > 0:
                test = False

    if discriminator == 'max':
        pi_half_time = time_new[n - 2] * 1e6  # us
    elif discriminator == 'min':
        pi_half_time = time_new[n - 2] * 1e6 / 2 # us
    else:
        return

    # plt.plot(time, rabiEcho, 'o', label='Original points')
    # plt.plot(time_new, amplitude_smooth, '-', label='Cubic spline')
    # plt.legend()
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.show()

    return pi_half_time, [time_new, spline_fid, spline_echo]