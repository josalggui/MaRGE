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
    
    # Get data
    data = mat_data['data'][0]
    etl = mat_data['etl'].item()
    echo_spacing = mat_data['echoSpacing'].item()
    acq_time = mat_data['acqTime'].item()
    n_points = mat_data['nPoints'].item()
    phase_mode = mat_data['phase_mode'].item()

    # --- reshape raw data into echoes ---
    data_2d = np.reshape(data, (etl, -1))

    # --- global phase correction using first echo ---
    abs0 = np.abs(data_2d[0])
    center0 = np.argmax(abs0)
    w_phase = 3

    ref = np.sum(data_2d[0, center0 - w_phase:center0 + w_phase + 1])
    phi = np.angle(ref)

    data_2d *= np.exp(-1j * phi)

    # --- echo-center tracking + symmetric integration ---
    w_int = 3  # half-width of integration window (points)
    echo_integrals = np.zeros(etl)
    echo_centers = np.zeros(etl, dtype=int)

    for k in range(etl):
        # locate echo center from magnitude
        mag = np.abs(data_2d[k])
        c = np.argmax(mag)
        echo_centers[k] = c

        # guard against boundary issues
        c0 = max(c - w_int, 0)
        c1 = min(c + w_int + 1, n_points)

        echo_integrals[k] = np.sum(
            np.real(data_2d[k, c0:c1])
        )
    # --- enforce consistent echo sign for APCPMG ---
    if phase_mode == 'APCPMG':
        # Use the first *reliable* echo as sign reference
        ref_sign = np.sign(echo_integrals[1])  # echo 2 (echo 1 discarded anyway)
        ref_sign = 1.0 if ref_sign == 0 else ref_sign

        for k in range(etl):
            if np.sign(echo_integrals[k]) != ref_sign:
                echo_integrals[k] *= -1.0

    # Prepare data for full data plot
    data_echoes = data_2d.copy()
    data_echoes[:, 0] = 0.
    data_echoes[:, -1] = 0.
    data_echoes = np.reshape(data_echoes, -1)
    t0 = np.linspace(echo_spacing - acq_time / 2, echo_spacing + acq_time / 2,
                     n_points)  # ms
    t1_vector = t0
    for echo_index in range(etl - 1):
        t1_vector = np.concatenate((t1_vector, t0 + echo_spacing * (echo_index + 1)), axis=0)

    # --- time axis: echo centers relative to excitation ---
    t2_vector = (np.arange(1, etl + 1) * echo_spacing)  # ms

    # --- discard first echo in the fit ---
    fit_data = echo_integrals[1:]
    fit_time = t2_vector[1:]

    # --- remove non-finite points before fitting ---
    mask = np.isfinite(fit_data) & np.isfinite(fit_time)
    fit_data = fit_data[mask]
    fit_time = fit_time[mask]

    # Save point here to sweep class
    output_dict['sampledPoint'] = echo_integrals[1]  # first echo used in fit

    # Functions for fitting
    def func1(x, m, t2):
        return m * np.exp(-x / t2)
    def func2(x, As, Ts, Al, Tl):
        return As * np.exp(-x / Ts) + Al * np.exp(-x / Tl)

    try:
        # Fitting to functions
        # --- mono-exponential fit ---
        fitData1, _ = curve_fit(func1, fit_time, fit_data,
                                p0=[fit_data[0], 10])
        fitting1 = func1(fit_time, fitData1[0], fitData1[1])
        M0 = fitData1[0]
        T2 = fitData1[1]
        print('Mono-exponential fit:')
        print(f'T2 = {T2:.2f} ms')
        output_dict['mono'] = {'M0': M0, 'T2': T2}

        # --- fitted curve extrapolated back to t = 0 ---
        t_fit_plot = np.linspace(
            0.0,
            t2_vector[-1],
            200
        )  # ms, smooth curve
        fit_curve_plot = func1(t_fit_plot, M0, T2)
        # Signal vs rf time
        # --- interpolate experimental data onto fit axis for plotting ---
        exp_interp = np.interp(
            t_fit_plot,
            fit_time,
            fit_data,
            left=np.nan,
            right=np.nan
        )
        result1 = {
            'widget': 'curve',
            'xData': t_fit_plot,
            'yData': [exp_interp, fit_curve_plot],
            'xLabel': 'Echo time (ms)',
            'yLabel': 'Integrated echo amplitude (a.u.)',
            'title': 'Echo-integrated amplitudes (fit from echo 2, extrapolated to t = 0)',
            'legend': [
                'Experimental (echoes 2…end)',
                'Mono-exponential fit (extrapolated)'
            ],
            'row': 0,
            'col': 0
        }
    except:
        print('Mono-exponential fit failed.')

        # --- fitted curve extrapolated back to t = 0 ---
        t_fit_plot = np.linspace(
            0.0,
            t2_vector[-1],
            200
        )  # ms, smooth curve
        # --- interpolate experimental data onto fit axis for plotting ---
        exp_interp = np.interp(
            t_fit_plot,
            fit_time,
            fit_data,
            left=np.nan,
            right=np.nan
        )
        result1 = {
            'widget': 'curve',
            'xData': t_fit_plot,
            'yData': [exp_interp],
            'xLabel': 'Echo time (ms)',
            'yLabel': 'Integrated echo amplitude (a.u.)',
            'title': 'Echo-integrated amplitudes (fit from echo 2, extrapolated to t = 0)',
            'legend': [
                'Experimental (echoes 2…end)'
            ],
            'row': 0,
            'col': 0
        }

    try:
        # --- bi-exponential fit ---
        # initial guesses (important!)
        amp0 = np.max(fit_data)
        # Guard against vanishing APCPMG signal
        if amp0 <= 0:
            raise RuntimeError("APCPMG: signal too small for bi-exponential fit")

        As0 = 0.6 * amp0
        Al0 = 0.4 * amp0

        Ts0 = 5.0
        Tl0 = 30.0
        p0 = [As0, Ts0, Al0, Tl0]

        bounds = (
            [0, 0.5, 0, 5.0],  # lower bounds
            [np.inf, 50, np.inf, 500]  # upper bounds
        )

        fitData2, _ = curve_fit(func2, fit_time, fit_data,
                                p0=p0, bounds=bounds)

        As, Ts, Al, Tl = fitData2

        # --- derived quantities ---
        A_tot = As + Al
        f_long = Al / A_tot

        print('Bi-exponential fit:')
        print(f'Long fraction: {f_long:.3f}')
        print(f'Short T2 = {Ts:.2f} ms')
        print(f'Long T2 = {Tl:.2f} ms')
        output_dict['bi'] = {'As': As, 'Ts': Ts, 'Al': Al, 'Tl': Tl, 'f_long': f_long}
        fit_curve_plot = func2(t_fit_plot, As, Ts, Al, Tl)
        # Signal vs rf time
        # --- interpolate experimental data onto fit axis for plotting ---
        exp_interp = np.interp(
            t_fit_plot,
            fit_time,
            fit_data,
            left=np.nan,
            right=np.nan
        )
        result1 = {
            'widget': 'curve',
            'xData': t_fit_plot,
            'yData': [exp_interp, fit_curve_plot],
            'xLabel': 'Echo time (ms)',
            'yLabel': 'Integrated echo amplitude (a.u.)',
            'title': 'Echo-integrated amplitudes (fit from echo 2, extrapolated to t = 0)',
            'legend': [
                'Experimental (echoes 2…end)',
                'Bi-exponential fit (extrapolated)'
            ],
            'row': 0,
            'col': 0
        }
    except:
        print('Bi-exponential fit failed.')

    result2 = {'widget': 'curve',
               'xData': t1_vector,
               'yData': [np.abs(data_echoes),np.real(data_echoes),np.imag(data_echoes)],
               'xLabel': 'Echo time (ms)',
               'yLabel': 'Echo amplitude (mV)',
               'title': 'Echo train',
               'legend': ['Experimental measurement','Real','Imag'],
               'row': 1,
               'col': 0}

    # Save results into rawData
    t1_vector = np.reshape(t1_vector, (-1, 1))
    data_echoes = np.reshape(data_echoes, (-1, 1))
    output_dict['signal_vs_time'] = np.concatenate((t1_vector, data_echoes), axis=1)
    t2_vector = np.reshape(t2_vector, (-1, 1))
    echo_integrals = np.reshape(echo_integrals, (-1, 1))
    output_dict['echo_amplitude_vs_time'] = np.concatenate((t2_vector, echo_integrals), axis=1)

    # Create self.output to run in iterative mode
    results = [result1, result2]

    return output_dict, results, dicom_meta_data