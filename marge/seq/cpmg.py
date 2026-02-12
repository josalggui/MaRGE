"""
@author: José Miguel Algarín Guisado
@modifield: T. Guallart Naval, february 28th 2022
MRILAB @ I3M
@modified with ChatGPT: Bernhard Blümich, January 2026
"""

import marge.marcos.marcos_client.experiment as ex
import numpy as np
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import marge.configs.hw_config as hw
import marge.configs.units as units
from scipy.optimize import curve_fit


class TSE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(TSE, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='TSEInfo', val='TSE')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, units=units.MHz, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, units=units.ms, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., units=units.ms, field='SEQ')
        self.addParameter(key='nPoints', string='Number of acquired points', val=60, field='IM')
        self.addParameter(key='etl', string='Echo train length', val=50, field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=2.0, units=units.ms, field='SEQ')
        self.addParameter(key='shimming', string='shimming', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='phase_mode', string='Phase mode', val='CPMG', tip='CP, CPMG, APCP, APCPMG', field='SEQ')

    def sequenceInfo(self):

        print("CPMG")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs an echo train with CPMG\n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        return(repetitionTime*nScans/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False  # Starts the gpa
        self.demo = demo

        # Check that self.phase_mode is once of the good values
        phase_modes = ['CP', 'CPMG', 'APCP', 'APCPMG']
        if not self.phase_mode in phase_modes:
            print('ERROR: Unexpected phase mode. Please select one of possible modes: CP, CPMG, APCP, APCPMG.')
            return False

        # I do not understand why I cannot create the input parameters automatically
        def createSequence():
            acq_points = 0

            # Initialize time
            t0 = 20
            tEx = 20e3

            # self.shimming
            self.iniSequence(t0, self.shimming)

            for scan in range(self.nScans):

                # Excitation pulse
                t0 = tEx - hw.blkTime - self.rfExTime / 2 + self.repetitionTime*scan
                self.rfRecPulse(t0, self.rfExTime, self.rfExAmp, 0)

                # Echo train
                for echoIndex in range(self.etl):
                    tEcho = tEx + (echoIndex + 1) * self.echoSpacing + self.repetitionTime*scan

                    # Refocusing pulse
                    t0 = tEcho - self.echoSpacing / 2 - hw.blkTime - self.rfReTime / 2
                    phase = 0
                    if self.phase_mode == 'CP':
                        phase = 0.0
                    elif self.phase_mode == 'CPMG':
                        phase = np.pi/2
                    elif self.phase_mode == 'APCP':
                        phase = ((-1)**(echoIndex)+1)*np.pi/2
  ###                  elif self.phase_mode == 'APCPMG':
  ###                      phase = (-1)**echoIndex*np.pi/2
                    elif self.phase_mode == 'APCPMG':                               ###
                        # Suh–Borsa–Torgeson APCPMG:                                ###
                        # +y, +y, -y, -y, ...                                       ###
                        phase = ((-1) ** (echoIndex // 2)) * np.pi/2                ###
                    self.rfRecPulse(t0, self.rfReTime, self.rfReAmp, rfPhase=phase)

                    # Rx gate
                    t0 = tEcho - self.acqTime / 2
                    self.rxGate(t0, self.acqTime)
                    acq_points += self.nPoints

            self.endSequence(self.repetitionTime*self.nScans)

            return acq_points

        # Time variables in us
        self.echoSpacing *= 1e6
        self.repetitionTime *= 1e6
        self.acqTime *= 1e6
        self.rfExTime *= 1e6
        self.rfReTime *= 1e6

        # Initialize the experiment
        bw = self.nPoints / self.acqTime * hw.oversamplingFactor  # MHz
        samplingPeriod = 1 / bw  # us
        if not self.demo:
            # Create experiment object and update parameters
            self.expt = ex.Experiment(lo_freq=self.larmorFreq*1e-6, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
            samplingPeriod = self.expt.get_rx_ts()[0] # us
            bw = 1 / samplingPeriod / hw.oversamplingFactor  # MHz
            self.acqTime = self.nPoints / bw  # us

        # Run the createSequence method
        acq_points = createSequence()
        self.mapVals['bw'] = bw
        print("Acquisition bandwidth = %0.1f kHz"%(bw*1e3))

        # Save instructions into MaRCoS if not a demo
        if not self.demo:
            if self.floDict2Exp():
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("ERROR: sequence waveforms out of hardware bounds")
                return False

        # Execute the experiment if not plot
        if not plotSeq:
            if not self.demo:
                rxd, msgs = self.expt.run()
                rxd['rx0'] *= hw.adcFactor
            else:
                rxd = {}
                rxd['rx0'] = self.mySignal()
            self.mapVals['dataFull'] = rxd['rx0']
            data = sig.decimate(rxd['rx0'], hw.oversamplingFactor, ftype='fir', zero_phase=True)
            data = np.average(np.reshape(data, (self.nScans, -1)), axis=0)
            self.mapVals['data'] = data

        if not self.demo: self.expt.__del__()

        return True

    # def sequenceAnalysis(self, mode='Standalone'):
    #     data = self.mapVals['data']
    #
    #     # --- reshape raw data into echoes ---
    #     data_2d = np.reshape(data, (self.etl, -1))
    #
    #     # --- global phase correction using first echo ---
    #     abs0 = np.abs(data_2d[0])
    #     center0 = np.argmax(abs0)
    #     w_phase = 3
    #
    #     ref = np.sum(data_2d[0, center0-w_phase:center0+w_phase+1])
    #     phi = np.angle(ref)
    #
    #     data_2d *= np.exp(-1j * phi)
    #
    #     # --- echo-center tracking + symmetric integration ---
    #     w_int = 3        # half-width of integration window (points)
    #     echo_integrals = np.zeros(self.etl)
    #     echo_centers = np.zeros(self.etl, dtype=int)
    #
    #     for k in range(self.etl):
    #         # locate echo center from magnitude
    #         mag = np.abs(data_2d[k])
    #         c = np.argmax(mag)
    #         echo_centers[k] = c
    #
    #         # guard against boundary issues
    #         c0 = max(c - w_int, 0)
    #         c1 = min(c + w_int + 1, self.nPoints)
    #
    #         echo_integrals[k] = np.sum(
    #         np.real(data_2d[k, c0:c1])
    #         )
    #     # --- enforce consistent echo sign for APCPMG ---
    #     if self.phase_mode == 'APCPMG':
    #         # Use the first *reliable* echo as sign reference
    #         ref_sign = np.sign(echo_integrals[1])  # echo 2 (echo 1 discarded anyway)
    #         ref_sign = 1.0 if ref_sign == 0 else ref_sign
    #
    #         for k in range(self.etl):
    #             if np.sign(echo_integrals[k]) != ref_sign:
    #                 echo_integrals[k] *= -1.0
    #
    #     # --- prepare echo train for plotting (unchanged style) ---
    #     data_echoes = data_2d.copy()
    #     data_echoes[:, 0] = 0.0
    #     data_echoes[:, -1] = 0.0
    #     data_echoes = np.reshape(data_echoes, -1)
    #
    #     t0 = np.linspace(self.echoSpacing - self.acqTime / 2,
    #              self.echoSpacing + self.acqTime / 2,
    #              self.nPoints) * 1e-3  # ms
    #
    #     t1_vector = t0
    #     for echo_index in range(self.etl - 1):
    #         t1_vector = np.concatenate(
    #             (t1_vector, t0 + self.echoSpacing * 1e-3 * (echo_index + 1)),
    #             axis=0
    #         )
    #
    #     # --- time axis: echo centers relative to excitation ---
    #     t2_vector = (np.arange(1, self.etl + 1) * self.echoSpacing) * 1e-3  # ms
    #
    #     # --- discard first echo in the fit ---
    #     fit_data = echo_integrals[1:]
    #     fit_time = t2_vector[1:]
    #
    #     # --- remove non-finite points before fitting ---
    #     mask = np.isfinite(fit_data) & np.isfinite(fit_time)
    #     fit_data = fit_data[mask]
    #     fit_time = fit_time[mask]
    #
    #     # Save sampled point (still echo 1 if needed elsewhere)
    #     self.mapVals['sampledPoint'] = echo_integrals[1]  # first echo used in fit
    #
    #     # Functions for fitting
    #     def func1(x, m, t2):
    #         return m*np.exp(-x/t2)
    #     def func2(x, As, Ts, Al, Tl):
    #         return As*np.exp(-x/Ts) + Al*np.exp(-x/Tl)
    #
    #     try:
    #         # Fitting to functions
    #         # --- mono-exponential fit ---
    #         fitData1, _ = curve_fit(func1, fit_time, fit_data,
    #                                 p0=[fit_data[0], 10])
    #         fitting1 = func1(fit_time, fitData1[0], fitData1[1])
    #         M0 = fitData1[0]
    #         T2 = fitData1[1]
    #         print('Mono-exponential fit:')
    #         print('T2 (ms):', round(T2, 2))
    #         self.mapVals['mono'] = {'M0': M0, 'T2': T2}
    #
    #         # --- fitted curve extrapolated back to t = 0 ---
    #         t_fit_plot = np.linspace(
    #             0.0,
    #             t2_vector[-1],
    #             200
    #         )  # ms, smooth curve
    #         fit_curve_plot = func1(t_fit_plot, M0, T2)
    #         # Signal vs rf time
    #         # --- interpolate experimental data onto fit axis for plotting ---
    #         exp_interp = np.interp(
    #             t_fit_plot,
    #             fit_time,
    #             fit_data,
    #             left=np.nan,
    #             right=np.nan
    #         )
    #         result1 = {
    #             'widget': 'curve',
    #             'xData': t_fit_plot,
    #             'yData': [exp_interp, fit_curve_plot],
    #             'xLabel': 'Echo time (ms)',
    #             'yLabel': 'Integrated echo amplitude (a.u.)',
    #             'title': 'Echo-integrated amplitudes (fit from echo 2, extrapolated to t = 0)',
    #             'legend': [
    #                 'Experimental (echoes 2…end)',
    #                 'Mono-exponential fit (extrapolated)'
    #             ],
    #             'row': 0,
    #             'col': 0
    #         }
    #     except:
    #         print('Mono-exponential fit failed.')
    #
    #         # --- fitted curve extrapolated back to t = 0 ---
    #         t_fit_plot = np.linspace(
    #             0.0,
    #             t2_vector[-1],
    #             200
    #         )  # ms, smooth curve
    #         # --- interpolate experimental data onto fit axis for plotting ---
    #         exp_interp = np.interp(
    #             t_fit_plot,
    #             fit_time,
    #             fit_data,
    #             left=np.nan,
    #             right=np.nan
    #         )
    #         result1 = {
    #             'widget': 'curve',
    #             'xData': t_fit_plot,
    #             'yData': [exp_interp],
    #             'xLabel': 'Echo time (ms)',
    #             'yLabel': 'Integrated echo amplitude (a.u.)',
    #             'title': 'Echo-integrated amplitudes (fit from echo 2, extrapolated to t = 0)',
    #             'legend': [
    #                 'Experimental (echoes 2…end)'
    #             ],
    #             'row': 0,
    #             'col': 0
    #         }
    #
    #     try:
    #         # --- bi-exponential fit ---
    #         # initial guesses (important!)
    #         amp0 = np.max(fit_data)
    #         # Guard against vanishing APCPMG signal
    #         if amp0 <= 0:
    #             raise RuntimeError("APCPMG: signal too small for bi-exponential fit")
    #
    #         As0 = 0.6 * amp0
    #         Al0 = 0.4 * amp0
    #
    #         Ts0 = 5.0
    #         Tl0 = 30.0
    #         p0 = [As0, Ts0, Al0, Tl0]
    #
    #         bounds = (
    #             [0, 0.5, 0, 5.0],  # lower bounds
    #             [np.inf, 50, np.inf, 500]  # upper bounds
    #         )
    #
    #         fitData2, _ = curve_fit(func2, fit_time, fit_data,
    #             p0=p0, bounds=bounds)
    #
    #         As, Ts, Al, Tl = fitData2
    #
    #         # --- derived quantities ---
    #         A_tot = As + Al
    #         f_long = Al / A_tot
    #
    #         print('Bi-exponential fit:')
    #         print('Long fraction:', round(f_long, 3))
    #         print('T2 short (ms):', round(Ts, 2))
    #         print('T2 long (ms):', round(Tl, 2))
    #         self.mapVals['bi'] = {'As': As, 'Ts': Ts, 'Al': Al, 'Tl': Tl, 'f_long': f_long}
    #         fit_curve_plot = func2(t_fit_plot, As, Ts, Al, Tl)
    #         # Signal vs rf time
    #         # --- interpolate experimental data onto fit axis for plotting ---
    #         exp_interp = np.interp(
    #             t_fit_plot,
    #             fit_time,
    #             fit_data,
    #             left=np.nan,
    #             right=np.nan
    #         )
    #         result1 = {
    #             'widget': 'curve',
    #             'xData': t_fit_plot,
    #             'yData': [exp_interp, fit_curve_plot],
    #             'xLabel': 'Echo time (ms)',
    #             'yLabel': 'Integrated echo amplitude (a.u.)',
    #             'title': 'Echo-integrated amplitudes (fit from echo 2, extrapolated to t = 0)',
    #             'legend': [
    #                 'Experimental (echoes 2…end)',
    #                 'Bi-exponential fit (extrapolated)'
    #             ],
    #             'row': 0,
    #             'col': 0
    #         }
    #     except:
    #         print('Bi-exponential fit failed.')
    #
    #     result2 = {'widget': 'curve',
    #                'xData': t1_vector,
    #                'yData': [np.abs(data_echoes),np.real(data_echoes),np.imag(data_echoes)],
    #                'xLabel': 'Echo time (ms)',
    #                'yLabel': 'Echo amplitude (mV)',
    #                'title': 'Echo train',
    #                'legend': ['Experimental measurement','Real','Imag'],
    #                'row': 1,
    #                'col': 0}
    #
    #     # Save results into rawData
    #     t1_vector = np.reshape(t1_vector, (-1, 1))
    #     data_echoes = np.reshape(data_echoes, (-1, 1))
    #     self.mapVals['signal_vs_time'] = np.concatenate((t1_vector, data_echoes), axis=1)
    #     t2_vector = np.reshape(t2_vector, (-1, 1))
    #     echo_integrals = np.reshape(echo_integrals, (-1, 1))
    #     self.mapVals['echo_amplitude_vs_time'] = np.concatenate(
    #         (t2_vector, echo_integrals), axis=1)
    #
    #     # Create self.output to run in iterative mode
    #     self.output = [result1, result2]
    #
    #     # Save rawData
    #     self.saveRawData()
    #
    #     return self.output

    def mySignal(self):
        # Get inputs
        te = self.mapVals['echoSpacing']
        acq_time = self.mapVals['acqTime']
        n_points = self.mapVals['nPoints'] * hw.oversamplingFactor
        t2 = 100.0 # ms
        t2_star = 10.0 # ms

        # Define gaussian function
        def gaussian(a, t, mu, sig):
            return a*np.exp(-np.power(t - mu, 2.) / (2 * np.power(sig, 2.)))

        # Generate signal vector
        t0 = np.linspace(te - acq_time / 2, te + acq_time / 2, n_points)  # ms
        t1_vector = t0
        signal = gaussian(np.exp(-te/t2), t0, te, t2_star)
        for echo_index in range(self.etl - 1):
            t0 += te
            sig_prov = gaussian(np.exp(-te*(echo_index+2)/t2), t0, te*(echo_index+2), t2_star)

            t1_vector = np.concatenate((t1_vector, t0), axis=0)
            signal = np.concatenate((signal, sig_prov), axis=0)

        if self.nScans > 1:
            signal0 = signal.copy()
            for repetition in range(self.nScans-1):
                signal0 = np.concatenate((signal0, signal), axis=0)
            signal = signal0

        # Add noise
        signal = signal + (np.random.randn(np.size(signal)) + 1j * np.random.randn(np.size(signal))) * 0.01

        return signal

if __name__ == '__main__':
    seq = TSE()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True)
    seq.sequenceAnalysis(mode='Standalone')