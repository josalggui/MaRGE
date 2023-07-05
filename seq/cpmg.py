"""
@author: José Miguel Algarín Guisado
@modifield: T. Guallart Naval, february 28th 2022
MRILAB @ I3M
"""

import experiment as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw
import configs.units as units
from scipy.optimize import curve_fit


class TSE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(TSE, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='TSEInfo', val='TSE')
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
        self.addParameter(key='echoObservation', string='Echo Observation', val=1, field='OTH')
        self.addParameter(key='phase_mode', string='Phase mode', val='CPMG', tip='CP, CPMG, APCP, APCPMG', field='SEQ')

    def sequenceInfo(self):
        print(" ")
        print("CPMG")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs an echo train with CPMG")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        return(repetitionTime*nScans/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False  # Starts the gpa
        self.demo = demo

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
                    elif self.phase_mode == 'APCPMG':
                        phase = (-1)**echoIndex*np.pi/2
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
        print("\nAcquisition bandwidth = %0.1f kHz"%(bw*1e3))

        # Save instructions into MaRCoS if not a demo
        if not self.demo:
            if self.floDict2Exp():
                print("\nSequence waveforms loaded successfully")
                pass
            else:
                print("\nERROR: sequence waveforms out of hardware bounds")
                return False

        # Execute the experiment if not plot
        if not plotSeq:
            if not self.demo:
                rxd, msgs = self.expt.run()
                rxd['rx0'] *= hw.adcFactor
            else:
                rxd = {}
                rxd['rx0'] = np.random.randn(acq_points * hw.oversamplingFactor) + 1j * \
                             np.random.randn(acq_points * hw.oversamplingFactor)
            self.mapVals['dataFull'] = rxd['rx0']
            data = sig.decimate(rxd['rx0'], hw.oversamplingFactor, ftype='fir', zero_phase=True)
            data = np.average(np.reshape(data, (self.nScans, -1)), axis=0)
            self.mapVals['data'] = data

        if not self.demo: self.expt.__del__()

        return True

    def sequenceAnalysis(self, obj=''):
        data = self.mapVals['data']

        # Prepare data for full data plot
        data_echoes = np.reshape(data, (self.etl, -1))
        data_echoes[:, 0] = 0.
        data_echoes[:, -1] = 0.
        data_echoes = np.reshape(data, -1)
        t0 = np.linspace(self.echoSpacing - self.acqTime / 2, self.echoSpacing + self.acqTime / 2, self.nPoints)*1e-3 # ms
        t1_vector = t0
        for echo_index in range(self.etl - 1):
            t1_vector = np.concatenate((t1_vector, t0 + self.echoSpacing*1e-3 * (echo_index + 1)), axis=0)

        # Prepare data for echo amplitude vs echo time
        data = np.reshape(data, (self.etl, -1))
        data = data[:, int(self.nPoints / 2)]
        t2_vector = np.linspace(self.echoSpacing, self.echoSpacing * self.etl, num=self.etl, endpoint=True)*1e-3 # ms

        # Save point here to sweep class
        self.mapVals['sampledPoint'] = data[0]

        # Functions for fitting
        def func1(x, m, t2):
            return m*np.exp(-x/t2)

        def func2(x, ma, t2a, mb, t2b):
            return ma*np.exp(-x/t2a)+mb*np.exp(-x/t2b)

        def func3(x, ma, t2a, mb, t2b, mc, t2c):
            return ma*np.exp(-x/t2a)+mb*np.exp(-x/t2b)+mc*np.exp(-x/t2c)

        # Fitting to functions
        # # For 1 component
        # fitData1, xxx = curve_fit(func1, results[0],  results[1])
        # print('For one component:')
        # print('mA', round(fitData1[0], 1))
        # print('T2', round(fitData1[1]), ' ms')
        # self.mapVals['T21'] = fitData1[1]
        # self.mapVals['M1'] = fitData1[0]
        #
        # # For 2 components
        # fitData2, xxx = curve_fit(func2, results[0],  results[1])
        # print('For two components:')
        # print('Ma', round(fitData2[0], 1))
        # print('Mb', round(fitData2[2], 1))
        # print('T2a', round(fitData2[1]), ' ms')
        # print('T2b', round(fitData2[3]), ' ms')
        # self.mapVals['T22'] = [fitData2[1], fitData2[3]]
        # self.mapVals['M2'] = [fitData2[0], fitData2[2]]
        #
        # # For 3 components
        # fitData3, xxx = curve_fit(func3, results[0],  results[1])
        # print('For three components:')
        # print('Ma', round(fitData3[0], 1), ' ms')
        # print('Mb', round(fitData3[2], 1), ' ms')
        # print('Mc', round(fitData3[4], 1), ' ms')
        # print('T2a', round(fitData3[1]), ' ms')
        # print('T2b', round(fitData3[3]), ' ms')
        # print('T2c', round(fitData3[5]), ' ms')
        # self.mapVals['T23'] = [fitData3[1], fitData3[3], fitData3[5]]
        # self.mapVals['M3'] = [fitData3[0], fitData3[2], fitData3[4]]


        # Signal vs rf time
        result1 = {'widget': 'curve',
                   'xData': t2_vector,
                   'yData': [np.abs(data)],
                   'xLabel': 'Echo time (ms)',
                   'yLabel': 'Echo amplitude (mV)',
                   'title': 'Echo amplitude VS Echo time',
                   'legend': ['Experimental at echo time', 'Experimental at maximum'],
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
        self.mapVals['signal_vs_time'] = np.concatenate((t1_vector, data_echoes), axis=1)
        t2_vector = np.reshape(t2_vector, (-1, 1))
        data = np.reshape(data, (-1, 1))
        self.mapVals['echo_amplitude_vs_time'] = np.concatenate((t2_vector, data), axis=1)

        self.saveRawData()

        # create self.out to run in iterative mode
        self.out = [result1, result2]
        return self.out