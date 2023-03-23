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
from scipy.optimize import curve_fit


class CPMG(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(CPMG, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='CPMGInfo', val='CPMG')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ')
        self.addParameter(key='nPoints', string='nPoints', val=60, field='IM')
        self.addParameter(key='etl', string='Echo train length', val=50, field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=2.0, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-12.5,-12.5,7.5], field='OTH')
        self.addParameter(key='echoObservation', string='Echo Observation', val=1, field='OTH')

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

    def sequenceRun(self, plotSeq=0):
        init_gpa = False  # Starts the gpa
        demo = False

        # Create the inputs automatically. For some reason it only works if there is a few code later...
        # for key in self.mapKeys:
        #     if type(self.mapVals[key])==list:
        #         locals()[key] = np.array(self.mapVals[key])
        #     else:
        #         locals()[key] = self.mapVals[key]

        # I do not understand why I cannot create the input parameters automatically
        seqName = self.mapVals['seqName']
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq']
        rfExAmp = self.mapVals['rfExAmp']
        rfExTime = self.mapVals['rfExTime']
        rfReAmp = self.mapVals['rfReAmp']
        rfReTime = self.mapVals['rfReTime']
        echoSpacing = self.mapVals['echoSpacing']
        repetitionTime = self.mapVals['repetitionTime']
        nPoints = self.mapVals['nPoints']
        etl = self.mapVals['etl']
        acqTime = self.mapVals['acqTime']
        shimming = np.array(self.mapVals['shimming'])
        echoObservation = self.mapVals['echoObservation']

        def createSequence():
            # Initialize time
            t0 = 20
            tEx = 20e3

            # Shimming
            self.iniSequence(t0, shimming)

            for scan in range(nScans):

                # Excitation pulse
                t0 = tEx - hw.blkTime - rfExTime / 2 + repetitionTime*scan
                self.rfRecPulse(t0, rfExTime, rfExAmp, 0)

                # Echo train
                for echoIndex in range(etl):
                    tEcho = tEx + (echoIndex + 1) * echoSpacing + repetitionTime*scan

                    # Refocusing pulse
                    t0 = tEcho - echoSpacing / 2 - hw.blkTime - rfReTime / 2
                    self.rfRecPulse(t0, rfReTime, rfReAmp, np.pi / 2)

                    # Rx gate
                    t0 = tEcho - acqTime / 2
                    self.rxGate(t0, acqTime)

            self.endSequence(repetitionTime*nScans)

        # Time variables in us
        echoSpacing = echoSpacing * 1e3
        repetitionTime = repetitionTime * 1e3
        acqTime = acqTime * 1e3
        shimming = np.array(shimming) * 1e-4

        # Initialize the experiment
        bw = nPoints / acqTime * hw.oversamplingFactor  # MHz
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1 / samplingPeriod / hw.oversamplingFactor  # MHz
        acqTime = nPoints / bw  # us
        self.mapVals['bw'] = bw
        createSequence()
        if self.floDict2Exp():
            pass
        else:
            return 0

        if plotSeq == 1:
            self.expt.__del__()
        elif plotSeq == 0:
            # Run the experiment and get data
            print('Runing...')
            rxd, msgs = self.expt.run()
            print(msgs)
            self.mapVals['dataFull'] = rxd['rx0'] * hw.adcFactor
            data = sig.decimate(rxd['rx0'] * hw.adcFactor, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            data = np.average(np.reshape(data, (nScans, -1)), axis=0)
            self.mapVals['data'] = data
            self.expt.__del__()

            data = np.reshape(data, (etl, -1))
            data = np.abs(data[:, int(nPoints/2)])
            echoTimeVector = np.linspace(echoSpacing, echoSpacing * etl, num=etl, endpoint=True)
            self.results = [echoTimeVector, data]
        return 0

    def sequenceAnalysis(self, obj=''):
        data = np.abs(self.mapVals['data'])
        etl = self.mapVals['etl']
        data = np.reshape(data, (etl, -1))
        results = self.results
        nPoints = self.mapVals['nPoints']
        echo1Amp = self.mapVals['data'][int(nPoints/2)]
        self.mapVals['sampledPoint'] = echo1Amp  # Save point here to sweep class

        # Functions for fitting
        def func1(x, m, t2):
            return m*np.exp(-x/t2)

        def func2(x, ma, t2a, mb, t2b):
            return ma*np.exp(-x/t2a)+mb*np.exp(-x/t2b)

        def func3(x, ma, t2a, mb, t2b, mc, t2c):
            return ma*np.exp(-x/t2a)+mb*np.exp(-x/t2b)+mc*np.exp(-x/t2c)

        # Fitting to functions
        # For 1 component
        fitData1, xxx = curve_fit(func1, results[0],  results[1])
        print('For one component:')
        print('mA', round(fitData1[0], 1))
        print('T2', round(fitData1[1]), ' ms')
        self.mapVals['T21'] = fitData1[1]
        self.mapVals['M1'] = fitData1[0]

        # For 2 components
        fitData2, xxx = curve_fit(func2, results[0],  results[1])
        print('For two components:')
        print('Ma', round(fitData2[0], 1))
        print('Mb', round(fitData2[2], 1))
        print('T2a', round(fitData2[1]), ' ms')
        print('T2b', round(fitData2[3]), ' ms')
        self.mapVals['T22'] = [fitData2[1], fitData2[3]]
        self.mapVals['M2'] = [fitData2[0], fitData2[2]]

        # For 3 components
        fitData3, xxx = curve_fit(func3, results[0],  results[1])
        print('For three components:')
        print('Ma', round(fitData3[0], 1), ' ms')
        print('Mb', round(fitData3[2], 1), ' ms')
        print('Mc', round(fitData3[4], 1), ' ms')
        print('T2a', round(fitData3[1]), ' ms')
        print('T2b', round(fitData3[3]), ' ms')
        print('T2c', round(fitData3[5]), ' ms')
        self.mapVals['T23'] = [fitData3[1], fitData3[3], fitData3[5]]
        self.mapVals['M3'] = [fitData3[0], fitData3[2], fitData3[4]]

        self.saveRawData()

        # Signal vs rf time
        result1 = {'widget': 'curve',
                   'xData': results[0]*1e-3,
                   'yData': [results[1], func1(results[0], *fitData1), func2(results[0], *fitData2)],
                   'xLabel': 'Echo time (ms)',
                   'yLabel': 'Echo amplitude (mV)',
                   'title': '',
                   'legend': ['Experimental', 'Fitting 1 component', 'Fitting 2 components'],
                   'row': 0,
                   'col': 0}

        # 2D image
        result2 = {'widget': 'image',
                   'data': data,
                   'xLabel': "Echo index",
                   'yLabel': "Acquired point",
                   'title': "Acquired echoes",
                   'row': 1,
                   'col': 0}


        # create self.out to run in iterative mode
        self.out = [result1, result2]
        return self.out