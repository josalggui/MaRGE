import controller.experiment_gui as ex
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import configs.hw_config as hw
import numpy as np
from scipy.optimize import curve_fit

class SpinLockingSweep(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(SpinLockingSweep, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='SpinLockingSweep', val='SpinLockingSweep')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=11.08, field='RF')

        self.addParameter(key='rfExAmp90', string='RFamp 90º pulse (a.u.)', val=0.5, field='RF')
        self.addParameter(key='rfExTime90', string='RFtime 90º pulse (us)', val=20.0, field='RF')

        self.addParameter(key='SLrfExAmpInit', string='SL-RFamp initial (a.u.)', val=0.1, field='RF')
        self.addParameter(key='SLrfExAmpFinal', string='SL-RFamp final (a.u.)', val=0.1, field='RF')
        self.addParameter(key='SLRFampSteps', string='SL-RFamp steps', val=1, field='RF')

        self.addParameter(key='SLrfExTimeInit', string='SL-RFtime initial (us)', val=20.0, field='RF')
        self.addParameter(key='SLrfExTimeFinal', string='SL-RFtime final (us)', val=500.0, field='RF')
        self.addParameter(key='SLRFtimeSteps', string='SL-RFtime steps', val=50, field='RF', tip="Only enabled 1, 4 or 9")

        self.addParameter(key='rotaryEcho', string='Rotary Echo', val=0, field='RF')

        self.addParameter(key='deadTime', string='RF dead time (us)', val=200.0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=300., field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=1.0, field='SEQ')
        self.addParameter(key='nReadout', string='Number of points', val=600, field='IM')
        self.addParameter(key='timePlot', string='Point to plot', val=50, field='IM')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-0, -0, 0], field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='rfExPhase', string='RF Exc Phase', val=0, field='RF')

    def sequenceInfo(self):

        print("SpinLockingSweep\n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        SLRFampSteps = self.mapVals['SLRFampSteps']
        SLRFtimeSteps = self.mapVals['SLRFtimeSteps']
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        return (SLRFampSteps * SLRFtimeSteps * repetitionTime * nScans / 60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False  # Starts the gpa

        # Create input parameters
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq']  # MHz

        rfExAmp90 = self.mapVals['rfExAmp90']
        rfExTime90 = self.mapVals['rfExTime90']  # us

        SLrfExAmpIni = self.mapVals['SLrfExAmpInit']
        SLrfExAmpEnd = self.mapVals['SLrfExAmpFinal']
        SLnExAmp = self.mapVals['SLRFampSteps']

        SLrfExTimeIni = self.mapVals['SLrfExTimeInit']  # us
        SLrfExTimeEnd = self.mapVals['SLrfExTimeFinal']  # us
        SLnExTime = self.mapVals['SLRFtimeSteps']

        deadTime = self.mapVals['deadTime']  # us
        repetitionTime = self.mapVals['repetitionTime'] * 1e3  # us
        acqTime = self.mapVals['acqTime'] * 1e3  # us
        nReadout = self.mapVals['nReadout']
        shimming = np.array(self.mapVals['shimming']) * 1e-4
        txChannel = self.mapVals['txChannel']
        rxChannel = self.mapVals['rxChannel']
        rfExPhase = self.mapVals['rfExPhase']

        rotaryEcho = self.mapVals['rotaryEcho']

        if SLnExAmp not in [1, 4, 9]:
            print("SL-RFamp steps only can be 1, 4 or 9")
            return(0)

        if SLnExAmp == 1:
            SLrfExAmpEnd = SLrfExAmpIni

        # RF PULSES
        rfExPhase = rfExPhase * np.pi / 180
        SLrfExAmp = np.linspace(SLrfExAmpIni, SLrfExAmpEnd, SLnExAmp, endpoint=True)
        SLrfExTime = np.linspace(SLrfExTimeIni, SLrfExTimeEnd, SLnExTime, endpoint=True)
        timeSLvector= SLrfExTime*1e-3
        timeFID = np.linspace(0, acqTime, nReadout, endpoint=True)*1e3 #ms
        self.mapVals['SLrfExTime'] = SLrfExTime
        self.mapVals['timeFID'] = timeFID
        tStart = float(20)

        def createSequence():
            # Shimming
            self.iniSequence(tStart, shimming)  # shimming is turned on 20 us after experiment beginning
            tTx = tStart

            for scan in range(nScans):
                for indexAmp in range(SLnExAmp):
                    for indexExTime in range(SLnExTime):
                        # Excitation pulse
                        tTx = tTx + repetitionTime

                        if rotaryEcho == 0:
                            self.SLblock(tTx, rfExTime90, SLrfExTime[indexExTime], rfExAmp90, SLrfExAmp[indexAmp], rfExPhase, channel=txChannel)
                        if rotaryEcho == 1:
                            self.SLblockWithRotaryEcho(tTx, rfExTime90, SLrfExTime[indexExTime], rfExAmp90, SLrfExAmp[indexAmp], rfExPhase, channel=txChannel)

                        # Rx gate
                        tRx = tTx + hw.blkTime + rfExTime90 + SLrfExTime[indexExTime] + deadTime
                        self.rxGateSync(tRx, acqTime, channel=rxChannel)
            self.endSequence(tStart + repetitionTime * SLnExAmp * (SLnExTime + 1) * nScans)

        # Initialize the experiment
        bw = nReadout / acqTime  # MHz
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.getSamplingRate()
        bw = 1 / samplingPeriod
        acqTime = nReadout / bw  # us

        createSequence()
        if self.floDict2Exp():
            print("Sequence waveforms loaded successfully")
            pass
        else:
            print("ERROR: sequence waveforms out of hardware bounds")
            return False

        if not plotSeq:
            # Run the experiment and get data
            rxd, msgs = self.expt.run()

            # Decimate the signal
            dataFull = self.decimate(rxd['rx%i' % rxChannel], nScans * SLnExAmp * SLnExTime)

            # Average data
            data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
            self.mapVals['dataRaw'] = data
            print('End')
            matrixData = np.reshape(data, (SLnExAmp, SLnExTime, nReadout))
            data0 = matrixData[:,:,0]

            monoexp = lambda t, S0, T1rho: S0 * np.exp(-t / T1rho)
            T1rho_values = np.zeros(SLnExAmp)
            for i in range(SLnExAmp):
                signal = np.abs(data0[i, :])
                signal = signal / np.max(signal)
                p0 = [1.0, 50]

                try:
                    popt, _ = curve_fit(monoexp, timeSLvector, signal, p0=p0, maxfev=10000)
                    S0_fit, T1rho_fit = popt
                    T1rho_values[i] = T1rho_fit
                except RuntimeError:
                    T1rho_values[i] = np.nan

            print("T1rho estimados (ms):", T1rho_values)

            self.mapVals['matrixData'] = matrixData
            self.mapVals['timeFID'] = timeFID
            self.mapVals['T1rho_values'] = T1rho_values
            self.mapVals['SLrfExAmp'] = SLrfExAmp
        self.expt.__del__()

        return True

    def sequenceAnalysis(self, obj=''):
        matrixData = self.mapVals['matrixData']
        t0 = self.mapVals['timePlot']
        timeSLVector  = self.mapVals['SLrfExTime']*1e-3
        T1rho_values = self.mapVals['T1rho_values']
        SLrfExAmp = self.mapVals['SLrfExAmp']
        t0 = self.mapVals['timePlot']

        if self.mapVals['SLRFampSteps'] == 9:
            n = 0
            result1 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 0,
                        'col': 0}
            n = 1
            result2 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 0,
                        'col': 1}
            n = 2
            result3 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 0,
                        'col': 2}
            n = 3
            result4 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 1,
                        'col': 0}
            n = 4
            result5 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 1,
                        'col': 1}
            n = 5
            result6 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 1,
                        'col': 2}
            n = 6
            result7 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 2,
                        'col': 0}
            n = 7
            result8 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 2,
                        'col': 1}
            n = 8
            result9 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 2,
                        'col': 2}

            # create self.out to run in iterative mode
            self.output = [result1, result2, result3, result4, result5, result6, result7, result8, result9]
            self.saveRawData()
            return self.output

        if self.mapVals['SLRFampSteps'] == 4:
            # Add time signal to the layout
            n = 0
            result1 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 0,
                        'col': 0}
            n = 1
            result2 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 0,
                        'col': 1}
            n = 2
            result3 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 1,
                        'col': 0}
            n = 3
            result4 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                        'row': 1,
                        'col': 1}

            # create self.out to run in iterative mode
            self.output = [result1, result2, result3, result4]
            self.saveRawData()
            return self.output

        if self.mapVals['SLRFampSteps'] == 1:
            n = 0
            result1 = {'widget': 'curve',
                       'xData': timeSLVector,
                       'yData': [np.abs(matrixData[n, :, t0])],
                       'xLabel': 'SpinLocking duration (ms)',
                       'yLabel': 'NMR signal (mV)',
                       'title': f'RFamp={SLrfExAmp[n]:.2f} a.u., T1rho={T1rho_values[n]:.2f} ms',
                       'legend': ['abs'],
                       'row': 0,
                       'col': 0}
            result2 = {}
            result2['widget'] = 'image'
            result2['data'] = np.abs(matrixData)
            result2['xLabel'] = "SpinLocking duration (ms)"
            result2['yLabel'] = "FID acquisition time (ms)"
            result2['title'] = "FID after Spin-Locking"
            result2['row'] = 0
            result2['col'] = 1

            # create self.out to run in iterative mode
            self.output = [result2,result1]
            self.saveRawData()
            return self.output
        
if __name__ == '__main__':
    seq = SpinLockingSweep()
    seq.sequenceRun()
    seq.sequenceAnalysis(obj='Standalone')
