
import marge.controller.experiment_gui as ex
import numpy as np
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import marge.configs.hw_config as hw
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline



class B1calibration(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(B1calibration, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='B1calibration', val='B1calibration')
        self.addParameter(key='toMaRGE', val=False)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmpInit', string='RF initial amplitude (a.u.)', val=0.2, field='RF')
        self.addParameter(key='rfExAmpFinal', string='RF final amplitude (a.u.)', val=1.0, field='RF')
        self.addParameter(key='RFampSteps', string='RF amp steps', val=9, field='RF')
        self.addParameter(key='rfExTimeInit', string='RF initial length (us)', val=1.0, field='RF')
        self.addParameter(key='rfExTimeFinal', string='RF final length (us)', val=1000.0, field='RF')
        self.addParameter(key='RFtimeSteps', string='RF time steps', val=100, field='RF')
        self.addParameter(key='deadTime', string='RF dead time (us)', val=100.0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=50., field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=1.0, field='SEQ')
        self.addParameter(key='nReadout', string='Number of points', val=100, field='IM')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='addRdPoints', string='addRdPoints', val=5, field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='rfExPhase', string='RF Exc Phase', val=0, field='RF')

    def sequenceInfo(self):
        
        print("B1calibration\n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        RFampSteps = self.mapVals['RFampSteps']
        RFtimeSteps = self.mapVals['RFtimeSteps']
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        return(RFampSteps*RFtimeSteps*repetitionTime*nScans/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False  # Starts the gpa

        # Create input parameters
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq']  # MHz
        rfExAmpIni = self.mapVals['rfExAmpInit']
        rfExAmpEnd = self.mapVals['rfExAmpFinal']
        nExAmp = self.mapVals['RFampSteps']
        rfExTimeIni = self.mapVals['rfExTimeInit']  # us
        rfExTimeEnd = self.mapVals['rfExTimeFinal']  # us
        nExTime  = self.mapVals['RFtimeSteps']
        deadTime = self.mapVals['deadTime']   # us
        repetitionTime = self.mapVals['repetitionTime'] * 1e3  # us
        acqTime = self.mapVals['acqTime'] * 1e3  # us
        nReadout = self.mapVals['nReadout']
        shimming = np.array(self.mapVals['shimming']) * 1e-4
        txChannel = self.mapVals['txChannel']
        rxChannel = self.mapVals['rxChannel']
        addRdPoints = self.mapVals['addRdPoints']
        rfExPhase = self.mapVals['rfExPhase']

        dataAll = []
        matrix = np.zeros((nExTime, nExAmp, nReadout+addRdPoints)) * np.exp(1j)
        t90 = np.zeros(nExAmp)
        B1 = np.zeros(nExAmp)

        # RF PULSES
        rfExPhase = rfExPhase * np.pi / 180
        rfAmp = np.linspace(rfExAmpIni, rfExAmpEnd, nExAmp, endpoint=True)
        rfExAmp = rfAmp * np.exp(1j * rfExPhase)
        rfExTime = np.linspace(rfExTimeIni, rfExTimeEnd, nExTime, endpoint=True)
        timeFID = np.linspace(0, acqTime, nReadout, endpoint=True)
        self.mapVals['rfExTime'] = rfExTime
        self.mapVals['timeFID'] = timeFID
        tStart = float(20)

        def createSequence():
            # Shimming
            self.iniSequence(tStart, shimming)  # shimming is turned on 20 us after experiment beginning
            tTx = tStart

            for scan in range(nScans):
                for indexAmp in range(nExAmp):
                    for indexExTime in range(nExTime):
                        # Excitation pulse
                        tTx = tTx + repetitionTime
                        self.rfRecPulse(tTx, rfExTime[indexExTime], rfExAmp[indexAmp], 0, channel=txChannel)
                        # Rx gate
                        tRx = tTx + hw.blkTime + rfExTime[indexExTime] + deadTime - addRdPoints / bwReal
                        self.rxGate(tRx, acqTimeReal + addRdPoints / bwReal, channel=rxChannel)
                        # self.rxGateSync(tRx, acqTimeReal, rxChannel=rxChannel)
                        # self.ttl(tRx - 100, acqTime, channel=1, rewrite=True)
                self.endSequence(repetitionTime * nExAmp * nExTime * nScans)

        # Initialize the experiment
        bw = nReadout / acqTime * hw.oversamplingFactor  # MHz
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriodReal = self.expt.get_rx_ts()[0]
        bwReal = 1 / samplingPeriodReal / hw.oversamplingFactor  # MHz
        acqTimeReal = nReadout / bwReal  # us
        self.mapVals['bwReal'] = bwReal  # MHz
        createSequence()
        if self.floDict2Exp():
            print("Sequence waveforms loaded successfully")
            pass
        else:
            print("ERROR: sequence waveforms out of hardware bounds")
            return False

        if plotSeq == 0:
            # Run the experiment and get data
            rxd, msgs = self.expt.run()
            self.expt.__del__()
            print('   End')
            dataFull = sig.decimate(rxd['rx0']*hw.adcFactor, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            dataFull = np.reshape(dataFull, (nScans, -1))
            data = np.average(dataFull, axis=0)
            nExAmp = self.mapVals['RFampSteps']
            nExTime = self.mapVals['RFtimeSteps']
            rfExTime = self.mapVals['rfExTime']
            timeFID = self.mapVals['timeFID']
            addRdPoints = self.mapVals['addRdPoints']
            matrix = np.reshape(data, (nExAmp, nExTime, -1))
            matrix = np.delete(matrix, np.s_[0:addRdPoints], axis=2)
            self.matrix = matrix

            # Fit to a interpolant spline to get accurate t90
            for indexAmp in range(nExAmp):
                data = matrix[indexAmp, :, 1]
                spl = InterpolatedUnivariateSpline(rfExTime, abs(data))
                spl.set_smoothing_factor(0.5)
                interpolatedFIDtime = np.linspace(rfExTimeIni, rfExTimeEnd, 5 * nExTime, endpoint=True)
                fitteddata = spl(interpolatedFIDtime)

                # Get the indices of maximum element in fitted data
                indexmax = np.argmax(fitteddata)
                t90[indexAmp] = interpolatedFIDtime[indexmax]
                B1[indexAmp] = ((np.pi / 2) / (2 * np.pi * hw.gammaB * t90[indexAmp] * 1e-6)) * 1e6

                # Plot results
                plt.figure(1)
                plt.subplot(3, 3, indexAmp + 1)
                plt.plot(rfExTime, abs(data), 'k-', rfExTime, data.real, 'r--', rfExTime, data.imag, 'b--')
                plt.xlabel('t(us)', fontsize=8)
                plt.ylabel('A(mV)', fontsize=8)
                plt.xlim([0, rfExTimeEnd])
                plt.tick_params(labelsize=8)
                titleRF = 'RF amp=' + str(float("{:.2f}".format(rfAmp[indexAmp]))) + ', t90=' + str(float("{:.3f}".format(t90[indexAmp]))) + ' us, B1=' + str(float("{:.3f}".format(B1[indexAmp]))) + ' uT'
                plt.title(titleRF, fontsize=10)
                plt.show()

            plt.figure(3)
            plt.subplot(1, 2, 1)
            plt.plot(rfExAmp, abs(t90), '-k')
            plt.xlabel('RF amplitude (a.u.)', fontsize=8)
            plt.ylabel('t90 (us)', fontsize=8)
            plt.title('t90 VS RF amp', fontsize=10)

            plt.subplot(1, 2, 2)
            plt.plot(rfExAmp, abs(B1), '-k')
            plt.xlabel('RF amplitude (a.u.)', fontsize=8)
            plt.ylabel('B1 (uT)', fontsize=8)
            plt.title('B1 strength VS RF amp', fontsize=10)

            plt.show()

            if nExAmp == 1:
                plt.figure(4)
                data = matrix[0, :, 1]
                plt.plot(rfExTime, abs(data), 'k-', rfExTime, data.real, 'r--', rfExTime, data.imag, 'b--')
                plt.ylabel('A(mV)', fontsize=8)
                plt.tick_params(labelsize=8)
                titleRF = 'RF amp=' + str(float("{:.2f}".format(rfAmp[0]))) + ', t90=' + str(
                    float("{:.3f}".format(t90[0]))) + ' us, B1=' + str(
                    float("{:.3f}".format(B1[0]))) + ' uT'
                plt.title(titleRF, fontsize=10)
                plt.show()

            # for indexAmp in range(nExAmp):
            #     plt.figure(2)
            #     plt.subplot(3, 3, indexAmp+1)
            #     l, = plt.plot(timeFID, abs(matrix[indexAmp, 0, :]))
            #     axfreq = plt.axes([0.25, 0.15, 0.65, 0.03])
            #     freq = Slider(axfreq, 'Pulse time', 0.0, nExTime-1, 0, valstep=1.0)
            #
            #     def update(val):
            #         f = int(freq.val)
            #         l.set_ydata(abs(matrix[0, f, :]))
            #         title = 'Pulse time=' + str(float("{:.2f}".format(rfExTime[f])))
            #         plt.title(title, fontsize=10)
            #
            #     freq.on_changed(update)
            #     plt.show()

        return True

    def sequenceAnalysis(self, obj=''):

        result1 = {'widget': 'image',
                   'data': np.squeeze(np.abs(self.matrix[self.mapVals['RFampSteps']-1, :, :])),
                   'xLabel': "tRF (us)",
                   'yLabel': "Npoint",
                   'title': "FID maps",
                   'row': 0,
                   'col': 0
                   }

        self.output = [result1]

        self.saveRawData()

        return self.output