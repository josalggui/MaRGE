"""Pulse sequence for B1 field calibration."""

import marge.controller.experiment_gui as ex
import numpy as np
import math
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import marge.configs.hw_config as hw
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

class B1calibration(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(B1calibration, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='B1calibration', val='B1calibration')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmpInit', string='RF initial amplitude (a.u.)', val=0.1, field='RF')
        self.addParameter(key='rfExAmpFinal', string='RF final amplitude (a.u.)', val=1.0, field='RF')
        self.addParameter(key='rfExTimeInit', string='RF initial length (us)', val=2.0, field='RF')
        self.addParameter(key='rfExTimeFinal', string='RF final length (us)', val=100.0, field='RF')
        self.addParameter(key='RFtimeSteps', string='RF time steps', val=50, field='RF')
        self.addParameter(key='RFampSteps', string='RF amp steps (1/4/9)', val=1, field='RF', tip="Only enabled 1, 4 or 9")
        self.addParameter(key='deadTime', string='RF dead time (us)', val=300.0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=300., field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=1.0, field='SEQ')
        self.addParameter(key='nReadout', string='Number of points', val=600, field='IM')
        self.addParameter(key='timePlot', string='Point to plot', val=50, field='IM')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-0, -0, 0], field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='rfExPhase', string='RF Exc Phase', val=0, field='RF')
        self.addParameter(key='PulseShape', string='Pulse shape', val=0, field='RF', tip="0=Rect, 1=Sinc, 2=Sinc+Hanning, 3=HypSec, 4=Chirp, 5=")
        self.addParameter(key='US', string='Samples pulses undersampling', val=4, field='RF')
        self.addParameter(key='beta', string='Modulation frequency beta (Hz)', val=50, field='RF')
        self.addParameter(key='mu', string='Freq sweep width mu (a.u.)', val=5, field='RF')

    def sequenceInfo(self):

        print("B1calibration\n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        RFampSteps = self.mapVals['RFampSteps']
        RFtimeSteps = self.mapVals['RFtimeSteps']
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        return (RFampSteps * RFtimeSteps * repetitionTime * nScans / 60)  # minutes, scanTime

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
        nExTime = self.mapVals['RFtimeSteps']
        deadTime = self.mapVals['deadTime']  # us
        repetitionTime = self.mapVals['repetitionTime'] * 1e3  # us
        acqTime = self.mapVals['acqTime'] * 1e3  # us
        nReadout = self.mapVals['nReadout']
        shimming = np.array(self.mapVals['shimming']) * 1e-4
        txChannel = self.mapVals['txChannel']
        rxChannel = self.mapVals['rxChannel']
        rfExPhase = self.mapVals['rfExPhase']
        timePlot = self.mapVals['timePlot']
        ShapeEnvelope = self.mapVals['PulseShape']
        undersampling = self.mapVals['US']
        beta = self.mapVals['beta']
        mu = self.mapVals['mu']

        if nExAmp not in [1, 4, 9]:
            print("RF amp steps only can be 1, 4 or 9")
            return(0)

        if nExAmp == 1:
            rfExAmpEnd = rfExAmpIni

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

                        if ShapeEnvelope == 0:
                            self.rfRecPulse(tTx, rfExTime[indexExTime], rfExAmp[indexAmp], 0, channel=txChannel)

                        if ShapeEnvelope == 1:
                            self.rfSincPulseWithoutHanning(tTx, rfExTime[indexExTime], rfExAmp[indexAmp],
                                                           int(math.floor(rfExTime[indexExTime] / undersampling)),
                                                           rfPhase=0, nLobes=7, channel=0, rewrite=True)

                        if ShapeEnvelope == 2:
                            self.rfSincPulseWithHanning(tTx, rfExTime[indexExTime], rfExAmp[indexAmp],
                                                        int(math.floor(rfExTime[indexExTime] / undersampling)),
                                                        rfPhase=0, nLobes=7, channel=0, rewrite=True)

                        if ShapeEnvelope == 3:
                            self.rfHyperbolicSecantPulse(tTx, rfExTime[indexExTime], rfExAmp[indexAmp],
                                                         int(math.floor(rfExTime[indexExTime] / undersampling)), beta,
                                                         mu, rfPhase=0, channel=0, rewrite=True)

                        if ShapeEnvelope == 4:
                            self.rfChirpPulse(tTx, rfExTime[indexExTime], rfExAmp[indexAmp],
                                              int(math.floor(rfExTime[indexExTime] / undersampling)), beta, rfPhase=0,
                                              channel=0, rewrite=True)

                        if ShapeEnvelope == 5:
                            self.rfAdiabaticConstantBeff(tTx, rfExTime[indexExTime], rfExAmp[indexAmp],
                                                         int(math.floor(rfExTime[indexExTime] / undersampling)), beta,
                                                         rfPhase=0, channel=0, rewrite=True)

                        # Rx gate
                        tRx = tTx + hw.blkTime + rfExTime[indexExTime] + deadTime
                        self.rxGateSync(tRx, acqTime, channel=rxChannel)
            self.endSequence(tStart + repetitionTime * nExAmp * (nExTime + 1) * nScans)

        # Initialize the experiment
        bw = nReadout / acqTime  # MHz
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa,
                                  gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
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
            dataFull = self.decimate(rxd['rx%i' % rxChannel], nScans * nExAmp * nExTime)

            # Average data
            data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
            self.mapVals['dataRaw'] = data
            print('End')
            matrixData = np.reshape(data, (nExAmp, nExTime, nReadout))

            t90List = np.zeros(nExAmp)
            B1List = np.zeros(nExAmp)
            fRabi = np.zeros(nExAmp)

            for indexAmp in range(nExAmp):
                data = matrixData[indexAmp, :, timePlot]
                spl = InterpolatedUnivariateSpline(rfExTime, abs(data))
                spl.set_smoothing_factor(0.5)
                interpolatedFIDtime = np.linspace(rfExTimeIni, rfExTimeEnd, 5 * nExTime, endpoint=True)
                fitteddata = spl(interpolatedFIDtime)

                # Get the indices of maximum element in fitted data
                indexmax = np.argmax(fitteddata)
                t90List[indexAmp] = interpolatedFIDtime[indexmax]
                B1List[indexAmp] = ((np.pi / 2) / (2 * np.pi * hw.gammaB * t90List[indexAmp] * 1e-6)) * 1e6
                fRabi[indexAmp] = 1 / (4 * t90List[indexAmp] * 1e-6)

            self.mapVals['matrixData'] = matrixData
            self.mapVals['rfExTime'] = rfExTime
            self.mapVals['timeFID'] = timeFID
            self.mapVals['t90List'] = t90List
            self.mapVals['B1List'] = B1List
            self.mapVals['rfAmpList'] = rfAmp
            self.mapVals['fRabi'] = fRabi
        self.expt.__del__()

        return True

    def sequenceAnalysis(self, obj=''):
        """
        Process raw acquired data and compute the output images and metrics.

        Reconstructs the image from k-space, computes SNR or other sequence-specific
        figures of merit, populates output_dict with result arrays, and fills
        dicom_meta_data with the relevant DICOM tags for saving.

        Args:
            obj (str, optional): Processing mode selector (sequence-dependent). Defaults to ''.

        Returns:
            tuple: (output_dict, dicom_meta_data) with processed results and metadata.
        """

        # Signal and spectrum from 'fir' and decimation
        matrixData = self.mapVals['matrixData']
        rfExTimeVector = self.mapVals['rfExTime']
        timeFID = self.mapVals['timeFID']
        t0 = self.mapVals['timePlot']
        t90List = self.mapVals['t90List']
        B1List = self.mapVals['B1List']
        rfAmpList = self.mapVals['rfAmpList']
        fRabi = self.mapVals['fRabi']

        if self.mapVals['RFampSteps'] == 9:
            # Add time signal to the layout
            n = 0
            result1 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 0,
                       'col': 0}

            # Add frequency spectrum to the layout
            n = 1
            result2 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 0,
                       'col': 1}
            n = 2
            result3 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 0,
                       'col': 2}

            # Add frequency spectrum to the layout
            n = 3
            result4 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 1,
                       'col': 0}

            n = 4
            result5 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 1,
                       'col': 1}

            # Add frequency spectrum to the layout
            n = 5
            result6 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 1,
                       'col': 2}

            n = 6
            result7 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 2,
                       'col': 0}

            # Add frequency spectrum to the layout
            n = 7
            result8 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 2,
                       'col': 1}

            n = 8
            result9 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 2,
                       'col': 2}

            # create self.out to run in iterative mode
            self.output = [result1, result2, result3, result4, result5, result6, result7, result8, result9]
            self.saveRawData()
            return self.output

        if self.mapVals['RFampSteps'] == 4:
            # Add time signal to the layout
            n = 0
            result1 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 0,
                       'col': 0}

            # Add frequency spectrum to the layout
            n = 1
            result2 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 0,
                       'col': 1}
            n = 2
            result3 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 1,
                       'col': 0}
            n = 3
            result4 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 1,
                       'col': 1}

            # create self.out to run in iterative mode
            self.output = [result1, result2, result3, result4]
            self.saveRawData()
            return self.output

        if self.mapVals['RFampSteps'] == 1:
            # Add time signal to the layout
            n = 0
            sampledsignal = np.abs(matrixData[n, :, t0])
            s_norm = sampledsignal / np.max(sampledsignal)
            alpha = np.arcsin(s_norm)
            interp_t = interp1d(alpha, rfExTimeVector, kind='linear', fill_value='extrapolate')
            alpha_query = np.linspace(0, np.pi / 2, 100)
            t_alpha = interp_t(alpha_query)

            result1 = {'widget': 'curve',
                       'xData': rfExTimeVector,
                       'yData': [np.abs(matrixData[n, :, t0]), np.real(matrixData[n, :, t0]),
                                 np.imag(matrixData[n, :, t0])],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Signal amplitude (mV)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs', 'real', 'imag'],
                       'row': 0,
                       'col': 0}

            result2 = {'widget': 'curve',
                       'xData': t_alpha,
                       'yData': [np.degrees(alpha_query)],
                       'xLabel': 'RF pulse length (us)',
                       'yLabel': 'Flip angle (º)',
                       'title': f'RF amp={rfAmpList[n]:.2f} a.u., t90º={t90List[n]:.2f} us, B1={B1List[n]:.2f} uT, fRabi={fRabi[n]:.2f} Hz',
                       'legend': ['abs'],
                       'row': 0,
                       'col': 1}

            # create self.out to run in iterative mode
            self.output = [result1]
            self.saveRawData()
            return self.output

if __name__ == '__main__':
    seq = B1calibration()
    seq.sequenceRun()
    seq.sequenceAnalysis(obj='Standalone')
