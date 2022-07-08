"""
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia, Spain
@date: 13 mon June 2022
@email: josalggui@i3m.upv.es
"""

import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw
from plotview.spectrumplot import SpectrumPlot
from PyQt5.QtWidgets import QLabel  # To set the figure title
from PyQt5 import QtCore            # To set the figure title
import pyqtgraph as pg              # To plot nice 3d images

class IRTSE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(IRTSE, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='IRTSEinfo', val='IRTSE')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, field='RF')
        self.addParameter(key='inversionTime', string='Inversion time (ms)', val=0.0, field='SEQ')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=500., field='SEQ')
        self.addParameter(key='nPoints', string='nPoints', val=60, field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='SEQ')
        self.addParameter(key='etl', string='Echo train length', val=1, field='SEQ')
        self.addParameter(key='crusherAmp', string='Crusher grad amp (mT/m)', val=0.0, field='OTH')
        self.addParameter(key='crusherTime', string='Crusher grad time (us)', val=0.0, field='OTH')
        self.addParameter(key='crusherDelay', string='Crusher grad delay (us)', val=0.0, field='OTH')
        self.addParameter(key='gradRiseTime', string='Gradien rise time (us)', val=400.0, field='OTH')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')

    def sequenceInfo(self):
        print(" ")
        print("Turbo Spin Echo with Inversion Recovery")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        return(repetitionTime*nScans/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq):
        init_gpa = False  # Starts the gpa

        # Create the inputs automatically. For some reason it only works if there is a few code later...
        # for key in self.mapKeys:
        #     locals()[key] = self.mapVals[key]
        #     if not key in locals():
        #         print('Error')
        #         locals()[key] = self.mapVals[key]

        # Create the inputs manually, pufff
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq']  # MHz
        rfExAmp = self.mapVals['rfExAmp']
        rfReAmp = self.mapVals['rfReAmp']
        rfExTime = self.mapVals['rfExTime'] * 1e-6
        rfReTime = self.mapVals['rfReTime'] * 1e-6
        inversionTime = self.mapVals['inversionTime'] * 1e-3
        acqTime = self.mapVals['acqTime'] * 1e-3
        echoSpacing = self.mapVals['echoSpacing'] * 1e-3
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        nPoints = self.mapVals['nPoints']
        etl = self.mapVals['etl']
        crusherAmp = self.mapVals['crusherAmp'] * 1e-3
        crusherTime = self.mapVals['crusherTime'] * 1e-6
        crusherDelay = self.mapVals['crusherDelay'] * 1e-6
        gradRiseTime = self.mapVals['gradRiseTime'] * 1e-6
        shimming = np.array(self.mapVals['shimming']) * 1e-4

        # Miscellaneous
        gSteps = int(gradRiseTime / 5e-6)
        self.mapVals['gSteps'] = gSteps

        def createSequence():
            # Set shimming
            self.iniSequence(20, shimming)

            # Initialize time
            tEx = 20e3 + inversionTime

            if inversionTime != 0:
                # Inversion pulse
                t0 = tEx-inversionTime-hw.blkTime-rfReTime/2
                self.rfRecPulse(t0, rfReTime, rfReAmp, 0)

                # Spoiler gradients to destroy residual transversal signal detected for ultrashort inversion times
                if crusherAmp!=0:
                    t0 = tEx-inversionTime+rfReTime/2
                    self.gradTrap(t0, gradRiseTime, inversionTime*0.5, crusherAmp, gSteps, 0, shimming)
                    self.gradTrap(t0, gradRiseTime, inversionTime*0.5, crusherAmp, gSteps, 1, shimming)
                    self.gradTrap(t0, gradRiseTime, inversionTime*0.5, crusherAmp, gSteps, 2, shimming)

            # Excitation pulse
            t0 = tEx-hw.blkTime-rfExTime/2
            self.rfRecPulse(t0, rfExTime, rfExAmp, 0)

            for echoIndex in range(etl):
                tEcho = tEx + (echoIndex + 1) * echoSpacing

                if crusherAmp!=0:
                    # Crusher gradient
                    t0 = tEcho - echoSpacing / 2 - crusherTime / 2 - gradRiseTime - hw.gradDelay - crusherDelay
                    self.gradTrap(t0, gradRiseTime, crusherTime, crusherAmp, gSteps, 0, shimming)
                    self.gradTrap(t0, gradRiseTime, crusherTime, crusherAmp, gSteps, 1, shimming)
                    self.gradTrap(t0, gradRiseTime, crusherTime, crusherAmp, gSteps, 2, shimming)

                # Refocusing pulse
                t0 = tEcho - echoSpacing / 2 - rfReTime / 2 - hw.blkTime
                self.rfRecPulse(t0, rfReTime, rfReAmp, np.pi / 2)

                # Rx gating
                t0 = tEcho - acqTime / 2
                self.rxGate(t0, acqTime)

            # End sequence
            self.endSequence(repetitionTime)

        # Time variables in us
        rfExTime *= 1e6
        rfReTime *= 1e6
        repetitionTime *= 1e6
        echoSpacing *= 1e6
        inversionTime *= 1e6
        crusherDelay *= 1e6
        crusherTime *= 1e6
        gradRiseTime *= 1e6
        acqTime *= 1e6

        # Bandwidth and sampling rate
        bw = nPoints / acqTime * hw.oversamplingFactor # MHz
        samplingPeriod = 1 / bw # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.get_rx_ts()[0]  # us
        bw = 1 / samplingPeriod / hw.oversamplingFactor  # MHz
        acqTime = nPoints / bw  # us
        self.mapVals['samplingPeriod'] = samplingPeriod * 1e-6
        self.mapVals['bw'] = bw * 1e6
        createSequence()
        if plotSeq:
            self.expt.__del__()
        else:
            dataOv = []
            for repeIndex in range(nScans):
                rxd, msgs = self.expt.run()
                print(msgs)
                dataOv = np.concatenate((dataOv, rxd['rx0'] * 13.788), axis=0)
            self.expt.__del__()
            self.mapVals['dataOv'] = dataOv
            dataFull = sig.decimate(dataOv, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['dataFull'] = dataFull
            data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
            self.mapVals['data'] = data
            data = np.reshape(data, (etl, -1))
            self.mapVals['sampledSignal'] = data[0, int(nPoints / 2)]   # To be used by sweep class
        return 0

    def sequenceAnalysis(self, obj=''):
        self.saveRawData()
        data = self.mapVals['data']
        acqTime = self.mapVals['acqTime']
        nPoints = self.mapVals['nPoints']
        etl = self.mapVals['etl']
        echoSpacing = self.mapVals['echoSpacing']
        t = np.linspace(-acqTime/2, acqTime/2, nPoints)
        tVector = []
        for echo in range(etl):
            tVector = np.concatenate((tVector, t + echoSpacing * (echo + 1)), axis=0)


        # Signal vs inverion time
        plot = SpectrumPlot(tVector, [np.abs(data)], [''], 'Time (ms)', 'Signal amplitude (mV)', '')

        return([plot])


