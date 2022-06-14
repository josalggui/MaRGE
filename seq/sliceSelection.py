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
from PyQt5 import QtCore  # To set the figure title

class SliceSelection(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(SliceSelection, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='SliceSelectionInfo', val='SliceSelection')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='OTH')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.03, field='OTH')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.06, field='OTH')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=200.0, field='OTH')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=200.0, field='OTH')
        self.addParameter(key='rfEnvelope', string='RF envelope 0->Rec, 1->Sinc', val=0, field='OTH')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=10., field='OTH')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=100., field='OTH')
        self.addParameter(key='axis', string='Axis 0->x, 1->y, 2->z', val=0, field='OTH')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='OTH')
        self.addParameter(key='nPoints', string='nPoints', val=100, field='OTH')
        self.addParameter(key='slThickness', string='Slice thickness (cm)', val=2.0, field='OTH')
        self.addParameter(key='slPreemphasis', string='Slice preemphasis', val=1.0, field='OTH')
        self.addParameter(key='gradRiseTime', string='Gradient rise time (us)', val=400.0, field='OTH')
        self.addParameter(key='shimming', string='Shimming', val=[-70, -90, 10], field='OTH')

    def sequenceRun(self, plotSeq=0):
        init_gpa = False

        # Inputs
        larmorFreq = self.mapVals['larmorFreq']
        rfExAmp = self.mapVals['rfExAmp']
        rfReAmp = self.mapVals['rfReAmp']
        rfExTime = self.mapVals['rfExTime']*1e-6
        rfReTime = self.mapVals['rfReTime']*1e-6
        rfEnvelope = self.mapVals['rfEnvelope']
        echoTime = self.mapVals['echoTime']*1e-3
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        acqTime = self.mapVals['acqTime']*1e-3
        nPoints = self.mapVals['nPoints']
        slThickness = self.mapVals['slThickness']*1e-2
        slPreemphasis = self.mapVals['slPreemphasis']
        gradRiseTime = self.mapVals['gradRiseTime']*1e-6
        shimming = np.array(self.mapVals['shimming'])*1e-4
        axis = self.mapVals['axis']

        # Miscellaneous
        rfSincLobes = 7  # Number of lobes for sinc rf excitation, BW = rfSincLobes/rfTime
        gSteps = int(gradRiseTime * 1e6 / 5)

        # Slice selection dephasing gradient time
        ssDephGradTime = (rfExTime - gradRiseTime) / 2
        self.mapVals['ssDephGradTime'] = ssDephGradTime

        # Slice selection gradient
        if rfEnvelope == 1: # Sinc pulse
            ssGradAmplitude = rfSincLobes / (hw.gammaB * slThickness * rfExTime)
        elif rfEnvelope == 0:   # Square pulse
            ssGradAmplitude = 1 / (hw.gammaB * slThickness * rfExTime)
        self.mapVals['ssGradAmplitude'] = ssGradAmplitude

        def createSequence():
            t0 = 20
            tEx = 20e3

            # Set shimming
            self.iniSequence(t0, shimming)

            # Dephasing slice selection gradient
            t0 = tEx - rfExTime / 2 - gradRiseTime - hw.gradDelay
            self.gradTrap(t0, gradRiseTime, rfExTime, ssGradAmplitude, gSteps, axis, shimming)

            # Excitation pulse
            t0 = tEx - hw.blkTime - rfExTime / 2
            if rfEnvelope == 0: # square pulse
                self.rfRecPulse(t0, rfExTime, rfExAmp, 0)
            elif rfEnvelope == 1:   # sinc pulsse
                self.rfSincPulse(t0, rfExTime, rfSincLobes, rfExAmp, 0)

            # Rephasing slice selection gradient
            t0 = tEx + rfExTime / 2 + gradRiseTime - hw.gradDelay
            self.gradTrap(t0, gradRiseTime, ssDephGradTime, -ssGradAmplitude * slPreemphasis, gSteps, axis, shimming)

            # Refocusing pulse
            t0 = tEx + echoTime / 2 - rfReTime / 2 - hw.blkTime
            if rfEnvelope == 0:
                self.rfRecPulse(t0, rfReTime, rfReAmp, np.pi / 2)
            if rfEnvelope == 1:
                self.rfSincPulse(t0, rfReTime, rfSincLobes, rfReAmp, np.pi / 2)

            # Rx gate
            t0 = tEx + echoTime - acqTime / 2
            self.rxGate(t0, acqTime)

            self.endSequence(repetitionTime)

        # Time variables in us
        rfExTime *= 1e6
        rfReTime *= 1e6
        repetitionTime *= 1e6
        echoTime *= 1e6
        gradRiseTime *= 1e6
        ssDephGradTime *= 1e6
        acqTime *= 1e6

        # Bandwidth and sampling rate
        bw = nPoints / acqTime * hw.oversamplingFactor  # MHz
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa,
                                  gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.get_rx_ts()[0]  # us
        bw = 1 / samplingPeriod / hw.oversamplingFactor  # MHz
        acqTime = nPoints / bw  # us
        self.mapVals['samplingPeriod'] = samplingPeriod * 1e-6
        self.mapVals['bw'] = bw * 1e6
        createSequence()
        if plotSeq:
            self.expt.plot_sequence()
            plt.show()
            self.expt.__del__()
        else:
            rxd, msgs = self.expt.run()
            print(msgs)
            data = sig.decimate(rxd['rx0'] * 13.788, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['data'] = data
            self.mapVals['sampledSignal'] = data[int(nPoints/2)]
            self.kind = 'Point'
            self.expt.__del__()
        return 0

    def sequenceAnalysis(self, obj=''):
        data = self.mapVals['data']
        acqTime = self.mapVals['acqTime'] # ms
        tVector = np.linspace(-acqTime/2, acqTime/2, np.size(data))

        self.saveRawData()

        if obj != '':
            # Add larmor frequency to the layout
            obj.label = QLabel(self.mapVals['fileName'])
            obj.label.setAlignment(QtCore.Qt.AlignCenter)
            obj.label.setStyleSheet("background-color: black;color: white")
            obj.parent.plotview_layout.addWidget(obj.label)

            # Signal vs time
            plot = SpectrumPlot(tVector, [np.abs(data)], [''], 'Time (ms)', 'Signal amplitude (mV)', '')
            obj.parent.plotview_layout.addWidget(plot)




