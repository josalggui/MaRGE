"""
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia, Spain
@date: 19 tue Apr 2022
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

class ShimmingSweep(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(ShimmingSweep, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='ShimmingSweepInfo', val='ShimmingSweep')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='OTH')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='OTH')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.3, field='OTH')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='OTH')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, field='OTH')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=10., field='OTH')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='OTH')
        self.addParameter(key='nPoints', string='nPoints', val=60, field='OTH')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='OTH')
        self.addParameter(key='shimming0', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='nShimming', string='n Shimming steps', val=10, field='OTH')
        self.addParameter(key='dShimming', string='Shiming step', val=[2.5, 2.5, 2.5], field='OTH')

    def sequenceRun(self, plotSeq):
        init_gpa = False  # Starts the gpa
        demo = False

        # # Create the inputs automatically. For some reason it only works if there is a few code later...
        # for key in self.mapKeys:
        #     if type(self.mapVals[key])==list:
        #         locals()[key] = np.array(self.mapVals[key])
        #     else:
        #         locals()[key] = self.mapVals[key]

        # I do not understand why I cannot create the input parameters automatically
        seqName = self.mapVals['seqName']
        larmorFreq = self.mapVals['larmorFreq']
        rfExAmp = self.mapVals['rfExAmp']
        rfExTime = self.mapVals['rfExTime'] # us
        rfReAmp = self.mapVals['rfReAmp']
        rfReTime = self.mapVals['rfReTime'] # us
        echoTime = self.mapVals['echoTime']*1e3 # us
        repetitionTime = self.mapVals['repetitionTime']*1e3 # us
        nPoints = self.mapVals['nPoints']
        acqTime = self.mapVals['acqTime']*1e3 # us
        shimming0 = np.array(self.mapVals['shimming0'])*1e-4
        nShimming = self.mapVals['nShimming']
        dShimming = np.array(self.mapVals['dShimming'])*1e-4

        # Shimming vectors
        dsx = nShimming * dShimming[0]
        dsy = nShimming * dShimming[1]
        dsz = nShimming * dShimming[2]
        sxVector = np.reshape(np.linspace(shimming0[0] - dsx / 2, shimming0[0] + dsx / 2, num=nShimming, endpoint=False), (nShimming, 1))
        syVector = np.reshape(np.linspace(shimming0[1] - dsy / 2, shimming0[1] + dsy / 2, num=nShimming, endpoint=False), (nShimming, 1))
        szVector = np.reshape(np.linspace(shimming0[2] - dsz / 2, shimming0[2] + dsz / 2, num=nShimming, endpoint=False), (nShimming, 1))
        sxStatic = np.reshape(np.ones(nShimming) * shimming0[0], (nShimming, 1))
        syStatic = np.reshape(np.ones(nShimming) * shimming0[1], (nShimming, 1))
        szStatic = np.reshape(np.ones(nShimming) * shimming0[2], (nShimming, 1))
        sx = np.concatenate((sxVector, syStatic, szStatic), axis=1)
        sy = np.concatenate((sxStatic, syVector, szStatic), axis=1)
        sz = np.concatenate((sxStatic, syStatic, szVector), axis=1)
        shimmingMatrix = np.concatenate((sx,sy,sz), axis=0)
        self.mapVals['sxVector'] = sxVector
        self.mapVals['syVector'] = syVector
        self.mapVals['szVector'] = szVector

        #  SEQUENCE  ############################################################################################
        def createSequence():
            for repeIndex in range(3*nShimming):
                # Set time for repetition
                t0 = 20+repeIndex*repetitionTime

                # Set shimming
                self.iniSequence(t0, shimmingMatrix[repeIndex, :])

                # Initialize time
                tEx = t0+20e3

                # Excitation pulse
                t0 = tEx - hw.blkTime - rfExTime / 2
                self.rfRecPulse(t0, rfExTime, rfExAmp, 0)

                # Refocusing pulse
                t0 = tEx + echoTime / 2 - rfReTime / 2 - hw.blkTime
                self.rfRecPulse(t0, rfReTime, rfReAmp, np.pi / 2)

                # Acquisition window
                t0 = tEx + echoTime - acqTime / 2
                self.rxGate(t0, acqTime)

            # End sequence
            self.endSequence(repetitionTime)

        # Create experiment
        bw = nPoints / acqTime * hw.oversamplingFactor  # MHz
        samplingPeriod = 1 / bw
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa,
                                  gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1 / samplingPeriod / hw.oversamplingFactor # MHz
        self.mapVals['bw'] = bw * 1e6 # Hz
        acqTime = nPoints / bw # us
        createSequence()

        if plotSeq:
            print('Ploting sequence...')
            self.expt.plot_sequence()
            plt.show()
            self.expt.__del__()
            return 0
        else:
            print('Runing...')
            rxd, msgs = self.expt.run()
            print(msgs)
            data = sig.decimate(rxd['rx0'] * 13.788, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['data'] = data
        self.expt.__del__()


    def sequenceAnalysisGUI(self, obj=''):
        nShimming = self.mapVals['nShimming']
        nPoints = self.mapVals['nPoints']
        data = np.reshape(self.mapVals['data'], (3, nShimming, -1))
        data = data[:, :, int(nPoints/2)]
        sxVector = np.squeeze(self.mapVals['sxVector'])
        syVector = np.squeeze(self.mapVals['syVector'])
        szVector = np.squeeze(self.mapVals['szVector'])
        self.mapVals['amplitudeVSshimming'] = data

        self.saveRawData()

        # Add larmor frequency to the layout
        obj.label = QLabel(self.mapVals['fileName'])
        obj.label.setAlignment(QtCore.Qt.AlignCenter)
        obj.label.setStyleSheet("background-color: black;color: white")
        obj.parent.plotview_layout.addWidget(obj.label)

        # Add shimming x to the layout
        plotX = SpectrumPlot(sxVector*1e4, np.abs(data[0,:]), [], [], 'ShimmingX', 'Signal amplitude (mV)',
                            "%s" % ('Shimming X'))
        obj.parent.plotview_layout.addWidget(plotX)

        # Add shimming y to the layout
        plotY = SpectrumPlot(syVector*1e4, np.abs(data[1, :]), [], [], 'ShimmingX', 'Signal amplitude (mV)',
                             "%s" % ('Shimming Y'))
        obj.parent.plotview_layout.addWidget(plotY)

        # Add shimming z to the layout
        plotZ = SpectrumPlot(szVector*1e4, np.abs(data[2, :]), [], [], 'ShimmingZ', 'Signal amplitude (mV)',
                             "%s" % ('Shimming Z'))
        obj.parent.plotview_layout.addWidget(plotZ)