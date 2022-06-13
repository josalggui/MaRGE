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
        sxVector = np.linspace(shimming0[0] - dsx / 2, shimming0[0] + dsx / 2, num=nShimming, endpoint=False)
        syVector = np.linspace(shimming0[1] - dsy / 2, shimming0[1] + dsy / 2, num=nShimming, endpoint=False)
        szVector = np.linspace(shimming0[2] - dsz / 2, shimming0[2] + dsz / 2, num=nShimming, endpoint=False)

        x = 1



        # #  SEQUENCE  ############################################################################################
        # def createSequence():
        #     for sx in sxVector:
        #         # Set time and shimming
        #         t0 = 20+repeIndex*repetitionTime
        #
        #
        #
        #         # Set shimming
        #         mri.iniSequence(expt, to, shimming)
        #
        #         # Initialize time
        #         tEx = 20e3
        #
        #         # Excitation pulse
        #         t0 = tEx - hw.blkTime - rfExTime / 2
        #         mri.rfRecPulse(expt, t0, rfExTime, rfExAmp, 0)
        #
        #         # Refocusing pulse
        #         t0 = tEx + echoTime / 2 - rfReTime / 2 - hw.blkTime
        #         mri.rfRecPulse(expt, t0, rfReTime, rfReAmp, np.pi / 2)
        #
        #         # Acquisition window
        #         t0 = tEx + echoTime - acqTime / 2
        #         mri.rxGate(expt, t0, acqTime)
        #
        #     # End sequence
        #     mri.endSequence(expt, repetitionTime)


