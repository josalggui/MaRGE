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


class FOV(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(FOV, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='fovInfo', val='fov')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='OTH')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='OTH')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.3, field='OTH')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='OTH')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, field='OTH')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=10., field='OTH')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='OTH')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='OTH')
        self.addParameter(key='nPoints', string='nPoints', val=100, field='OTH')
        self.addParameter(key='fov', string='FoV (cm)', val=20.0, field='OTH')
        self.addParameter(key='shimming', string='Shimming', val=[-70, -90, 10], field='OTH')

    def sequenceInfo(self):
        print(" ")
        print("FOV check")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")


    def sequenceTime(self):
        return 0  # minutes, scanTime

    def sequenceRun(self, plotSeq=0):
        init_gpa = False  # Starts the gpa

        # Create inputs
        larmorFreq = self.mapVals['larmorFreq']
        rfExAmp = self.mapVals['rfExAmp']
        rfReAmp = self.mapVals['rfReAmp']
        rfExTime = self.mapVals['rfExTime']*1e-6
        rfReTime = self.mapVals['rfReTime']*1e-6
        echoTime = self.mapVals['echoTime']*1e-3
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        acqTime = self.mapVals['acqTime']*1e-3
        nPoints = self.mapVals['nPoints']
        fov = self.mapVals['fov']*1e-2
        shimming = self.mapVals['shimming']

        # Miscellaneous
        gradRiseTime = 400e-6  # s
        gSteps = int(gradRiseTime * 1e6 / 5)    # one point each 5 microseconds
        addRdPoints = 10  # Initial rd points to avoid artifact at the begining of rd
        resolution = fov / nPoints
        rdGradTime = acqTime+1e-3
        rdDephTime = 2e-3
        self.mapVals['resolution'] = resolution
        self.mapVals['gradRiseTime'] = gradRiseTime
        self.mapVals['addRdPoints'] = addRdPoints
        self.mapVals['gSteps'] = gSteps

        # BW
        bw = nPoints / acqTime * 1e-6 * hw.oversamplingFactor # MHz
        samplingPeriod = 1 / bw  # us

        # Readout gradient time
        if rdGradTime < acqTime:
            rdGradTime = acqTime
        self.mapVals['rdGradTime'] = rdGradTime

        # Max gradient amplitude
        rdGradAmplitude = nPoints / (hw.gammaB * fov * acqTime)
        self.mapVals['rdGradAmplitude'] = rdGradAmplitude

        # Readout dephasing amplitude
        rdDephAmplitude = 0.5 * rdGradAmplitude * (gradRiseTime + rdGradTime) / (gradRiseTime + rdDephTime)
        self.mapVals['rdDephAmplitude'] = rdDephAmplitude

        def createSequence():
            t0 = 20

            # Set shimming
            self.iniSequence(t0, shimming)
            axes = [0, 1, 2]

            for repeIndex in range(3):
                tEx = 20e3+repetitionTime*repeIndex

                # Excitation pulse
                t0 = tEx - hw.blkTime - rfExTime / 2
                self.rfRecPulse(t0, rfExTime, rfExAmp, 0)

                # Dephasing gradient
                t0 = tEx + rfExTime / 2 - hw.gradDelay
                self.gradTrap(t0, gradRiseTime, rdDephTime, rdDephAmplitude, gSteps, axes[repeIndex], shimming)

                # Refocusing pulse
                t0 = tEx + echoTime / 2 - rfReTime / 2 - hw.blkTime
                self.rfRecPulse(t0, rfReTime, rfReAmp, np.pi / 2)

                # Readout gradient
                t0 = tEx + echoTime - rdGradTime / 2 - gradRiseTime - hw.gradDelay
                self.gradTrap(t0, gradRiseTime, rdGradTime, rdGradAmplitude, gSteps, axes[repeIndex], shimming)

                # Rx gate
                t0 = tEx + echoTime - acqTime / 2 - addRdPoints / bw
                self.rxGate(t0, acqTime + 2 * addRdPoints / bw)

            # Turn off the gradients after the end of the batch
            self.endSequence(3 * repetitionTime)

        # Changing time parameters to us
        rfExTime = rfExTime * 1e6
        rfReTime = rfReTime * 1e6
        echoTime = echoTime * 1e6
        repetitionTime = repetitionTime * 1e6
        gradRiseTime = gradRiseTime * 1e6
        rdGradTime = rdGradTime * 1e6
        rdDephTime = rdDephTime * 1e6

        # Create the experiment and input the sequence instructions to experiment
        dataFull = []
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa,
                                  gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.get_rx_ts()[0] # us
        bw = 1 / samplingPeriod / hw.oversamplingFactor # MHz
        acqTime = nPoints / bw  # us
        self.mapVals['bw'] = bw
        createSequence()

        # Run or plot sequence
        if plotSeq == 1:    # Plot sequence
            print('Ploting sequence...')
            self.expt.__del__()
        else:   # Run sequence
            print('Runing...')
            rxd, msgs = self.expt.run()
            print(msgs)
            data = sig.decimate(rxd['rx0'] * 13.788, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.expt.__del__()

            # Delete addRdPoints from data
            data = np.reshape(data, (3, -1))
            data = data[:, addRdPoints:addRdPoints+nPoints]
            self.mapVals['data'] = data
        return 0

    def sequenceAnalysis(self, obj=''):
        # Get spectrum from data
        nPoints = self.mapVals['nPoints']
        acqTime = self.mapVals['acqTime']   # ms
        fov = self.mapVals['fov'] # cm
        data = self.mapVals['data']
        spectrum = np.zeros((np.size(data, 0), np.size(data, 1)), dtype=complex)
        for ii in range(3):
            spectrum[ii, :] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data[ii, :])))
        self.mapVals['spectrum'] = spectrum
        self.saveRawData()

        if obj != '':
            tVector = np.linspace(-acqTime/2, acqTime/2, nPoints)
            fVector = np.linspace(-fov/2, fov/2, nPoints)

            # Create label with rawdata name
            obj.label = QLabel(self.mapVals['fileName'])
            obj.label.setAlignment(QtCore.Qt.AlignCenter)
            obj.label.setStyleSheet("background-color: black;color: white")
            obj.parent.plotview_layout.addWidget(obj.label)

            # Signal vs rf time
            tplot = SpectrumPlot(tVector, [np.abs(data[0, :]), np.abs(data[1, :]), np.abs(data[2, :])],
                                 ['X axis', 'Y axis', 'Z axis'], 'Time (ms)', 'Signal amplitude (mV)', 'Signal vs time')
            obj.parent.plotview_layout.addWidget(tplot)

            tplot = SpectrumPlot(fVector, [np.abs(spectrum[0, :]), np.abs(spectrum[1, :]), np.abs(spectrum[2, :])],
                                 ['X axis', 'Y axis', 'Z axis'], 'Position (cm)', 'Amplitude (a.u.)', 'Spectrum')
            obj.parent.plotview_layout.addWidget(tplot)

