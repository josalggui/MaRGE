"""
@author: José Miguel Algarín Guisado
MRILAB @ I3M
"""

import os
import sys
#*****************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************
import controller.experiment_gui as ex
import numpy as np
import seq.mriBlankSeq as blankSeq
import scipy.signal as sig
import configs.hw_config as hw
import pyqtgraph as pg
from plotview.spectrumplot import SpectrumPlot

class FID(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(FID, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='FIDinfo', val='FID')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
        self.addParameter(key='deadTime', string='RF dead time (us)', val=400.0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='SEQ')
        self.addParameter(key='nPoints', string='Number of points', val=100, field='IM')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='shimmingTime', string='Shimming time (ms)', val=1, field='OTH')

    def sequenceInfo(self):
        print(" ")
        print("FID")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs a single FID")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        return(repetitionTime*nScans/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0):
        init_gpa = False  # Starts the gpa

        # Create input parameters
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq'] # MHz
        rfExAmp = self.mapVals['rfExAmp']
        rfExTime = self.mapVals['rfExTime'] # us
        deadTime = self.mapVals['deadTime'] # us
        repetitionTime = self.mapVals['repetitionTime']*1e3 # us
        acqTime = self.mapVals['acqTime']*1e3 # us
        nPoints = self.mapVals['nPoints']
        shimming = np.array(self.mapVals['shimming'])*hw.shimming_factor
        txChannel = self.mapVals['txChannel']
        rxChannel = self.mapVals['rxChannel']
        shimmingTime = self.mapVals['shimmingTime']*1e3 # us

        # Miscellaneus
        bw = nPoints/acqTime # MHz

        def createSequence():
            # Shimming
            self.iniSequence(20, shimming)  # shimming is turned on 20 us after experiment beginning

            for scan in range(nScans):
                tEx = shimmingTime + repetitionTime*scan + hw.blkTime + rfExTime / 2

                # Excitation pulse
                t0 = tEx - hw.blkTime - rfExTime / 2
                self.rfRecPulse(t0, rfExTime, rfExAmp, 0, txChannel=txChannel)

                # Rx gate
                t0 = tEx + rfExTime / 2 + deadTime
                self.rxGateSync(t0, acqTime, rxChannel=rxChannel)
                # self.ttl(t0, acqTime, channel=1, rewrite=True)

            self.endSequence(repetitionTime*nScans)


        # Initialize the experiment
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.getSamplingRate()
        bw = 1 / samplingPeriod
        acqTime = nPoints / bw  # us
        self.mapVals['acqTime'] = acqTime*1e-3 # ms
        self.mapVals['bw'] = bw # MHz
        createSequence()

        if not plotSeq:
            # Run the experiment and get data
            rxd, msgs = self.expt.run()

            # Decimate the signal
            dataFull = self.decimate(rxd['rx%i' % rxChannel], nScans)

            # Average data
            data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
            self.mapVals['data'] = data

            # Save data to sweep plot (single point)
            self.mapVals['sampledPoint'] = data[0]

        self.expt.__del__()

    def sequenceAnalysis(self, obj=''):
        # Signal and spectrum from 'fir' and decimation
        signal = self.mapVals['data']
        bw = self.mapVals['bw']*1e3 # kHz
        nPoints = self.mapVals['nPoints']
        deadTime = self.mapVals['deadTime']*1e-3 # ms
        rfExTime = self.mapVals['rfExTime']*1e-3 # ms
        tVector = np.linspace(rfExTime/2 + deadTime + 0.5/bw, rfExTime/2 + deadTime + (nPoints-0.5)/bw, nPoints)
        fVector = np.linspace(-bw/2, bw/2, nPoints)
        spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal))))
        fitedLarmor=self.mapVals['larmorFreq'] + fVector[np.argmax(np.abs(spectrum))] * 1e-3

        # Get the central frequency
        print('Larmor frequency: %1.5f MHz' % fitedLarmor)
        self.mapVals['signalVStime'] = [tVector, signal]
        self.mapVals['spectrum'] = [fVector, spectrum]

        # Saver raw data after processing
        self.saveRawData()

        # Add time signal to the layout (Signal acquired with 'fir' fiter and decimation)
        signalPlotWidget = SpectrumPlot(xData=tVector,
                                        yData=[np.abs(signal), np.real(signal), np.imag(signal)],
                                        legend=['abs', 'real', 'imag'],
                                        xLabel='Time (ms)',
                                        yLabel='Signal amplitude (mV)',
                                        title='Signal vs time')

        # Add frequency spectrum to the layout
        spectrumPlotWidget = SpectrumPlot(xData=fVector,
                                          yData=[spectrum],
                                          legend=[''],
                                          xLabel='Frequency (kHz)',
                                          yLabel='Spectrum amplitude (a.u.)',
                                          title='Spectrum')        # spectrumPlotWidget.plotitem.setLogMode(y=True)

        # create self.out to run in iterative mode
        self.out = [signalPlotWidget, spectrumPlotWidget]

        if obj=='Standalone':
            signalPlotWidget.show()
            spectrumPlotWidget.show()
            pg.exec()

        return(self.out)

if __name__=='__main__':
    seq = FID()
    seq.sequenceRun()
    seq.sequenceAnalysis(obj='Standalone')