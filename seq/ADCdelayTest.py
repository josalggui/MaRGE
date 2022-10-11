"""
@author: P. Borreguero
@author: J.M. Algarín
September 2022
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
import experiment as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw
from plotview.spectrumplot import SpectrumPlot
import pyqtgraph as pg

class ADCdelayTest(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(ADCdelayTest, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='ADCdelayTest', val='ADCdelayTest')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=8.31, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=1.0, field='RF')
        self.addParameter(key='nPoints', string='Number of points', val=1000, field='IM')
        self.addParameter(key='addRdPoints', string='Add rd points', val=0, field='IM')
        self.addParameter(key='delayPoints', string='Delay points', val=2, field='IM')
        self.addParameter(key='bw', string='Acq BW (kHz)', val=100.0, field='IM')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')

    def sequenceInfo(self):
        print(" ")
        print("ADCdelayTest")
        print("Connect Tx to Rx.")

    def sequenceTime(self):
        return(0)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0):
        init_gpa = False  # Starts the gpa

        # Create input parameters
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq'] # MHz
        rfExAmp = self.mapVals['rfExAmp']
        bw = self.mapVals['bw']*1e-3 # MHz
        nPoints = self.mapVals['nPoints']
        addRdPoints = self.mapVals['addRdPoints']
        txChannel = self.mapVals['txChannel']
        rxChannel = self.mapVals['rxChannel']

        # Miscellaneus

        def createSequence():
            tRx = 10000
            self.rxGate(tRx + addRdPoints / bwReal, acqTimeReal, rxChannel=rxChannel)
            self.rfRecPulse(tRx - hw.blkTime, 200 / bwReal, rfExAmp, 0, txChannel=txChannel)
            self.endSequence((tRx + addRdPoints / bwReal + 2 * acqTimeReal)*nScans)


        # Initialize the experiment
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriodReal = self.expt.get_rx_ts()[0]
        bwReal = 1 / samplingPeriodReal  # MHz
        acqTimeReal = nPoints / bwReal  # us
        self.mapVals['acqTime'] = acqTimeReal
        self.mapVals['bw'] = bwReal*1e3 # kHz
        createSequence()

        if plotSeq == 0:
            # Run the experiment and get data
            rxd, msgs = self.expt.run()
            rxd['rx%i' % rxChannel] = np.real(rxd['rx%i'%rxChannel])-1j*np.imag(rxd['rx%i'%rxChannel])
            overData = rxd['rx%i'%rxChannel]*13.788
            # dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            dataFull = overData
            self.mapVals['overData'] = overData
            data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
            self.mapVals['data'] = data

            signal = self.mapVals['data']
            signal = np.reshape(signal, (-1))
            # Look for that first index with signal larger than 210 uV (criterion of be close to flat top of the RF pulse
            found = 0
            ii = 0
            while found == 0 and ii<nPoints:
                if abs(signal[ii]) > 210:
                    found = 1
                    self.mapVals['sampledPoint'] = ii
                ii += 1
            # print('\n First working point: %i' % (self.mapVals['sampledPoint'] + 1))
        self.expt.__del__()

    def sequenceAnalysis(self, obj=''):
        delayPoints = self.mapVals['delayPoints']
        signal = self.mapVals['data']
        signal = np.reshape(signal, (-1))
        bw = self.mapVals['bw']*1e3 # kHz
        nPoints = self.mapVals['nPoints']
        nVector = np.linspace(1, nPoints, nPoints)

        # Correct the signal position
        signal2 = signal*0
        signal2[0:nPoints-delayPoints] = signal[delayPoints::]

        self.saveRawData()

        # Add time signal to the layout
        signalVsPointWidget = SpectrumPlot(xData=nVector,
                                        yData=[np.abs(signal), np.abs(signal2)],
                                        legend=['Acquired', 'Corrected'],
                                        xLabel='Points',
                                        yLabel='Signal amplitude (mV)',
                                        title='Signal vs acquired point')
        signalVsPointWidget.plotitem.curves[0].setSymbol('x')
        signalVsPointWidget.plotitem.curves[1].setSymbol('o')

        if obj == 'Standalone':
            signalVsPointWidget.show()
            pg.exec()

        return([signalVsPointWidget])

if __name__=='__main__':
    seq = ADCdelayTest()
    seq.sequenceRun()
    seq.sequenceAnalysis(obj='Standalone')