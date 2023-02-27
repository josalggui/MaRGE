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
import configs.hw_config as hw

class ADCdelayTest(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(ADCdelayTest, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='ADCdelayTestInfo', val='ADCdelayTest')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=8.31, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.1, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30, field='RF')
        self.addParameter(key='nPoints', string='Number of points', val=1000, field='IM')
        self.addParameter(key='addRdPoints', string='Add rd points', val=0, field='IM')
        self.addParameter(key='deadTime', string='Dead time (us)', val=100, field='RF')
        self.addParameter(key='delayPoints', string='Delay points', val=2, field='IM')
        self.addParameter(key='bw', string='Acq BW (kHz)', val=100.0, field='IM')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=50.0, field='SEQ')

    def sequenceInfo(self):
        print(" ")
        print("ADCdelayTest")
        print("Connect Tx to Rx.")

    def sequenceTime(self):
        return  # minutes, scanTime

    def sequenceRun(self, plotSeq=0):
        init_gpa = False  # Starts the gpa

        # Create input parameters
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq'] # MHz
        rfExAmp = self.mapVals['rfExAmp']
        rfExTime = self.mapVals['rfExTime'] # us
        bw = self.mapVals['bw']*1e-3 # MHz
        nPoints = self.mapVals['nPoints']
        addRdPoints = self.mapVals['addRdPoints']
        txChannel = self.mapVals['txChannel']
        rxChannel = self.mapVals['rxChannel']
        repetitionTime = self.mapVals['repetitionTime']*1e3 # us
        hw.deadTime = self.mapVals['deadTime'] # us

        # Miscellaneus

        def createSequence():
            for scan in range(nScans):
                tRx = 10000+scan*repetitionTime

                # Excitation rf pulse
                self.rfRecPulse(tRx - hw.blkTime, rfExTime, rfExAmp, 0, txChannel=txChannel)

                # Rx gate
                t0 = tRx+rfExTime/2+hw.deadTime-addRdPoints/bw
                self.rxGate(t0, acqTime+addRdPoints/bw, rxChannel=rxChannel)

            self.endSequence(tRx + addRdPoints / bw + 2 * acqTime)


        # Initialize the experiment
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1 / samplingPeriod  # MHz
        acqTime = nPoints / bw  # us
        self.mapVals['acqTime'] = acqTime
        self.mapVals['bw'] = bw*1e3 # kHz
        createSequence()

        if plotSeq == 0:
            # Run the experiment and get data
            rxd, msgs = self.expt.run()
            rxd['rx%i' % rxChannel] = np.real(rxd['rx%i'%rxChannel])-1j*np.imag(rxd['rx%i'%rxChannel])
            overData = rxd['rx%i'%rxChannel]*hw.adcFactor
            # dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            dataFull = overData
            self.mapVals['overData'] = overData
            data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
            self.mapVals['data'] = data[addRdPoints::]

            nPoints = self.mapVals['nPoints']
            bw = self.mapVals['bw']
            acqTime = nPoints/bw
            t = np.linspace(0, acqTime, nPoints)*1e3

            signal = self.mapVals['data']
            signal = np.reshape(signal, (-1))
            # Look for that first index with signal larger than 210 uV (criterion of be close to flat top of the RF pulse
            found = 0
            ii = 5
            while found == 0 and ii<nPoints:
                if np.real(signal[ii]) > 0:   # Signal corsses 0 mV
                    found = 1
                    y2 = np.real(signal[ii])
                    y1 = np.real(signal[ii-1])
                    x2 = t[ii]
                    x1 = t[ii-1]
                    m = (y2-y1)/(x2-x1)
                    t0 = -y1/m+x1
                    self.mapVals['sampledPoint'] = t0
                    print('\n0 mV crossing time: %0.0f ms' % (self.mapVals['sampledPoint']))
                ii += 1
        self.expt.__del__()

    def sequenceAnalysis(self, obj=''):
        data = self.mapVals['data']
        bw = self.mapVals['bw']
        nPoints = self.mapVals['nPoints']
        time = np.linspace(0.5 / bw, (nPoints - 0.5) / bw, nPoints)

        # Plot
        result1 = {'widget': 'curve',
                   'xData': time,
                   'yData': [np.real(data)],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Signal vs time',
                   'legend': [''],
                   'row': 0,
                   'col': 0}

        self.saveRawData()

        return [result1]

if __name__=='__main__':
    seq = ADCdelayTest()
    seq.sequenceRun()
    seq.sequenceAnalysis(obj='Standalone')
