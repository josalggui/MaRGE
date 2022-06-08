"""
@author: T. Guallart
@author: J.M. Algarín

@summary: increase the pulse width and plot the peak value of the signal received 
@status: under development
@todo:

"""
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw
from plotview.spectrumplot import SpectrumPlot

class RabiFlops(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RabiFlops, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='RabiFlops', val='RabiFlops')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='OTH')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='OTH')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='OTH')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=10.0, field='OTH')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=100., field='OTH')
        self.addParameter(key='nPoints', string='nPoints (rd, ph, sl)', val=60, field='OTH')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='OTH')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='rfExTime0', string='Rf pulse time, Start (us)', val=5.0, field='OTH')
        self.addParameter(key='rfExTime1', string='RF pulse time, End (us)', val=50.0, field='OTH')
        self.addParameter(key='nSteps', string='Number of steps', val=10, field='OTH')

    def sequenceRun(self, plotSeq):
        init_gpa = False  # Starts the gpa

        # # Create the inputs automatically. For some reason it only works if there is a few code later...
        # for key in self.mapKeys:
        #     if type(self.mapVals[key])==list:
        #         locals()[key] = np.array(self.mapVals[key])
        #     else:
        #         locals()[key] = self.mapVals[key]

        # I do not understand why I cannot create the input parameters automatically
        seqName = self.mapVals['seqName']
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq']
        rfExAmp = self.mapVals['rfExAmp']
        echoTime = self.mapVals['echoTime']
        repetitionTime = self.mapVals['repetitionTime']
        nPoints = self.mapVals['nPoints']
        acqTime = self.mapVals['acqTime']
        shimming = np.array(self.mapVals['shimming'])
        rfExTime0 = self.mapVals['rfExTime0']
        rfExTime1 = self.mapVals['rfExTime1']
        nSteps = self.mapVals['nSteps']
        drfPhase = self.mapVals['drfPhase']

        rawData = {}
        rawData['seqName'] = seqName
        rawData['nScans'] = nScans
        rawData['larmorFreq'] = larmorFreq*1e6
        rawData['rfExAmp'] = rfExAmp
        rawData['echoTime'] = echoTime*1e-3
        rawData['repetitionTime'] = repetitionTime*1e-3
        rawData['nPoints'] = nPoints
        rawData['acqTime'] = acqTime*1e-3
        rawData['shimming'] = shimming*1e-4
        rawData['rfExTime0'] = rfExTime0*1e-6
        rawData['rfExTime1'] = rfExTime1*1e-6
        rawData['nSteps'] = nSteps

        # Time variables in us
        echoTime *= 1e3
        repetitionTime *= 1e3
        acqTime *= 1e3

        # Rf excitation time vector
        rfTime = np.linspace(rfExTime0, rfExTime1, num=nSteps, endpoint=True) # us
        rawData['rfTime'] = rfTime*1e-6 # s

        def createSequence():

            tIni = 1000

            # Set shimming
            self.iniSequence(20, shimming)

            for repeIndex in range(nSteps):
                tEx = tIni+repetitionTime*repeIndex

                # Excitation pulse
                t0 = tEx - hw.blkTime - rfTime[repeIndex]/2
                self.rfRecPulse(t0, rfTime[repeIndex], rfExAmp, drfPhase)

                # Refocusing pulse
                t0 = tEx + echoTime/2 - hw.blkTime - rfTime[repeIndex]
                self.rfRecPulse(t0, rfTime[repeIndex]*2, rfExAmp, drfPhase+np.pi/2)

                # Rx gate
                t0 = tEx + echoTime - acqTime/2
                self.rxGate(t0, acqTime)

            # Turn off the gradients after the end of the batch
            self.endSequence(repetitionTime*nSteps)

        # Create experiment
        bw = nPoints/acqTime*hw.oversamplingFactor # MHz
        samplingPeriod = 1/bw
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1/samplingPeriod/hw.oversamplingFactor
        rawData['bw'] = bw*1e6
        acqTime = nPoints/bw

        # Execute the experiment
        createSequence()
        print('Runing...')
        rxd, msgs = self.expt.run()
        rxd['rx0'] = rxd['rx0'] * 13.788  # Here I normalize to get the result in mV
        data = sig.decimate(rxd['rx0'], hw.oversamplingFactor, ftype='fir', zero_phase=True)
        rawData['data'] = data
        name = self.saveRawData(rawData)

        # Process data to be plotted
        data = np.reshape(data, (nSteps, -1))
        data = data[:, int(nPoints/2)]

        self.data = [rfTime, data]

        return msgs

    def sequenceAnalysisGUI(self, obj):
        # Signal versus time
        timePlot = SpectrumPlot(self.data[0],
                                np.abs(self.data[1]),
                                [], [],
                                'Time (ms)', 'Signal amplitude (mV)',
                                "%s" % (self.mapVals['seqName']))

        # Spectrum
        freqPlot = SpectrumPlot(self.data[0],
                                np.abs(self.data[1]),
                                [], [],
                                'Frequency (kHz)', 'Mag FFT (a.u.)',
                                "%s" % (self.mapVals['seqName']))

        # Update figures
        obj.parent.plotview_layout.addWidget(timePlot)
        obj.parent.plotview_layout.addWidget(freqPlot)
