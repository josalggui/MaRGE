"""
@author: T. Guallart
@author: J.M. Algarín

@summary: increase the pulse width and plot the peak value of the signal received 
@status: under development
@todo:

"""
import experiment as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw
from plotview.spectrumplot import SpectrumPlot

class RabiFlops(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RabiFlops, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='RabiFlopsInfo', val='RabiFlops')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=0.0, field='RF')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=10.0, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=500., field='SEQ')
        self.addParameter(key='nPoints', string='nPoints', val=60, field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (*1e-4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='rfExTime0', string='Rf pulse time, Start (us)', val=5.0, field='RF')
        self.addParameter(key='rfExTime1', string='RF pulse time, End (us)', val=100.0, field='RF')
        self.addParameter(key='nSteps', string='Number of steps', val=20, field='RF')
        self.addParameter(key='deadTime', string='Dead time (us)', val=60, field='SEQ')
        self.addParameter(key='rfRefPhase', string='Refocusing phase (degrees)', val=0.0, field='RF')
        self.addParameter(key='method', string='Rephasing method: 0->Amp, 1->Time', val=0, field='RF')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=0, field='SEQ')

    def sequenceInfo(self):
        print(" ")
        print("Rabi Flops")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Rabi Flops with different methods")
        print("Notes:")
        print("Set RF refocusing amplitude to 0.0 to get single excitation behavior")
        print("Set RF refocusing time to 0.0 to auto set the RF refocusing time:")
        print("-If Rephasing method = 0, refocusing amplitude is twice the excitation amplitude")
        print("-If Rephasing method = 1, refocusing time is twice the excitation time")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        nSteps = self.mapVals['nSteps']
        dummyPulses = self.mapVals['dummyPulses']
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        return(repetitionTime*nScans*nSteps*(dummyPulses+1)/60)  # minutes, scanTime

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
        rfReAmp = self.mapVals['rfReAmp']
        echoTime = self.mapVals['echoTime']
        repetitionTime = self.mapVals['repetitionTime']
        nPoints = self.mapVals['nPoints']
        acqTime = self.mapVals['acqTime']
        shimming = np.array(self.mapVals['shimming'])
        rfExTime0 = self.mapVals['rfExTime0']
        rfExTime1 = self.mapVals['rfExTime1']
        nSteps = self.mapVals['nSteps']
        deadTime = self.mapVals['deadTime']
        rfRefPhase = self.mapVals['rfRefPhase']
        method = self.mapVals['method']
        rfReTime = self.mapVals['rfReTime']
        dummyPulses = self.mapVals['dummyPulses']

        # Time variables in us
        echoTime *= 1e3
        repetitionTime *= 1e3
        acqTime *= 1e3
        shimming = shimming * 1e-4

        # Rf excitation time vector
        rfTime = np.linspace(rfExTime0, rfExTime1, num=nSteps, endpoint=True) # us
        self.mapVals['rfTime'] = rfTime*1e-6 # s

        def createSequence():
            # Set shimming
            self.iniSequence(20, shimming)

            tEx = 1000 # First excitation at 1 ms
            for scan in range(nScans):
                for step in range(nSteps):
                    for pulse in range(dummyPulses+1):
                        # Excitation pulse
                        t0 = tEx - hw.blkTime - rfTime[step]/2
                        self.rfRecPulse(t0, rfTime[step], rfExAmp, 0)

                        # Rx gate for FID
                        if pulse==dummyPulses:
                            t0 = tEx + rfTime[step] / 2 + deadTime
                            self.rxGate(t0, acqTime)

                        # Refocusing pulse
                        if method:  # time
                            if rfReTime==0.0:
                                t0 = tEx + echoTime/2 - hw.blkTime - rfTime[step]
                                self.rfRecPulse(t0, rfTime[step]*2, rfReAmp, rfRefPhase*np.pi/180.0)
                            else:
                                t0 = tEx + echoTime/2 - hw.blkTime - rfReTime/2
                                self.rfRecPulse(t0, rfReTime, rfReAmp, rfRefPhase * np.pi / 180.0)
                        else:   # amplitude
                            if rfReTime==0.0:
                                t0 = tEx + echoTime / 2 - hw.blkTime - rfTime[step]/2
                                self.rfRecPulse(t0, rfTime[step], rfExAmp * 2, rfRefPhase * np.pi / 180.0)
                            else:
                                t0 = tEx + echoTime / 2 - hw.blkTime - rfReTime
                                self.rfRecPulse(t0, rfReTime, rfExAmp * 2, rfRefPhase * np.pi / 180.0)

                        # Rx gate for Echo
                        if pulse==dummyPulses:
                            t0 = tEx + echoTime - acqTime/2
                            self.rxGate(t0, acqTime)

                        # Update exitation time for next repetition
                        tEx += repetitionTime

            # Turn off the gradients after the end of the batch
            self.endSequence(repetitionTime*nSteps*nScans*(dummyPulses+1))

        # Create experiment
        bw = nPoints/acqTime*hw.oversamplingFactor # MHz
        samplingPeriod = 1/bw
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa,
                                  gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1 / samplingPeriod / hw.oversamplingFactor
        self.mapVals['bw'] = bw * 1e6
        acqTime = nPoints / bw

        # Execute the experiment
        createSequence()
        if not plotSeq:
            print('Runing...')
            rxd, msgs = self.expt.run()
            self.expt.__del__()
            rxd['rx0'] = rxd['rx0'] * 13.788  # Here I normalize to get the result in mV
            self.mapVals['dataOversampled'] = rxd['rx0']
        return 0

    def sequenceAnalysis(self, obj=''):
        nScans = self.mapVals['nScans']
        nSteps = self.mapVals['nSteps']
        nPoints = self.mapVals['nPoints']
        dataOversampled = self.mapVals['dataOversampled']
        timeVector = self.mapVals['rfTime']

        # Get FID and Echo
        dataFull = sig.decimate(dataOversampled, hw.oversamplingFactor, ftype='fir', zero_phase=True)
        self.mapVals['dataFull'] = dataFull
        dataFull = np.reshape(dataFull, (nScans, nSteps, 2, -1))
        dataFID = dataFull[:, :, 0, :]
        self.mapVals['dataFID'] = dataFID
        dataEcho = dataFull[:, :, 1, :]
        self.mapVals['dataEcho'] = dataEcho
        dataFIDAvg = np.mean(dataFID, axis=0)
        self.mapVals['dataFIDAvg'] = dataFIDAvg
        dataEchoAvg = np.mean(dataEcho, axis=0)
        self.mapVals['dataEchoAvg'] = dataEchoAvg

        rabiFID = dataFIDAvg[:, 5]
        self.mapVals['rabiFID'] = rabiFID
        rabiEcho = dataEchoAvg[:, np.int(nPoints/2)]
        self.mapVals['rabiEcho'] = rabiEcho

        self.saveRawData()

        # Signal vs rf time
        rabiFIDWidget = SpectrumPlot(xData=timeVector*1e6,
                                  yData=[np.abs(rabiFID), np.real(rabiFID),np.imag(rabiFID)],
                                  legend=['abs', 'real', 'imag'],
                                  xLabel='Time (ms)',
                                  yLabel='Signal amplitude (mV)',
                                  title='Rabi Flops with FID')

        rabiEchoWidget = SpectrumPlot(xData=timeVector*1e6,
                                  yData=[np.abs(rabiEcho), np.real(rabiEcho),np.imag(rabiEcho)],
                                  legend=['abs', 'real', 'imag'],
                                  xLabel='Time (ms)',
                                  yLabel='Signal amplitude (mV)',
                                  title='Rabi Flops with Spin Echo')

        return([rabiFIDWidget, rabiEchoWidget])