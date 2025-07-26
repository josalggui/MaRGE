"""
@author: T. Guallart Naval, november 2021
@modifield: J.M. Algarín, february 25th 2022
@modified: J.M. Algarín, june 8th 2022, adapted to new gui structure
MRILAB @ I3M
@summary: spin echo with inversion recovery where we sweep the time between the IR pulse and the excitation pulse.
"""


import marge.marcos.marcos_client.experiment
import numpy as np
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import marge.configs.hw_config as hw

class InversionRecovery(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(InversionRecovery, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='InverionRecoveryInfo', val='InversionRecovery')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, field='RF')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=10.0, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=2000., field='SEQ')
        self.addParameter(key='nPoints', string='nPoints', val=60, field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='crusherAmp', string='Crusher grad amp (mT/m)', val=5.0, field='OTH')
        self.addParameter(key='crusherTime', string='Crusher grad time (us)', val=100.0, field='OTH')
        self.addParameter(key='crusherDelay', string='Crusher grad delay (us)', val=0.0, field='OTH')
        self.addParameter(key='tInv0', string='Inversion time, Start (ms)', val=50.0, field='SEQ')
        self.addParameter(key='tInv1', string='Inversion time, End (ms)', val=1000.0, field='SEQ')
        self.addParameter(key='nSteps', string='Number of steps', val=10, field='SEQ')

    def sequenceInfo(self):
        
        print("Inversion Recovery")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs Inversion Recovery and sweep the inversion time\n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        return(repetitionTime*nScans/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq, demo=False):
        init_gpa = False  # Starts the gpa

        # Create the inputs automatically. For some reason it only works if there is a few code later...
        # for key in self.mapKeys:
        #     locals()[key] = self.mapVals[key]
        #     if not key in locals():
        #         print('Error')
        #         locals()[key] = self.mapVals[key]

        # Create the inputs manually, pufff
        seqName = self.mapVals['seqName']
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq']  # MHz
        rfExAmp = self.mapVals['rfExAmp']
        rfReAmp = self.mapVals['rfReAmp']
        rfExTime = self.mapVals['rfExTime']  # us
        rfReTime = self.mapVals['rfReTime']  # us
        acqTime = self.mapVals['acqTime']  # ms
        echoTime = self.mapVals['echoTime']  # ms
        repetitionTime = self.mapVals['repetitionTime']  # ms
        tInv0 = self.mapVals['tInv0']  # ms
        tInv1 = self.mapVals['tInv1']  # ms
        nSteps = self.mapVals['nSteps']  # number of samples
        nPoints = self.mapVals['nPoints'] # number of readout points
        shimming = self.mapVals['shimming']
        crusherAmp = self.mapVals['crusherAmp'] * 1e-3
        crusherTime = self.mapVals['crusherTime']
        crusherDelay = self.mapVals['crusherDelay']

        # Parameters in fundamental units
        rfExTime = rfExTime * 1e-6
        rfReTime = rfReTime * 1e-6
        acqTime = acqTime * 1e-3
        echoTime = echoTime * 1e-3
        repetitionTime = repetitionTime*1e-3
        shimming = np.array(shimming) * 1e-4
        crusherTime = crusherTime*1e-6
        crusherDelay = crusherDelay*1e-6
        tInv0 = tInv0*1e-3
        tInv1 = tInv1*1e-3

        # Miscellaneous
        gradRiseTime = 200  # us
        gSteps = int(gradRiseTime / 5)
        axes = np.array([0, 1, 2])
        self.mapVals['gradRiseTime'] = gradRiseTime
        self.mapVals['gSteps'] = gSteps

        # Inversion time vector
        irTimeVector = np.geomspace(tInv0, tInv1, nSteps)
        self.mapVals['irTimeVector'] = irTimeVector

        def createSequence():
            # Set shimming
            self.iniSequence(20, shimming)

            for repeIndex in range(nSteps):
                # Inversion time for current iteration
                inversionTime = irTimeVector[repeIndex]

                # Initialize time
                tEx = 20e3 + np.max(irTimeVector) + repetitionTime * repeIndex

                # Inversion pulse
                t0 = tEx-inversionTime-hw.blkTime-rfReTime/2
                self.rfRecPulse(t0, rfReTime, rfReAmp, 0)

                # Spoiler gradients to destroy residual transversal signal detected for ultrashort inversion times
                t0 = tEx-inversionTime+rfReTime/2
                self.gradTrap(t0, gradRiseTime, inversionTime*0.5, crusherAmp, gSteps, axes[0], shimming)
                self.gradTrap(t0, gradRiseTime, inversionTime*0.5, crusherAmp, gSteps, axes[1], shimming)
                self.gradTrap(t0, gradRiseTime, inversionTime*0.5, crusherAmp, gSteps, axes[2], shimming)

                # Excitation pulse
                t0 = tEx-hw.blkTime-rfExTime/2
                self.rfRecPulse(t0, rfExTime, rfExAmp, 0)

                # Crusher gradient
                t0 = tEx + echoTime / 2 - crusherTime / 2 - gradRiseTime - hw.gradDelay - crusherDelay
                self.gradTrap(t0, gradRiseTime, crusherTime, crusherAmp, gSteps, axes[0], shimming)
                self.gradTrap(t0, gradRiseTime, crusherTime, crusherAmp, gSteps, axes[1], shimming)
                self.gradTrap(t0, gradRiseTime, crusherTime, crusherAmp, gSteps, axes[2], shimming)

                # Refocusing pulse
                t0 = tEx + echoTime / 2 - rfReTime / 2 - hw.blkTime
                self.rfRecPulse(t0, rfReTime, rfReAmp, np.pi / 2)

                # Rx gating
                t0 = tEx + echoTime - acqTime / 2
                self.rxGate(t0, acqTime)

            # End sequence
            self.endSequence(scanTime)

        # Time variables in us
        rfExTime *= 1e6
        rfReTime *= 1e6
        repetitionTime *= 1e6
        echoTime *= 1e6
        irTimeVector *= 1e6
        scanTime = nSteps*repetitionTime  # us
        crusherDelay *= 1e6
        crusherTime *= 1e6
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
        if self.floDict2Exp():
            print("Sequence waveforms loaded successfully")
            pass
        else:
            print("ERROR: sequence waveforms out of hardware bounds")
            return False
        if plotSeq:
            self.expt.__del__()
        else:
            rxd, msgs = self.expt.run()
            print(msgs)
            data = sig.decimate(rxd['rx0']*hw.adcFactor, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['data'] = data
            self.expt.__del__()

            # Process data to be plotted
            data = np.reshape(data, (nSteps, -1))
            data = data[:, int(nPoints / 2)]
            self.data = [irTimeVector*1e-3, data]
            self.mapVals['sampledPoint'] = data
        return True

    def sequenceAnalysis(self, obj=''):

        # Signal vs inverion time
        result1 = {'widget': 'curve',
                   'xData': self.data[0],
                   'yData': [np.abs(self.data[1])],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': '',
                   'legend': [''],
                   'row': 0,
                   'col': 0}

        # create self.out to run in iterative mode
        self.output = [result1]

        self.saveRawData()

        return self.output


