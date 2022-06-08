"""
Rabi map

@author:    Yolanda Vives

@summary: increase the pulse width and plot the peak value of the signal received 
@status: under development
@todo:

"""
import sys
sys.path.append('../marcos_client')
import matplotlib.pyplot as plt
#from spinEcho_standalone import spin_echo
import numpy as np
import experiment as ex
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.

class InversionRecovery(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RARE, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='InverionRecovery', val='InversionRecovery')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='ALL')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='ALL')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='ALL')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.3, field='ALL')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='ALL')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, field='ALL')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=10.0, field='ALL')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=500., field='ALL')
        self.addParameter(key='nPoints', string='nPoints (rd, ph, sl)', val=[60, 1, 1], field='ALL')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='ALL')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='ALL')
        self.addParameter(key='tInvIni', string='Ini IT (ms)', val=50.0, field='ALL')
        self.addParameter(key='tInvFin', string='Fin IT (ms)', val=1000.0, field='ALL')
        self.addParameter(key='nRepetitions', string='n Samples', val=10, field='ALL')


    def sequenceRun(self, plotSeq):
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
        larmorFreq = self.mapVals['larmoFreq']  # MHz
        rfExAmp = self.mapVals['rfExAmp']
        rfReAmp = self.mapVals['rfReAmp']
        rfExTime = self.mapVals['rfExTime']  # us
        rfReTime = self.mapVals['rfReTime']  # us
        acqTime = self.mapVals['acqTime']  # ms
        echoTime = self.mapVals['echoTime']  # ms
        repetitionTime = self.mapVals['repetitionTime']  # ms
        tInvIni = self.mapVals['tInvIni']  # s
        tInvFin = self.mapVals['tInvFin']  # s
        nRepetitions = self.mapVals['nRepetitions']  # number of samples
        nPoints = self.mapVals['nPoints'] # number of readout points
        shimming = self.mapVals['shimming']

        shimming = np.array(shimming) * 1e-4

        if rfReTime is None:
            rfReTime = 2 * rfExTime

        rfExTime = rfExTime * 1e-6
        rfReTime = rfReTime * 1e-6
        acqTime = acqTime * 1e-3
        echoTime = echoTime * 1e-3
        repetitionTime = repetitionTime*1e-3

        rawData = {}
        rawData['seqName'] = seqName
        rawData['larmorFreq'] = larmorFreq * 1e6
        rawData['rfExAmp'] = rfExAmp
        rawData['rfReAmp'] = rfReAmp
        rawData['rfExTime'] = rfExTime
        rawData['rfRetime'] = rfReTime
        rawData['repetitionTime'] = repetitionTime
        rawData['tInvIni'] = tInvIni
        rawData['tInvFin'] = tInvFin
        rawData['nRepetitions'] = nRepetitions
        rawData['acqTime'] = acqTime
        rawData['nPoints'] = nPoints
        rawData['echoTime'] = echoTime

        # Miscellaneous
        gradRiseTime = 200  # us
        crusherTime = 1000  # us
        gSteps = int(gradRiseTime / 5)
        axes = np.array([0, 1, 2])
        rawData['gradRiseTime'] = gradRiseTime
        rawData['gSteps'] = gSteps

        # Bandwidth and sampling rate
        bw = nRD / acqTime * 1e-6  # MHz
        bwov = bw * hw.oversamplingFactor
        samplingPeriod = 1 / bwov

        tIR = np.geomspace(tInvIni, tInvFin, nRepetitions)
        rawData['tIR'] = tIR

        def createSequence():
            # Set shimming
            mri.iniSequence(expt, 20, shimming)

            for repeIndex in range(nRepetitions):
                # Initialize time
                tEx = 20e3 + np.max(tIR) + repetitionTime * repeIndex

                # Inversion time for current iteration
                inversionTime = tIR[repeIndex]

                # Crusher gradient for inversion rf pulse
                #            t0 = tEx-inversionTime-crusherTime/2-gradRiseTime-hw.gradDelay-50
                #            mri.gradTrap(expt, t0, gradRiseTime, crusherTime, 0.005, gSteps, axes[0], shimming)
                #            mri.gradTrap(expt, t0, gradRiseTime, crusherTime, 0.005, gSteps, axes[1], shimming)
                #            mri.gradTrap(expt, t0, gradRiseTime, crusherTime, 0.005, gSteps, axes[2], shimming)

                # Inversion pulse
                t0 = tEx - inversionTime - hw.blkTime - rfReTime / 2
                mri.rfRecPulse(expt, t0, rfReTime, rfReAmp, 0)

                # Spoiler gradients to destroy residual transversal signal detected for ultrashort inversion times
                #            mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[0], shimming)
                #            mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[1], shimming)
                #            mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[2], shimming)

                # Excitation pulse
                t0 = tEx - hw.blkTime - rfExTime / 2
                mri.rfRecPulse(expt, t0, rfExTime, rfExAmp, 0)

                # Crusher gradient
                t0 = tEx + echoTime / 2 - crusherTime / 2 - gradRiseTime - hw.gradDelay - 50
                mri.gradTrap(expt, t0, gradRiseTime, crusherTime, 0.005, gSteps, axes[0], shimming)
                mri.gradTrap(expt, t0, gradRiseTime, crusherTime, 0.005, gSteps, axes[1], shimming)
                mri.gradTrap(expt, t0, gradRiseTime, crusherTime, 0.005, gSteps, axes[2], shimming)

                # Refocusing pulse
                t0 = tEx + echoTime / 2 - rfReTime / 2 - hw.blkTime
                mri.rfRecPulse(expt, t0, rfReTime, rfReAmp, np.pi / 2)

                # Rx gating
                t0 = tEx + echoTime - acqTime / 2
                mri.rxGate(expt, t0, acqTime)

            # End sequence
            mri.endSequence(expt, scanTime)

        # Time variables in us
        rfExTime *= 1e6
        rfReTime *= 1e6
        repetitionTime *= 1e6
        echoTime *= 1e6
        tIR *= 1e6
        scanTime = nRepetitions * repetitionTime  # us

        expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa,
                             gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = expt.get_rx_ts()[0]  # us
        bw = 1 / samplingPeriod / hw.oversamplingFactor  # MHz
        acqTime = nRD / bw  # us
        rawData['samplingPeriod'] = samplingPeriod * 1e-6
        rawData['bw'] = bw * 1e6
        createSequence()

        rxd, msgs = expt.run()

        expt.__del__()
        return rxd['rx0'], msgs


