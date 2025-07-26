"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: rare sequence class
"""

import numpy as np
import marge.controller.experiment_gui as ex
import marge.configs.hw_config as hw # Import the scanner hardware config
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from scipy.interpolate import griddata


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class PETRA(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(PETRA, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='PETRAInfo', val='PETRA')
        self.addParameter(key='toMaRGE', val=False)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=22.0, field='RF')
        self.addParameter(key='deadTime', string='TxRx dead time (us)', val=150.0, field='RF')
        self.addParameter(key='gapGtoRF', string='Gap G to RF (us)', val=100.0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=10., field='SEQ')
        self.addParameter(key='fov', string='FOV (cm)', val=[4.0, 4.0, 4.0], field='IM')
        self.addParameter(key='dfov', string='dFOV (mm)', val=[0.0, 0.0, 0.0], field='IM')
        self.addParameter(key='nPoints', string='nPoints (rd, ph, sl)', val=[30, 30, 1], field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=1.0, field='SEQ')
        self.addParameter(key='undersampling', string='Radial undersampling', val=10, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes', val=[0, 2, 1], field='IM')
        self.addParameter(key='axesEnable', string='Axes enable', val=[1, 1, 0], field='IM')
        self.addParameter(key='axesOn', string='Axes ON', val=[1, 1, 1], field='IM')
        self.addParameter(key='drfPhase', string='Phase of excitation pulse (º)', val=0.0, field='RF')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=0, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='gradRiseTime', string='Grad Rise Time (us)', val=1000, field='OTH')
        self.addParameter(key='nStepsGradRise', string='Grad steps', val=5, field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='NyquistOS', string='Radial oversampling', val=1, field='SEQ')
        self.addParameter(key='reco', string='ART->0,  FFT->1', val=1, field='IM')
        self.addParameter(key='boolGrid', string='Bool regridding', val=1, field='OTH')

    def sequenceInfo(self):
        
        print("3D PETRA sequence")
        print("Author: Jose Borreguero")
        print("Contact: pepe.morata@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain\n")


    def sequenceTime(self):
        self.sequenceRun(2)
        return self.mapVals['nScans'] * self.mapVals['repetitionTime'] * 1e-3 * self.mapVals['SequenceGradients'].shape[0] / 60

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False  # Starts the gpa
        freqCal = True  # Swich off only if you want and you are on debug mode

        seqName = self.mapVals['seqName']
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq']  # MHz
        rfExAmp = self.mapVals['rfExAmp']  #  a.u.
        rfExTime = self.mapVals['rfExTime']  # us
        gapGtoRF = self.mapVals['gapGtoRF']  # us
        deadTime = self.mapVals['deadTime']  # us
        repetitionTime = self.mapVals['repetitionTime']  # ms
        fov = np.array(self.mapVals['fov'])  # cm
        dfov = np.array(self.mapVals['dfov'])  # mm
        nPoints = np.array(self.mapVals['nPoints'])
        acqTime = self.mapVals['acqTime']  # ms
        axes = self.mapVals['axesOrientation']
        axesEnable = self.mapVals['axesEnable']
        drfPhase = self.mapVals['drfPhase']  # degrees
        dummyPulses = self.mapVals['dummyPulses']
        shimming = np.array(self.mapVals['shimming'])  # *1e4
        gradRiseTime = self.mapVals['gradRiseTime']
        nStepsGradRise = self.mapVals['nStepsGradRise']
        undersampling = self.mapVals['undersampling']
        undersampling = np.sqrt(undersampling)
        txChannel = self.mapVals['txChannel']
        rxChannel = self.mapVals['rxChannel']
        NyquistOS = self.mapVals['NyquistOS']
        boolGrid = self.mapVals['boolGrid']

        # Conversion of variables to non-multiplied units
        larmorFreq = larmorFreq*1e6
        rfExTime = rfExTime*1e-6
        gapGtoRF = gapGtoRF*1e-6
        deadTime = deadTime*1e-6
        gradRiseTime = gradRiseTime*1e-6  # s
        fov = fov*1e-2
        dfov = dfov*1e-3
        acqTime = acqTime*1e-3  # s
        shimming = shimming*1e-4
        repetitionTime= repetitionTime*1e-3  # s

        # Miscellaneous
        larmorFreq = larmorFreq*1e-6    # MHz
        resolution = fov/nPoints
        self.mapVals['resolution'] = resolution

        # Get cartesian parameters
        dK = 1 / fov
        kMax = nPoints / (2 * fov)  # m-1

        # SetSamplingParameters
        BW = (np.max(nPoints))*NyquistOS / (2 * acqTime) * 1e-6 # MHz
        samplingPeriod = 1 / BW
        self.mapVals['BW'] = BW
        self.mapVals['kMax'] = kMax
        self.mapVals['dK'] = dK

        gradientAmplitudes = kMax / (hw.gammaB * acqTime)
        if axesEnable[0] == 0:
            gradientAmplitudes[0] = 0
        if axesEnable[1] == 0:
            gradientAmplitudes[1] = 0
        if axesEnable[2] == 0:
            gradientAmplitudes[2] = 0


        nPPL = int(np.ceil((1 * acqTime - deadTime - 0.5 * rfExTime) * BW * 1e6 + 1))
        nLPC = int(np.ceil(max(nPoints[0], nPoints[1]) * np.pi / undersampling))
        nLPC = max(nLPC - (nLPC % 2), 1)
        nCir = max(int(np.ceil(nPoints[2] * np.pi / 2 / undersampling) + 1), 1)

        if axesEnable[0] == 0 or axesEnable[1] == 0 or axesEnable[2] == 0:
            nCir = 1
        if axesEnable[0] == 0 and axesEnable[1] == 0:
            nLPC = 2
        if axesEnable[0] == 0 and axesEnable[2] == 0:
            nLPC = 2
        if axesEnable[2] == 0 and axesEnable[1] == 0:
            nLPC = 2

        acqTime = nPPL / BW # us
        self.mapVals['acqTimeReal'] = acqTime * 1e-3  # ms
        self.mapVals['nPPL'] = nPPL
        self.mapVals['nLPC'] = nLPC
        self.mapVals['nCir'] = nCir

        # Get number of radial repetitions
        nRepetitions = 0
        if nCir == 1:
            theta = np.array([np.pi / 2])
        else:
            theta = np.linspace(0, np.pi, nCir)

        for jj in range(nCir):
            nRepetitions = nRepetitions + max(int(np.ceil(nLPC * np.sin(theta[jj]))), 1)
        self.mapVals['nRadialReadouts'] = nRepetitions
        self.mapVals['theta'] = theta

        # Calculate radial gradients
        normalizedGradientsRadial = np.zeros((nRepetitions, 3))
        n = -1

        # Get theta vector for current block
        if nCir == 1:
            theta = np.array([np.pi / 2])
        else:
            theta = np.linspace(0, np.pi, nCir)

        # Calculate the normalized gradients:
        for jj in range(nCir):
            nLPCjj = max(int(np.ceil(nLPC * np.sin(theta[jj]))), 1)
            deltaPhi = 2 * np.pi / nLPCjj
            phi = np.linspace(0, 2 * np.pi - deltaPhi, nLPCjj)

            for kk in range(nLPCjj):
                n += 1
                normalizedGradientsRadial[n, 0] = np.sin(theta[jj]) * np.cos(phi[kk])
                normalizedGradientsRadial[n, 1] = np.sin(theta[jj]) * np.sin(phi[kk])
                normalizedGradientsRadial[n, 2] = np.cos(theta[jj])

        # Set gradients to T/m
        gradientVectors1 = np.matmul(normalizedGradientsRadial, np.diag(gradientAmplitudes))

        # Calculate radial k-points at t = 0.5*rfExTime+td
        kRadial = []
        normalizedKRadial = np.zeros((nRepetitions, 3, nPPL))
        normalizedKRadial[:, :, 0] = (0.5 * rfExTime + deadTime + (0.5 / (BW*1e6))) * normalizedGradientsRadial
        # Calculate all k-points
        for jj in range(1, nPPL):
            normalizedKRadial[:, :, jj] = normalizedKRadial[:, :, 0] + jj* normalizedGradientsRadial / (BW*1e6)

        a = np.zeros(shape=(normalizedKRadial.shape[2], normalizedKRadial.shape[0], normalizedKRadial.shape[1]))
        a[:, :, 0] = np.transpose(np.transpose(np.transpose(normalizedKRadial[:, 0, :])))
        a[:, :, 1] = np.transpose(np.transpose(np.transpose(normalizedKRadial[:, 1, :])))
        a[:, :, 2] = np.transpose(np.transpose(np.transpose(normalizedKRadial[:, 2, :])))

        aux0reshape = np.reshape(np.transpose(a[:, :, 0]), [nRepetitions * nPPL, 1])
        aux1reshape = np.reshape(np.transpose(a[:, :, 1]), [nRepetitions * nPPL, 1])
        aux2reshape = np.reshape(np.transpose(a[:, :, 2]), [nRepetitions * nPPL, 1])

        normalizedKRadial = np.concatenate((aux0reshape, aux1reshape, aux2reshape), axis=1)
        kRadial = (np.matmul(normalizedKRadial, np.diag((hw.gammaB * gradientAmplitudes))))

        # Get cartesian kPoints
        # Get minimun time
        tMin = 0.5 * rfExTime + deadTime + 0.5 / (BW * 1e6)

        # Get the full cartesian points
        kx = np.linspace(-kMax[0] * (nPoints[0] != 1), kMax[0] * (nPoints[0] != 1), nPoints[0])
        ky = np.linspace(-kMax[1] * (nPoints[1] != 1), kMax[1] * (nPoints[1] != 1), nPoints[1])
        kz = np.linspace(-kMax[2] * (nPoints[2] != 1), kMax[2] * (nPoints[2] != 1), nPoints[2])

        kx, ky, kz = np.meshgrid(kx, ky, kz)

        kx = np.transpose(kx, (2, 0, 1))
        ky = np.transpose(ky, (2, 0, 1))
        kz = np.transpose(kz, (2, 0, 1))

        kCartesian = np.zeros(shape=(kx.shape[0] * kx.shape[1] * kx.shape[2], 3))
        kCartesian[:, 0] = np.reshape(kx, [kx.shape[0] * kx.shape[1] * kx.shape[2]])
        kCartesian[:, 1] = np.reshape(ky, [ky.shape[0] * ky.shape[1] * ky.shape[2]])
        kCartesian[:, 2] = np.reshape(kz, [kz.shape[0] * kz.shape[1] * kz.shape[2]])
        self.mapVals['kCartesian'] = kCartesian

        # Get the points that should be acquired in a time shorter than tMin
        normalizedKCartesian = np.zeros(shape=(kCartesian.shape[0], kCartesian.shape[1] + 1))

        if gradientAmplitudes[0] != 0:
            normalizedKCartesian[:, 0] = kCartesian[:, 0] / (hw.gammaB * (gradientAmplitudes[0]))
        else:
            normalizedKCartesian[:, 0] = 0

        if gradientAmplitudes[1] != 0:
            normalizedKCartesian[:, 1] = kCartesian[:, 1] / (hw.gammaB * (gradientAmplitudes[1]))
        else:
            normalizedKCartesian[:, 1] = 0

        if gradientAmplitudes[2] != 0:
            normalizedKCartesian[:, 2] = kCartesian[:, 2] / (hw.gammaB * (gradientAmplitudes[2]))
        else:
            normalizedKCartesian[:, 2] = 0

        kk = 0
        normalizedKSinglePointAux = np.zeros(shape=(kCartesian.shape[0], kCartesian.shape[1]))

        for jj in range(1, normalizedKCartesian.shape[0]):
            normalizedKCartesian[jj, 3] = np.sqrt(
                np.power(normalizedKCartesian[jj, 0], 2) + np.power(normalizedKCartesian[jj, 1], 2) + np.power(normalizedKCartesian[jj, 2], 2))

            if (normalizedKCartesian[jj, 3] < tMin):
                normalizedKSinglePointAux[kk, 0:3] = normalizedKCartesian[jj, 0:3]
                kk = kk + 1

        normalizedKSinglePoint = normalizedKSinglePointAux[0:kk, :]
        kSinglePoint = np.matmul(normalizedKSinglePoint, np.diag(hw.gammaB * gradientAmplitudes))
        kSpaceValues = np.concatenate((kRadial, kSinglePoint))
        self.mapVals['kSpaceValues'] = kSpaceValues

        # Set gradients for cartesian sampling
        gradientVectors2 = kSinglePoint / (hw.gammaB * tMin)
        MaxSPGradTransitions = kMax / (hw.gammaB * acqTime)
        MaxSPGradTransitions[0] = max(gradientVectors2[:, 0])
        MaxSPGradTransitions[1] = max(gradientVectors2[:, 1])
        MaxSPGradTransitions[2] = max(gradientVectors2[:, 2])

        gSeq = - np.concatenate((gradientVectors1, gradientVectors2), axis=0)
        gSeqDif = np.diff(gSeq, n=1, axis=0)
        MaxGradTransitions = kMax / (hw.gammaB * acqTime)
        MaxGradTransitions[0] = max(gSeqDif[:, 0])
        MaxGradTransitions[1] = max(gSeqDif[:, 1])
        MaxGradTransitions[2] = max(gSeqDif[:, 2])

        print(gradientVectors1.shape[0], " radial lines and ", gradientVectors2.shape[0], " pointwise")
        print("Radial max gradient strengths are  ", gradientAmplitudes * 1e3, " mT/m")
        print("Pointwise max gradient strengths are  ", MaxSPGradTransitions * 1e3, " mT/m")
        print("Max grad transitions are  ", MaxGradTransitions * 1e3, " mT/m")

        self.mapVals['SequenceGradients'] = gSeq
        self.mapVals['nSPReadouts'] = gradientVectors2.shape[0]

        def createSequence():
            nRep = gSeq.shape[0]
            Grisetime = gradRiseTime * 1e6
            tr = repetitionTime * 1e6
            delayGtoRF = gapGtoRF * 1e6
            RFpulsetime = rfExTime * 1e6
            axesOn=self.mapVals['axesOn']
            TxRxtime = deadTime * 1e6
            repeIndex = 0
            ii = 1
            tInit = 20
            # Set shimming
            self.iniSequence(tInit, shimming)

            for ii in range(dummyPulses):
                tdummy = tInit + tr * (ii + 1) + Grisetime + delayGtoRF
                self.rfRecPulse(tdummy, RFpulsetime, rfExAmp, drfPhase * np.pi / 180, channel=txChannel)

            tInit = tInit + tr*dummyPulses

            while repeIndex < nRep:
                # Initialize time
                t0 = tInit + tr * (repeIndex + 1)

                # Set gradients
                if repeIndex == 0:
                    ginit = np.array([0, 0, 0])
                    self.setGradientRamp(t0, Grisetime, nStepsGradRise, ginit[0], gSeq[0, 0]*axesOn[0], axes[0], shimming)
                    self.setGradientRamp(t0, Grisetime, nStepsGradRise, ginit[1], gSeq[0, 1]*axesOn[1], axes[1], shimming)
                    self.setGradientRamp(t0, Grisetime, nStepsGradRise, ginit[2], gSeq[0, 2]*axesOn[2], axes[2], shimming)
                elif repeIndex > 0:
                    if gSeq[repeIndex-1, 0] != gSeq[repeIndex, 0]:
                        self.setGradientRamp(t0, Grisetime, nStepsGradRise, gSeq[repeIndex-1, 0]*axesOn[0], gSeq[repeIndex, 0]*axesOn[0], axes[0], shimming)
                    if gSeq[repeIndex-1, 1] != gSeq[repeIndex, 1]:
                        self.setGradientRamp(t0, Grisetime, nStepsGradRise, gSeq[repeIndex-1, 1]*axesOn[1], gSeq[repeIndex, 1]*axesOn[1], axes[1], shimming)
                    if gSeq[repeIndex-1, 2] != gSeq[repeIndex, 2]:
                        self.setGradientRamp(t0, Grisetime, nStepsGradRise, gSeq[repeIndex-1, 2]*axesOn[2], gSeq[repeIndex, 2]*axesOn[2], axes[2], shimming)

                # Excitation pulse
                trf0 = t0 + Grisetime + delayGtoRF
                self.rfRecPulse(trf0, RFpulsetime, rfExAmp, drfPhase * np.pi / 180)

                if repeIndex < gradientVectors1.shape[0]:
                    tACQ = acqTimeSeq
                if repeIndex >= gradientVectors1.shape[0]:
                    tACQ = 1 / BWreal

                # Rx gate
                t0rx = trf0 + hw.blkTime + RFpulsetime + TxRxtime
                self.rxGateSync(t0rx, tACQ)

                if repeIndex == nRep-1:
                    self.endSequence(tInit + (nRep+1) * tr)

                repeIndex = repeIndex + 1
                ii = ii + 1

        # Calibrate frequency
        if freqCal and (not plotSeq):
            # larmorFreq = self.freqCalibration(bw=0.05)
            # larmorFreq = self.freqCalibration(bw=0.005)
            drfPhase = self.mapVals['drfPhase']

        # Create full sequence
        # Run the experiment
        overData = []
        if plotSeq == 0 or plotSeq == 1:
            self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
            samplingPeriod = self.expt.getSamplingRate()
            BWreal = 1 / samplingPeriod
            acqTimeSeq = nPPL / BWreal  # us
            self.mapVals['BWSeq'] = BWreal * 1e6  # Hz
            self.mapVals['acqTimeSeq'] = acqTimeSeq * 1e-6  # s
            createSequence()
            if self.floDict2Exp():
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("ERROR: sequence waveforms out of hardware bounds")
                return False

            tRadio = np.linspace(deadTime + 0.5 / (self.mapVals['BWSeq']),
                                 deadTime + 0.5 / (self.mapVals['BWSeq']) + self.mapVals['acqTimeSeq'], nPPL)
            tVectorRadial2 = []
            for pp in range(0, self.mapVals['nRadialReadouts']):
                tVectorRadial2 = np.concatenate((tVectorRadial2, tRadio), axis=0)
            self.mapVals['tVectorRadial2'] = tVectorRadial2

            tPoint = np.linspace(deadTime + 0.5 / (self.mapVals['BWSeq']), deadTime + 0.5 / (self.mapVals['BWSeq']), 1)
            tVectorSP = []
            for pp in range(0, self.mapVals['nSPReadouts']):
                tVectorSP = np.concatenate((tVectorSP, tPoint), axis=0)
            self.mapVals['tVectorSP'] = tVectorSP


            if plotSeq == 0:
                # Warnings before run sequence
                if axes[0] == axes[1] or axes[0] == axes[2] or axes[2] == axes[1]:
                    print("Two different gradient coils has been introduced as the same")
                if gradRiseTime + gapGtoRF + rfExTime + deadTime + acqTimeSeq*1e-6 >= repetitionTime:
                    print("So short TR")

                # Run all scans
                for ii in range(nScans):
                    rxd, msgs = self.expt.run()
                    rxd['rx0'] = rxd['rx0']  # mV
                    print(ii, "/", nScans, "PETRA sequence finished")
                    # Get data
                    overData = np.concatenate((overData, rxd['rx0']), axis=0)

                # Decimate the result
                overData = np.reshape(overData, (nScans, -1))
                radPoints = gradientVectors1.shape[0]*(nPPL+2*hw.addRdPoints)*hw.oversamplingFactor
                carPoints = gradientVectors2.shape[0]*(1+2*hw.addRdPoints)*hw.oversamplingFactor
                overDataRad = np.reshape(overData[:, 0:radPoints], -1)
                overDataCar = np.reshape(overData[:, radPoints: radPoints+carPoints], -1)
                fullDataRad = self.decimate(overDataRad, nScans*gradientVectors1.shape[0], option='PETRA')
                fullDataCar = self.decimate(overDataCar, nScans*gradientVectors2.shape[0], option='PETRA')

                # Average results
                RadialSampledPointsRaw = np.average(np.reshape(fullDataRad, (nScans, -1)), axis=0)
                CartesianSampledPointsRaw = np.average(np.reshape(fullDataCar, (nScans, -1)), axis=0)

                RadialSampledPointsReshaped = np.reshape(RadialSampledPointsRaw, (gradientVectors1.shape[0], nPPL))
                RadialSampledList = np.reshape(RadialSampledPointsReshaped, (nPPL*gradientVectors1.shape[0], 1))

                CartesianSampledPointsReshaped = np.reshape(CartesianSampledPointsRaw, (gradientVectors2.shape[0], 1))
                CartesianSampledList = np.reshape(CartesianSampledPointsReshaped, (1*gradientVectors2.shape[0], 1))

                signalPoints = np.concatenate((RadialSampledList, CartesianSampledList), axis=0)
                kSpace = np.concatenate((kSpaceValues, signalPoints, signalPoints.real, signalPoints.imag), axis=1)
                self.mapVals['kSpaceRaw'] = kSpace

                if nCir > 1:
                    kxOriginal = np.reshape(np.real(kSpace[:, 0]), -1)
                    kyOriginal = np.reshape(np.real(kSpace[:, 1]), -1)
                    kzOriginal = np.reshape(np.real(kSpace[:, 2]), -1)
                    kxTarget = np.reshape(kCartesian[:, 0], -1)
                    kyTarget = np.reshape(kCartesian[:, 1], -1)
                    kzTarget = np.reshape(kCartesian[:, 2], -1)
                    if boolGrid == 0:
                        valCartesian = 1
                    else:
                        valCartesian = griddata((kxOriginal, kyOriginal, kzOriginal), np.reshape(kSpace[:, 3], -1),(kxTarget, kyTarget, kzTarget), method='linear', fill_value=0, rescale=False)
                    DELX = dfov[0]
                    DELY = dfov[1]
                    DELZ = dfov[2]
                    phase = np.exp(-2 * np.pi * 1j * (DELX * kCartesian[:, 0] + DELY * kCartesian[:, 1] + DELZ * kCartesian[:, 2]))
                    valCartesian = valCartesian * phase

                if (nCir == 1) and (nLPC > 2):
                    kxOriginal = np.reshape(np.real(kSpace[:, 0]), -1)
                    kyOriginal = np.reshape(np.real(kSpace[:, 1]), -1)
                    kxTarget = np.reshape(kCartesian[:, 0], -1)
                    kyTarget = np.reshape(kCartesian[:, 1], -1)
                    if boolGrid == 0:
                        valCartesian = 1
                    else:
                        valCartesian = griddata((kxOriginal, kyOriginal), np.reshape(kSpace[:, 3], -1),(kxTarget, kyTarget), method='linear', fill_value=0, rescale=False)
                    DELX = dfov[0]
                    DELY = dfov[1]
                    phase = np.exp(-2 * np.pi * 1j * (DELX * kCartesian[:, 0] + DELY * kCartesian[:, 1]))
                    valCartesian = valCartesian * phase

                if (nCir == 1) and (nLPC == 2):
                    kxOriginal = np.reshape(np.real(kSpace[:, 0]), -1)
                    kxTarget = np.reshape(kCartesian[:, 0], -1)
                    valCartesian = griddata((kxOriginal), np.reshape(kSpace[:, 3], -1), (kxTarget), method='linear',fill_value=0, rescale=False)
                    self.valCartesian = valCartesian
                    DELX = dfov[0]
                    DELY = dfov[1]
                    DELZ = dfov[2]
                    phase = np.exp(
                        -2 * np.pi * 1j * (DELX * kCartesian[:, 0] + DELY * kCartesian[:, 1] + DELZ * kCartesian[:, 2]))
                    valCartesian = valCartesian * phase

                kSpaceCartesian = np.zeros((kCartesian.shape[0], 6))
                kSpaceCartesian[:, 0] = kCartesian[:, 0]
                kSpaceCartesian[:, 1] = kCartesian[:, 1]
                kSpaceCartesian[:, 2] = kCartesian[:, 2]
                kSpaceCartesian[:, 3] = abs(valCartesian)
                kSpaceCartesian[:, 4] = valCartesian.real
                kSpaceCartesian[:, 5] = valCartesian.imag
                kSpaceArray = np.reshape(valCartesian, (nPoints[2], nPoints[1], nPoints[0]))
                ImageFFT = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpaceArray)))
                self.mapVals['kSpaceCartesian'] = kSpaceCartesian
                self.mapVals['kSpaceArray'] = kSpaceArray
                self.mapVals['ImageFFT'] = ImageFFT
            self.expt.__del__()

        return True

    def sequenceAnalysis(self, obj=''):
        axesEnable = self.mapVals['axesEnable']
        kSpace = self.mapVals['kSpaceArray']
        image = self.mapVals['ImageFFT']
        axes = self.mapVals['axesOrientation']
        reco = self.mapVals['reco']

        if reco == 0:
            niter = 1
            update = 1
            fov = self.mapVals['fov']/ 100
            nPoints = self.mapVals['nPoints']
            sampled_Kspace = self.mapVals['kSpaceRaw']
            kS = np.array(sampled_Kspace[:, 0:3].real)
            signal = sampled_Kspace[:, 3]
            tSTEPS = signal.shape[0]
            FoVx = fov[0]
            FoVy = fov[1]
            FoVz = fov[2]
            NX = nPoints[0]
            NY = nPoints[1]
            NZ = nPoints[2]
            dx2 = FoVx / NX
            dy2 = FoVy / NY
            dz2 = FoVz / NZ
            nn = NX * NY * NZ
            RHO = np.zeros((1, nn))
            kk = np.arange(NZ)
            jj = np.arange(NY)
            ii = np.arange(NX)
            count = 1

            for n in range(niter):
                for tt in range(tSTEPS):
                    Mtk = np.reshape(np.array(np.exp(-1*1j * (2 * np.pi * kS[tt, 2] * (-(NZ - 1) / 2 + kk) * dz2))), [1, NZ])
                    Mtj = np.reshape(np.array(np.exp(-1*1j * (2 * np.pi * kS[tt, 1] * (-(NY - 1) / 2 + jj) * dy2))), [NY, 1])
                    Mtjk = np.reshape(np.matmul(Mtj, Mtk), [1, NY*NZ])
                    aux = np.reshape(np.exp(-1*1j * (2 * np.pi * kS[tt, 0] * (-(NX - 1) / 2 + ii) * dx2)), [NX, 1])
                    Mt = np.reshape(np.matmul(aux, Mtjk), [1, NX*NY*NZ])
                    delta_t = (signal[tt] - np.sum(np.matmul(np.reshape(Mt, [NX*NY*NZ, 1]), RHO))) / np.dot(np.squeeze(np.asarray(Mt)), np.squeeze(np.asarray(Mt)))
                    RHO = (RHO + update * delta_t * np.conj(Mt))
                    count = count + 1
                    print(tt, "/", tSTEPS, " ART")

            if NZ == 1:
                image = np.reshape(RHO, [NX, NY])
            if NZ > 1:
                image = np.reshape(RHO, [NX, NY, NZ])

        if axes[0] == 0 and axes[1] == 2:
            axislegend = ['Z', 'X']
        if axes[0] == 0 and axes[1] == 1:
            axislegend = ['Y', 'X']
        if axes[0] == 1 and axes[1] == 0:
            axislegend = ['X', 'Y']
        if axes[0] == 1 and axes[1] == 2:
            axislegend = ['Z', 'Y']
        if axes[0] == 2 and axes[1] == 0:
            axislegend = ['X', 'Z']
        if axes[0] == 2 and axes[1] == 1:
            axislegend = ['Y', 'Z']

        if axesEnable[1] == 0 and axesEnable[2] == 0:
            k = (self.mapVals['kSpaceCartesian'][:, 0])
            signal = self.mapVals['kSpaceCartesian']
            timesignal = np.linspace(0,self.mapVals['acqTime'], self.mapVals['nPoints'][0])
            pos = np.linspace(-self.mapVals['fov'][0]/2, self.mapVals['fov'][0]/2, self.mapVals['nPoints'][0])

            # Plots to show into the GUI
            result1 = {}
            result1['widget'] = 'curve'
            result1['xData'] = timesignal
            result1['yData'] = [signal[:, 3], signal[:, 4], signal[:, 5]]
            result1['xLabel'] = 'Time (ms)'
            result1['yLabel'] = 'Signal amplitude (mV)'
            result1['title'] = "Signal"
            result1['legend'] = ['Magnitude', 'Real', 'Imaginary']
            result1['row'] = 0
            result1['col'] = 0

            result2 = {}
            result2['widget'] = 'curve'
            result2['xData'] = pos
            result2['yData'] = [np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(self.valCartesian))))]
            result2['xLabel'] = 'Position (cm)'
            result2['yLabel'] = "Amplitude (a.u.)"
            result2['title'] = "Spectrum"
            result2['legend'] = ['G=0']
            result2['row'] = 1
            result2['col'] = 0

            self.output = [result1, result2]
            
        else:
            if self.axesOrientation[2] == 2:  # Sagittal
                title = "Sagittal"
                if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 1:  # OK
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    xLabel = "(-Y) A | PHASE | P (+Y)"
                    yLabel = "(-X) I | READOUT | S (+X)"
                else:
                    image = np.transpose(image, (0, 2, 1))
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    xLabel = "(-Y) A | READOUT | P (+Y)"
                    yLabel = "(-X) I | PHASE | S (+X)"
            elif self.axesOrientation[2] == 1:  # Coronal
                title = "Coronal"
                if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 2:  # OK
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    image = np.flip(image, axis=0)
                    xLabel = "(+Z) R | PHASE | L (-Z)"
                    yLabel = "(-X) I | READOUT | S (+X)"
                else:
                    image = np.transpose(image, (0, 2, 1))
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    image = np.flip(image, axis=0)
                    xLabel = "(+Z) R | READOUT | L (-Z)"
                    yLabel = "(-X) I | PHASE | S (+X)"
            elif self.axesOrientation[2] == 0:  # Transversal
                title = "Transversal"
                if self.axesOrientation[0] == 1 and self.axesOrientation[1] == 2:
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    xLabel = "(+Z) R | PHASE | L (-Z)"
                    yLabel = "(+Y) P | READOUT | A (-Y)"
                else:  # OK
                    image = np.transpose(image, (0, 2, 1))
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    xLabel = "(+Z) R | READOUT | L (-Z)"
                    yLabel = "(+Y) P | PHASE | A (-Y)"

            result1 = {}
            result1['widget'] = 'image'
            result1['data'] = np.abs(image)
            result1['xLabel'] = axislegend[0]
            result1['yLabel'] = axislegend[1]
            result1['title'] = "Image magnitude"
            result1['row'] = 0
            result1['col'] = 0

            result2 = {}
            result2['widget'] = 'image'
            result2['data'] = np.abs(kSpace)
            result2['xLabel'] = "k"
            result2['yLabel'] = "k"
            result2['title'] = "k-Space"
            result2['row'] = 0
            result2['col'] = 1

            self.output = [result1, result2]

        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()
            
        return self.output


# if __name__=='__main__':
#     seq = PETRA()
#     seq.sequenceRun()