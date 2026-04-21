"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: rare sequence class
"""

import numpy as np
import controller.experiment_gui as ex
import configs.hw_config as hw  # Import the scanner hardware config
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from skimage.restoration import unwrap_phase as unwrap
from sklearn.preprocessing import PolynomialFeatures
from numpy.linalg import lstsq


# *********************************************************************************
# *********************************************************************************
# *********************************************************************************

class SPRITE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(SPRITE, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='SPRITEInfo', val='SPRITE')
	self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.2, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=10.0, field='RF')
        self.addParameter(key='deadTime', string='TxRx dead time (us)', val=250.0, field='RF')
        self.addParameter(key='gapGtoRF', string='Gap G to RF (us)', val=7000.0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=10., field='SEQ')
        self.addParameter(key='fov', string='FOV (cm)', val=[30.0, 30.0, 30.0], field='IM')
        self.addParameter(key='dfov', string='dFOV (mm)', val=[0.0, 0.0, 0.0], field='IM')
        self.addParameter(key='nPoints', string='nPoints (rd, ph, sl)', val=[30, 30, 1], field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=0.005, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes', val=[0, 1, 2], field='IM')
        self.addParameter(key='axesOn', string='Axes ON', val=[1, 1, 1], field='IM')
        self.addParameter(key='drfPhase', string='Phase of excitation pulse (º)', val=0.0, field='RF')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=20, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0, 0, 0], field='OTH')
        self.addParameter(key='gradRiseTime', string='Grad Rise Time (us)', val=1000, field='OTH')
        self.addParameter(key='nStepsGradRise', string='Grad steps', val=5, field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='rectPremphasis', string='Premphasis Enable', val=1, field='RF', tip="0=NO, 1=Yes")
        self.addParameter(key='factorV0', string='Factor V0', val=4.0, field='RF')
        self.addParameter(key='factorV2', string='Factor V2', val=3.0, field='RF')
        self.addParameter(key='durationPulse0', string='Duration Pulse 0 (us)', val=2.0, field='RF')
        self.addParameter(key='durationPulse2', string='Duration Pulse 2 (us)', val=3.0, field='RF')
        self.addParameter(key='interpOrder', string='Zero Padding Order', val=3, field='IM', tip='Zero Padding Order')



    def sequenceInfo(self):

        print("3D SPRITE sequence")
        print("Author: Jose Borreguero")
        print("Contact: pepe.morata@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain\n")

    def sequenceTime(self):
        self.sequenceRun(2)
        return self.mapVals['nScans'] * self.mapVals['repetitionTime'] * 1e-3 * 2 * self.mapVals['nSPReadouts'] / 60

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False  # Starts the gpa
        freqCal = True  # Swich off only if you want and you are on debug mode

        seqName = self.mapVals['seqName']
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq']  # MHz
        rfExAmp = self.mapVals['rfExAmp']  # a.u.

        gapGtoRF = self.mapVals['gapGtoRF']  # us
        deadTime = self.mapVals['deadTime'] # us
        repetitionTime = self.mapVals['repetitionTime']  # ms
        fov = np.array(self.mapVals['fov'])  # cm
        dfov = np.array(self.mapVals['dfov'])  # mm
        nPoints = np.array(self.mapVals['nPoints'])
        acqTime = self.mapVals['acqTime']  # ms
        axes = self.mapVals['axesOrientation']
        drfPhase = self.mapVals['drfPhase']  # degrees
        dummyPulses = self.mapVals['dummyPulses']
        shimming = np.array(self.mapVals['shimming'])  # *1e4
        gradRiseTime = self.mapVals['gradRiseTime']
        nStepsGradRise = self.mapVals['nStepsGradRise']
        txChannel = self.mapVals['txChannel']
        rectPremphasis = self.mapVals['rectPremphasis']
        factorV0 = self.mapVals['factorV0']
        factorV2 = self.mapVals['factorV2']
        durationPulse0 = self.mapVals['durationPulse0']
        durationPulse2 = self.mapVals['durationPulse2']

        if rectPremphasis == 0:
            rfExTime = self.mapVals['rfExTime']  # us
        if rectPremphasis == 1:
            rfExTime = self.mapVals['rfExTime'] + self.mapVals['durationPulse0'] + self.mapVals['durationPulse2']  # us
        if rectPremphasis != 0 and rectPremphasis != 1:
            print("ERROR: Preemphasis input should be 0 or 1")
            return False

        # Conversion of variables to non-multiplied units
        larmorFreq = larmorFreq * 1e6
        rfExTime = rfExTime * 1e-6
        gapGtoRF = gapGtoRF * 1e-6
        deadTime = deadTime * 1e-6
        gradRiseTime = gradRiseTime * 1e-6  # s
        fov = fov * 1e-2
        dfov = dfov * 1e-3
        acqTime = acqTime * 1e-3  # s
        shimming = shimming * 1e-4
        repetitionTime = repetitionTime * 1e-3  # s

        # Miscellaneous
        larmorFreq = larmorFreq * 1e-6  # MHz
        resolution = fov / nPoints
        self.mapVals['resolution'] = resolution

        # Get cartesian parameters
        dK = 1 / fov
        kMax = nPoints / (2 * fov)  # m-1

        # SetSamplingParameters
        BW = 1 / (acqTime) * 1e-6  # MHz
        samplingPeriod = 1 / BW
        self.mapVals['BW'] = BW
        self.mapVals['kMax'] = kMax
        self.mapVals['dK'] = dK

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

        Teff1 = deadTime + rfExTime / 2 + acqTime / 2
        Gx1 = kx.flatten() / (hw.gammaB * Teff1)
        Gy1 = ky.flatten() / (hw.gammaB * Teff1)
        Gz1 = kz.flatten() / (hw.gammaB * Teff1)
        gSeq = np.vstack((Gx1, Gy1, Gz1)).T
        if nPoints[2] == 1:
            gSeq[2] = 0

        self.mapVals['SequenceGradients'] = gSeq
        self.mapVals['nSPReadouts'] = gSeq.shape[0]

        def createSequence():
            nRep = self.mapVals['nSPReadouts']
            gSeq = self.mapVals['SequenceGradients']
            Grisetime = gradRiseTime * 1e6
            tr = repetitionTime * 1e6
            delayGtoRF = gapGtoRF * 1e6
            RFpulsetime = rfExTime * 1e6
            axesOn = self.mapVals['axesOn']
            TxRxtime1 = deadTime * 1e6
            repeIndex = 0
            ii = 1
            tInit = 20
            # Set shimming
            self.iniSequence(tInit, shimming)

            for ii in range(dummyPulses):
                tdummy = tInit + tr * (ii + 1) + Grisetime + delayGtoRF
                if rectPremphasis == 0:
                    self.rfRecPulse(tdummy, RFpulsetime, rfExAmp, drfPhase * np.pi / 180)
                if rectPremphasis == 1:
                    self.rfRecPulsePreemphasized(tdummy, durationPulse0, RFpulsetime, durationPulse2, factorV0 * rfExAmp, rfExAmp, factorV2 * rfExAmp, 0, channel=txChannel)

            tInit = tInit + tr * dummyPulses

            while repeIndex < nRep:
                # Initialize time
                t0 = tInit + tr * (repeIndex + 1)

                # Set gradients
                if repeIndex == 0:
                    ginit = np.array([0, 0, 0])
                    self.setGradientRamp(t0, Grisetime, nStepsGradRise, ginit[0], gSeq[0, 0] * axesOn[0], axes[0], shimming)
                    self.setGradientRamp(t0, Grisetime, nStepsGradRise, ginit[1], gSeq[0, 1] * axesOn[1], axes[1], shimming)
                    self.setGradientRamp(t0, Grisetime, nStepsGradRise, ginit[2], gSeq[0, 2] * axesOn[2], axes[2], shimming)
                elif repeIndex > 0:
                    if gSeq[repeIndex - 1, 0] != gSeq[repeIndex, 0]:
                        self.setGradientRamp(t0, Grisetime, nStepsGradRise, gSeq[repeIndex - 1, 0] * axesOn[0], gSeq[repeIndex, 0] * axesOn[0], axes[0], shimming)
                    if gSeq[repeIndex - 1, 1] != gSeq[repeIndex, 1]:
                        self.setGradientRamp(t0, Grisetime, nStepsGradRise, gSeq[repeIndex - 1, 1] * axesOn[1], gSeq[repeIndex, 1] * axesOn[1], axes[1], shimming)
                    if gSeq[repeIndex - 1, 2] != gSeq[repeIndex, 2]:
                        self.setGradientRamp(t0, Grisetime, nStepsGradRise, gSeq[repeIndex - 1, 2] * axesOn[2], gSeq[repeIndex, 2] * axesOn[2], axes[2], shimming)

                # Excitation pulse
                trf0 = t0 + Grisetime + delayGtoRF
                if rectPremphasis == 0:
                    self.rfRecPulse(trf0, RFpulsetime, rfExAmp, drfPhase * np.pi / 180)
                if rectPremphasis == 1:
                    self.rfRecPulsePreemphasized(trf0, durationPulse0, RFpulsetime, durationPulse2, factorV0 * rfExAmp, rfExAmp, factorV2 * rfExAmp, 0, channel=txChannel)

                # Rx gate
                t0rx = trf0 + hw.blkTime + RFpulsetime + TxRxtime1
                self.rxGateSync(t0rx, 1 / BWreal)

                if repeIndex == 2 * nRep - 1:
                    self.endSequence(tInit + (nRep + 1) * tr)

                repeIndex = repeIndex + 1
                ii = ii + 1

        # Calibrate frequency
        if freqCal and (not plotSeq):
            drfPhase = self.mapVals['drfPhase']

        # Create full sequence
        # Run the experiment
        overData = []
        if plotSeq == 0 or plotSeq == 1:
            self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
            samplingPeriod = self.expt.getSamplingRate()
            BWreal = 1 / samplingPeriod
            acqTimeSeq = 1 / BWreal  # us
            self.mapVals['BWSeq'] = BWreal * 1e6  # Hz
            self.mapVals['acqTimeSeq'] = acqTimeSeq * 1e-6  # s
            createSequence()
            if self.floDict2Exp():
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("ERROR: sequence waveforms out of hardware bounds")
                return False

            if plotSeq == 0:
                # Warnings before run sequence
                if axes[0] == axes[1] or axes[0] == axes[2] or axes[2] == axes[1]:
                    print("Two different gradient coils has been introduced as the same")
                if gradRiseTime + gapGtoRF + rfExTime + deadTime + acqTimeSeq * 1e-6 >= repetitionTime:
                    print("So short TR")

                # Run all scans
                for ii in range(nScans):
                    rxd, msgs = self.expt.run()
                    rxd['rx0'] = rxd['rx0']  # mV
                    print(f"{ii + 1}/{nScans} SPRITE sequence finished")
                    # Get data
                    overData = np.concatenate((overData, rxd['rx0']), axis=0)

                overData = np.reshape(overData, (nScans, -1))
                radPoints = gSeq.shape[0] * (1 + 2 * hw.addRdPoints) * hw.oversamplingFactor
                overDataRad = np.reshape(overData[:, 0:radPoints], -1)
                fullDataRad = self.decimate(overDataRad, nScans * gSeq.shape[0], option='PETRA')

                # Average results
                RadialSampledPointsRaw = np.average(np.reshape(fullDataRad, (nScans, -1)), axis=0)

                RadialSampledPointsReshaped = np.reshape(RadialSampledPointsRaw, (gSeq.shape[0], 1))
                k_data_a = np.reshape(RadialSampledPointsReshaped, (1 * gSeq.shape[0], 1))

                kSpace1 = np.column_stack((
                    Gx1 * hw.gammaB * Teff1,
                    Gy1 * hw.gammaB * Teff1,
                    Gz1 * hw.gammaB * Teff1,
                    k_data_a.ravel(),
                    k_data_a.real.ravel(),
                    k_data_a.imag.ravel()
                ))

                self.mapVals['kSpaceRaw1'] = kSpace1

            self.expt.__del__()

        return True

    def sequenceAnalysis(self, obj=''):

        def zero_padding(data, order):
            original_shape = data.shape
            if len(original_shape) == 3:
                if original_shape[0] == 1:
                    new_shape = (1, original_shape[1] * order, original_shape[2] * order)
                else:
                    new_shape = tuple(dim * order for dim in original_shape)
            else:
                raise ValueError("Error of matrix shape")

            k_dataZP_a = np.zeros(new_shape, dtype=data.dtype)
            start_indices = tuple((new_dim - old_dim) // 2 for new_dim, old_dim in zip(new_shape, original_shape))
            end_indices = tuple(start + old_dim for start, old_dim in zip(start_indices, original_shape))
            if original_shape[0] == 1:
                k_dataZP_a[0, start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]] = data[0]
            else:
                k_dataZP_a[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1],
                start_indices[2]:end_indices[2]] = data

            return k_dataZP_a

        k_data_aRaw = (np.reshape(self.mapVals['kSpaceRaw1'][:, 4], (self.nPoints[2], self.nPoints[1], self.nPoints[0])))
        self.mapVals['k_data_aRaw'] = k_data_aRaw
        k_data_a = zero_padding(k_data_aRaw, self.mapVals['interpOrder'])
        i_data_a = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(k_data_a)))
        self.mapVals['space_k_a'] = k_data_a
        self.mapVals['space_i_a'] = i_data_a

        # Plots in GUI
        if self.nPoints[2] == 1:
            i_data_a = np.squeeze(i_data_a)
            k_data_a = np.squeeze(k_data_a)

            result1 = {}
            result1['widget'] = 'image'
            result1['data'] = np.abs(i_data_a.reshape(1, self.nPoints[0]*self.mapVals['interpOrder'], self.nPoints[1]*self.mapVals['interpOrder']))
            result1['xLabel'] = "xx"
            result1['yLabel'] = "xx"
            result1['title'] = "Abs image"
            result1['row'] = 0
            result1['col'] = 0

            result2 = {}
            result2['widget'] = 'image'
            result2['data'] = np.abs(k_data_a.reshape(1, self.nPoints[0]*self.mapVals['interpOrder'], self.nPoints[1]*self.mapVals['interpOrder']))
            result2['xLabel'] = "xx"
            result2['yLabel'] = "xx"
            result2['title'] = "Abs k-Space"
            result2['row'] = 0
            result2['col'] = 1

            self.output = [result1, result2]

        if self.nPoints[0] > 1 and self.nPoints[1] > 1 and self.nPoints[2] > 1:
            result1 = {}
            result1['widget'] = 'image'
            result1['data'] = np.abs(i_data_a)
            result1['xLabel'] = "xx"
            result1['yLabel'] = "xx"
            result1['title'] = "Abs image"
            result1['row'] = 0
            result1['col'] = 0

            result2 = {}
            result2['widget'] = 'image'
            result2['data'] = np.abs(k_data_a)
            result2['xLabel'] = "xx"
            result2['yLabel'] = "xx"
            result2['title'] = "Abs k-Space"
            result2['row'] = 0
            result2['col'] = 1

            self.output = [result1, result2]

        # save data once self.output is created
        self.saveRawData()

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        return self.output

# if __name__=='__main__':
#     seq = SPRITE()
#     seq.sequenceRun()