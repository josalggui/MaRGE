"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: rare sequence class
"""


import numpy as np
import controller.experiment_gui as ex
import configs.hw_config as hw  # Import the scanner hardware config
import configs.units as units
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from marga_pulseq.interpreter import PSInterpreter  # Import the marga_pulseq interpreter
import pypulseq as pp  # Import PyPulseq
from skimage.restoration import unwrap_phase as unwrap
from sklearn.preprocessing import PolynomialFeatures
from numpy.linalg import lstsq
from scipy.optimize import curve_fit


# *********************************************************************************
# *********************************************************************************
# *********************************************************************************

class SPDS2(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(SPDS2, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='SPDS2Info', val='SPDS2')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=10.0, field='RF')
        self.addParameter(key='deadTime', string='TxRx dead time (us)', val=[250.0, 450.0], field='RF')
        self.addParameter(key='gapGtoRF', string='Gap G to RF (us)', val=7000.0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=10., field='SEQ')
        self.addParameter(key='fov', string='FOV (cm)', val=[30.0, 30.0, 30.0], field='IM')
        self.addParameter(key='nPoints', string='nPoints (rd, ph, sl)', val=[30, 30, 1], field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=0.02, field='SEQ')
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
        self.addParameter(key='fittingOrder', string='Poly Fitting Order', val=4, field='IM', tip='Polynomics fitting order')
        self.addParameter(key='thresholdMask', string='% Threshold Mask', val=10, field='IM', tip='% Threshold Mask')

    def sequenceInfo(self):
        print("3D SPDS2 sequence")
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
        deadTime1 = self.mapVals['deadTime'][0]  # us
        deadTime2 = self.mapVals['deadTime'][1]  # us
        repetitionTime = self.mapVals['repetitionTime']  # ms
        fov = np.array(self.mapVals['fov'])  # cm
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
        deadTime1 = deadTime1 * 1e-6
        deadTime2 = deadTime2 * 1e-6
        gradRiseTime = gradRiseTime * 1e-6  # s
        fov = fov * 1e-2
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
        kxList = np.linspace(-kMax[0] * (nPoints[0] != 1), kMax[0] * (nPoints[0] != 1), nPoints[0])
        kyList = np.linspace(-kMax[1] * (nPoints[1] != 1), kMax[1] * (nPoints[1] != 1), nPoints[1])
        kzList = np.linspace(-kMax[2] * (nPoints[2] != 1), kMax[2] * (nPoints[2] != 1), nPoints[2])

        kxMesh, kyMesh, kzMesh = np.meshgrid(kxList, kyList, kzList, indexing='ij')
        kxMesh = np.transpose(kxMesh, (2, 0, 1))
        kyMesh = np.transpose(kyMesh, (2, 0, 1))
        kzMesh = np.transpose(kzMesh, (2, 0, 1))

        kCartesianStack = np.column_stack((kxMesh.ravel(), kyMesh.ravel(), kzMesh.ravel()))
        mask = np.sqrt(np.sum(kCartesianStack ** 2, axis=1)) < kMax[0]
        kCartesianMask = kCartesianStack[mask]

        kx = kCartesianMask[:, 0]
        ky = kCartesianMask[:, 1]
        kz = kCartesianMask[:, 2]

        print(f"Repetitions per sequence: {np.sum(mask)} points")

        kCartesianFullGrid = np.zeros(shape=(kxMesh.shape[0] * kxMesh.shape[1] * kxMesh.shape[2], 3))
        kCartesianFullGrid[:, 0] = np.reshape(kxMesh, [kxMesh.shape[0] * kxMesh.shape[1] * kxMesh.shape[2]])
        kCartesianFullGrid[:, 1] = np.reshape(kyMesh, [kyMesh.shape[0] * kyMesh.shape[1] * kyMesh.shape[2]])
        kCartesianFullGrid[:, 2] = np.reshape(kzMesh, [kzMesh.shape[0] * kzMesh.shape[1] * kzMesh.shape[2]])
        self.mapVals['kCartesian'] = kCartesianFullGrid

        Teff1 = deadTime1 + rfExTime / 2 + acqTime / 2
        Gx1 = ky.flatten() / (hw.gammaB * Teff1)
        Gy1 = kx.flatten() / (hw.gammaB * Teff1)
        Gz1 = kz.flatten() / (hw.gammaB * Teff1)
        gSeq1 = np.vstack((Gx1, Gy1, Gz1)).T
        if nPoints[2] == 1:
            gSeq1[2] = 0

        Teff2 = deadTime2 + rfExTime / 2 + acqTime / 2
        Gx2 = ky.flatten() / (hw.gammaB * Teff2)
        Gy2 = kx.flatten() / (hw.gammaB * Teff2)
        Gz2 = kz.flatten() / (hw.gammaB * Teff2)
        gSeq2 = np.vstack((Gx2, Gy2, Gz2)).T
        if nPoints[2] == 1:
            gSeq2[2] = 0

        gSeq = np.concatenate((gSeq1, gSeq2), axis=0)

        self.mapVals['SequenceGradients'] = gSeq
        self.mapVals['nSPReadouts'] = gSeq1.shape[0]

        def createSequence():
            nRep = self.mapVals['nSPReadouts']
            gSeq = self.mapVals['SequenceGradients']
            Grisetime = gradRiseTime * 1e6
            tr = repetitionTime * 1e6
            delayGtoRF = gapGtoRF * 1e6
            axesOn = self.mapVals['axesOn']
            TxRxtime1 = deadTime1 * 1e6
            TxRxtime2 = deadTime2 * 1e6
            durationPulse0 = self.mapVals['durationPulse0']
            durationPulse2 = self.mapVals['durationPulse2']
            rfExTime = self.mapVals['rfExTime']

            repeIndex = 0
            ii = 1
            tInit = 20
            # Set shimming
            self.iniSequence(tInit, shimming)

            for ii in range(dummyPulses):
                tdummy = tInit + tr * (ii + 1) + Grisetime + delayGtoRF
                if rectPremphasis == 0:
                    self.rfRecPulse(tdummy, rfExTime, rfExAmp, drfPhase * np.pi / 180)
                if rectPremphasis == 1:
                    self.rfRecPulsePreemphasized(tdummy, durationPulse0, rfExTime, durationPulse2, factorV0 * rfExAmp,
                                                 rfExAmp, factorV2 * rfExAmp, 0, channel=txChannel)

            tInit = tInit + tr * dummyPulses

            while repeIndex < 2 * nRep:
                # Initialize time
                t0 = tInit + tr * (repeIndex + 1)

                # Set gradients
                if repeIndex == 0:
                    ginit = np.array([0, 0, 0])
                    self.setGradientRamp(t0, Grisetime, nStepsGradRise, ginit[0], gSeq[0, 0] * axesOn[0], axes[0],
                                         shimming)
                    self.setGradientRamp(t0, Grisetime, nStepsGradRise, ginit[1], gSeq[0, 1] * axesOn[1], axes[1],
                                         shimming)
                    self.setGradientRamp(t0, Grisetime, nStepsGradRise, ginit[2], gSeq[0, 2] * axesOn[2], axes[2],
                                         shimming)
                elif repeIndex > 0:
                    if gSeq[repeIndex - 1, 0] != gSeq[repeIndex, 0]:
                        self.setGradientRamp(t0, Grisetime, nStepsGradRise, gSeq[repeIndex - 1, 0] * axesOn[0],
                                             gSeq[repeIndex, 0] * axesOn[0], axes[0], shimming)
                    if gSeq[repeIndex - 1, 1] != gSeq[repeIndex, 1]:
                        self.setGradientRamp(t0, Grisetime, nStepsGradRise, gSeq[repeIndex - 1, 1] * axesOn[1],
                                             gSeq[repeIndex, 1] * axesOn[1], axes[1], shimming)
                    if gSeq[repeIndex - 1, 2] != gSeq[repeIndex, 2]:
                        self.setGradientRamp(t0, Grisetime, nStepsGradRise, gSeq[repeIndex - 1, 2] * axesOn[2],
                                             gSeq[repeIndex, 2] * axesOn[2], axes[2], shimming)

                # Excitation pulse
                trf0 = t0 + Grisetime + delayGtoRF
                if rectPremphasis == 0:
                    self.rfRecPulse(trf0, rfExTime, rfExAmp, drfPhase * np.pi / 180)
                if rectPremphasis == 1:
                    self.rfRecPulsePreemphasized(trf0, durationPulse0, rfExTime, durationPulse2, factorV0 * rfExAmp,
                                                 rfExAmp, factorV2 * rfExAmp, 0, channel=txChannel)

                # Rx gate
                if repeIndex < nRep:
                    if rectPremphasis == 0:
                        t0rx = trf0 + hw.blkTime + rfExTime / 2 + TxRxtime1
                    if rectPremphasis == 1:
                        t0rx = trf0 + hw.blkTime + (rfExTime + durationPulse2 + durationPulse0) / 2 + TxRxtime1

                if repeIndex >= nRep:
                    if rectPremphasis == 0:
                        t0rx = trf0 + hw.blkTime + rfExTime / 2 + TxRxtime2
                    if rectPremphasis == 1:
                        t0rx = trf0 + hw.blkTime + (rfExTime + durationPulse2 + durationPulse0) / 2 + TxRxtime2

                self.rxGateSync(t0rx, 1 / BWreal)

                if repeIndex == 2 * nRep - 1:
                    self.endSequence(tInit + (2 * nRep + 3) * tr)

                repeIndex = repeIndex + 1
                ii = ii + 1

        # Calibrate frequency
        if freqCal and (not plotSeq):
            drfPhase = self.mapVals['drfPhase']

        # Create full sequence
        # Run the experiment
        overData = []
        if plotSeq == 0 or plotSeq == 1:
            self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
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
                if gradRiseTime + gapGtoRF + rfExTime + deadTime1 + acqTimeSeq * 1e-6 >= repetitionTime:
                    print("So short TR")

                # Run all scans
                for ii in range(nScans):
                    rxd, msgs = self.expt.run()
                    rxd['rx0'] = rxd['rx0']  # mV
                    print(f"{ii + 1}/{nScans} SPDS2 sequence finished")
                    # Get data
                    overData = np.concatenate((overData, rxd['rx0']), axis=0)

                overData = np.reshape(overData, (nScans, -1))
                samplesSeq1 = gSeq1.shape[0] * (1 + 2 * hw.addRdPoints) * hw.oversamplingFactor
                samplesSeq2 = gSeq1.shape[0] * (1 + 2 * hw.addRdPoints) * hw.oversamplingFactor
                samplesListSeq1 = np.reshape(overData[:, 0:samplesSeq1], -1)
                samplesListSeq2 = np.reshape(overData[:, samplesSeq1: samplesSeq1 + samplesSeq2], -1)
                samplesDecimatedSeq1 = self.decimate(samplesListSeq1, nScans * gSeq1.shape[0], option='PETRA')
                samplesDecimatedSeq2 = self.decimate(samplesListSeq2, nScans * gSeq1.shape[0], option='PETRA')
                samplesAveragedSeq1 = np.average(np.reshape(samplesDecimatedSeq1, (nScans, -1)), axis=0)
                samplesAveragedSeq2 = np.average(np.reshape(samplesDecimatedSeq2, (nScans, -1)), axis=0)
                samplesReshapededSeq1 = np.reshape(samplesAveragedSeq1, (gSeq1.shape[0], 1))
                data_a = np.reshape(samplesReshapededSeq1, (1 * gSeq1.shape[0], 1))
                samplesReshapededSeq2 = np.reshape(samplesAveragedSeq2, (gSeq1.shape[0], 1))
                data_b = np.reshape(samplesReshapededSeq2, (1 * gSeq1.shape[0], 1))

                # Fill k_space
                k_points = self.mapVals['kCartesian']
                k_data_a = np.zeros(np.size(k_points, 0), dtype=complex)
                k_data_b = np.zeros(np.size(k_points, 0), dtype=complex)
                jj = 0
                for ii in range(np.size(mask)):
                    if mask[ii]:
                        k_data_a[ii] = data_a[jj]
                        k_data_b[ii] = data_b[jj]
                        jj += 1

                #
                # kSpace1 = np.column_stack((
                #     Gx1 * hw.gammaB * Teff1,
                #     Gy1 * hw.gammaB * Teff1,
                #     Gz1 * hw.gammaB * Teff1,
                #     k_data_a.ravel(),
                #     k_data_a.real.ravel(),
                #     k_data_a.imag.ravel()
                # ))
                #
                # kSpace2 = np.column_stack((
                #     Gx2 * hw.gammaB * Teff2,
                #     Gy2 * hw.gammaB * Teff2,
                #     Gz2 * hw.gammaB * Teff2,
                #     k_data_b.ravel(),
                #     k_data_b.real.ravel(),
                #     k_data_b.imag.ravel()
                # ))

                self.mapVals['kSpaceRaw1'] = k_data_a
                self.mapVals['kSpaceRaw2'] = k_data_b

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

        k_data_aRaw = (np.reshape(self.mapVals['kSpaceRaw1'], (self.nPoints[2], self.nPoints[1], self.nPoints[0])))
        k_data_bRaw = (np.reshape(self.mapVals['kSpaceRaw2'], (self.nPoints[2], self.nPoints[1], self.nPoints[0])))
        k_data_a = zero_padding(k_data_aRaw, self.mapVals['interpOrder'])
        k_data_b = zero_padding(k_data_bRaw, self.mapVals['interpOrder'])
        i_data_a = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(k_data_a)))
        i_data_b = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(k_data_b)))
        self.mapVals['space_k_a_Raw'] = k_data_aRaw
        self.mapVals['space_k_b_Raw'] = k_data_bRaw
        self.mapVals['space_k_a'] = k_data_a
        self.mapVals['space_k_b'] = k_data_b
        self.mapVals['space_i_a'] = i_data_a
        self.mapVals['space_i_b'] = i_data_b

        # Plots in GUI
        if self.nPoints[2] == 1:
            i_data_a = np.squeeze(i_data_a)
            i_data_b = np.squeeze(i_data_b)

            # Generate mask
            p_max = np.max(np.abs(i_data_a))
            mask = np.abs(i_data_a) < p_max * self.mapVals['thresholdMask'] / 100

            # Get phase
            RawPhase1 = np.angle(i_data_a)
            RawPhase1[mask] = 0
            RawPhase2 = np.angle(i_data_b)
            RawPhase2[mask] = 0

            i_phase_a = unwrap(RawPhase1)
            i_phase_b = unwrap(RawPhase2)

            # Get magnetic field
            b_field = ((i_phase_b - i_phase_a) / (2 * np.pi * hw.gammaB * (self.deadTime[1] - self.deadTime[0])))
            b_field[mask] = 0
            self.mapVals['b_field'] = b_field

            NX = self.nPoints[0] * self.mapVals['interpOrder']
            NY = self.nPoints[1] * self.mapVals['interpOrder']
            dx = self.fov[0] / NX
            dy = self.fov[1] / NY

            # Here we define the grid of the full FOV and select the indexs where B0 is no null
            ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
            condition = b_field != 0
            ii = ii[condition]
            jj = jj[condition]

            # Here we define the coordinates of the FOV where the B0 is no null
            x = (-(NX - 1) / 2 + ii) * dx
            y = (-(NY - 1) / 2 + jj) * dy

            # Store in values the B0 value in the indexs that accomplishes B0 different of 0
            values = b_field[condition]

            # Save in mapList all the {╥x,y,z,B0} data where B0 is no null
            mapList = np.column_stack((x, y, values))
            self.mapVals['mapList'] = mapList

            # And now we proceed with the fitting
            x_fit = mapList[:, 0]
            y_fit = mapList[:, 1]
            B_fit = mapList[:, 2]
            degree = self.mapVals['fittingOrder']
            poly = PolynomialFeatures(degree)
            coords = np.vstack((x_fit, y_fit)).T
            X_poly = poly.fit_transform(coords)
            coeffs, _, _, _ = lstsq(X_poly, B_fit, rcond=None)
            terms = terms = poly.powers_
            polynomial_expression = ""
            for i, coeff in enumerate(coeffs):
                if coeff != 0:  # Ignore null coefficients
                    powers = terms[i]  # Powers (x**i, y**j)
                    term = f"{coeff}"
                    if any(powers):
                        if powers[0] > 0:
                            term += f"*(x**{powers[0]})"
                        if powers[1] > 0:
                            term += f"*(y**{powers[1]})"
                    polynomial_expression += f" + {term}" if coeff > 0 and i > 0 else f" {term}"

            print("B0 fitting:")
            print(polynomial_expression)

            # Export in txt the model fitted
            output_file = "B0modelledBySPDS.txt"
            with open(output_file, "w") as f:
                f.write(polynomial_expression + "\n")
            print(f"Fitting exported to '{output_file}'")

            result1 = {}
            result1['widget'] = 'image'
            result1['data'] = np.real(b_field.reshape(1, NX, NY))
            result1['xLabel'] = "xx"
            result1['yLabel'] = "xx"
            result1['title'] = "B0 field"
            result1['row'] = 0
            result1['col'] = 3

            result4 = {}
            result4['widget'] = 'image'
            result4['data'] = np.real(RawPhase1.reshape(1, NX, NY))
            result4['xLabel'] = "xx"
            result4['yLabel'] = "xx"
            result4['title'] = "Raw Phase Image Td1"
            result4['row'] = 0
            result4['col'] = 1

            result5 = {}
            result5['widget'] = 'image'
            result5['data'] = np.real(RawPhase2.reshape(1, NX, NY))
            result5['xLabel'] = "xx"
            result5['yLabel'] = "xx"
            result5['title'] = "Raw Phase Image Td2"
            result5['row'] = 1
            result5['col'] = 1

            result2 = {}
            result2['widget'] = 'image'
            result2['data'] = np.real(i_phase_a.reshape(1, NX, NY))
            result2['xLabel'] = "xx"
            result2['yLabel'] = "xx"
            result2['title'] = "Unwrapped Phase Image Td1"
            result2['row'] = 0
            result2['col'] = 2

            result3 = {}
            result3['widget'] = 'image'
            result3['data'] = np.real(i_phase_b.reshape(1, NX, NY))
            result3['xLabel'] = "xx"
            result3['yLabel'] = "xx"
            result3['title'] = "Unwrapped Phase Image Td2"
            result3['row'] = 1
            result3['col'] = 2

            result6 = {}
            result6['widget'] = 'image'
            result6['data'] = np.abs(i_data_a.reshape(1, NX, NY))
            result6['xLabel'] = "xx"
            result6['yLabel'] = "xx"
            result6['title'] = "Raw Abs Image Td1"
            result6['row'] = 0
            result6['col'] = 0

            result7 = {}
            result7['widget'] = 'image'
            result7['data'] = np.abs(i_data_b.reshape(1, NX, NY))
            result7['xLabel'] = "xx"
            result7['yLabel'] = "xx"
            result7['title'] = "Raw Abs Image Td2"
            result7['row'] = 1
            result7['col'] = 0
            #
            # result8 = {}
            # result8['widget'] = 'image'
            # result8['data'] = np.abs(k_data_aRaw.reshape(1, NX, NY))
            # result8['xLabel'] = "xx"
            # result8['yLabel'] = "xx"
            # result8['title'] = "kSpace 1"
            # result8['row'] = 0
            # result8['col'] = 4
            #
            # result9 = {}
            # result9['widget'] = 'image'
            # result9['data'] = np.abs(k_data_bRaw.reshape(1, NX, NY))
            # result9['xLabel'] = "xx"
            # result9['yLabel'] = "xx"
            # result9['title'] = "kSpace 2"
            # result9['row'] = 1
            # result9['col'] = 4

            self.output = [result1, result2, result3, result4, result5, result6, result7]

        if self.nPoints[0] > 1 and self.nPoints[1] > 1 and self.nPoints[2] > 1:
            # Generate mask
            p_max = np.max(np.abs(i_data_a))
            mask = np.abs(i_data_a) < p_max * self.mapVals['thresholdMask'] / 100

            # Get phase
            RawPhase1 = np.angle(i_data_a)
            RawPhase1[mask] = 0
            RawPhase2 = np.angle(i_data_b)
            RawPhase2[mask] = 0

            i_phase_a = unwrap(RawPhase1)
            i_phase_b = unwrap(RawPhase2)

            # Get magnetic field
            b_field = -(i_phase_b - i_phase_a) / (2 * np.pi * hw.gammaB * ((self.deadTime[1] - self.deadTime[0])*1e-6))
            b_field[mask] = 0
            self.mapVals['b_field'] = b_field
            B0mapReorganized = np.flip(np.flip(np.flip(np.transpose(b_field, (2, 1, 0)), axis=0), axis=1), axis=2)
            self.mapVals['B0mapReorganized'] = B0mapReorganized

            NX = self.nPoints[0] * self.mapVals['interpOrder']
            NY = self.nPoints[1] * self.mapVals['interpOrder']
            NZ = self.nPoints[2] * self.mapVals['interpOrder']
            dx = self.fov[0] / NX
            dy = self.fov[1] / NY
            dz = self.fov[2] / NZ

            mapList = []
            cont = 0

            for ii in range(NX):
                for jj in range(NX):
                    for kk in range(NX):
                        if B0mapReorganized[ii, jj, kk] != 0:
                            z_coord = (-(NZ - 1) / 2 + kk) * dz
                            y_coord = (-(NY - 1) / 2 + jj) * dy
                            x_coord = (-(NX - 1) / 2 + ii) * dx
                            value = B0mapReorganized[ii, jj, kk]

                            mapList.append([x_coord, y_coord, z_coord, value])
                            cont += 1

            mapList = np.array(mapList)

            # And now we proceed with the fitting
            x_fit = mapList[:, 0]
            y_fit = mapList[:, 1]
            z_fit = mapList[:, 2]
            B_fit = mapList[:, 3]
            self.mapVals['mapList'] = mapList

            degree = self.mapVals['fittingOrder']
            poly = PolynomialFeatures(degree)
            coords = np.vstack((x_fit, y_fit, z_fit)).T
            X_poly = poly.fit_transform(coords)
            coeffs, _, _, _ = lstsq(X_poly, B_fit, rcond=None)
            terms = poly.powers_
            polynomial_expression = ""
            polynomial_expressionGUI = ""
            for i, coeff in enumerate(coeffs):
                if coeff != 0:  # Ignore null coefficients
                    powers = terms[i]  # Powers (x**i, y**j, z**k)
                    term = f"{coeff}"
                    termGUI = f"{coeff}"
                    if any(powers):
                        if powers[2] > 0:
                            term += f"*(z**{powers[2]})"
                            termGUI += f"*(z^{powers[2]})"
                        if powers[1] > 0:
                            term += f"*(y**{powers[1]})"
                            termGUI += f"*(y^{powers[1]})"
                        if powers[0] > 0:
                            term += f"*(x**{powers[0]})"
                            termGUI += f"*(x^{powers[0]})"
                    polynomial_expression += f" + {term}" if coeff > 0 and i > 0 else f" {term}"
                    polynomial_expressionGUI += f" + {termGUI}" if coeff > 0 and i > 0 else f" {termGUI}"

            print("B0 fitting:")
            print(polynomial_expressionGUI)

            # Export in txt the model fitted
            output_file = "B0modelledBySPDS.txt"
            with open(output_file, "w") as f:
                f.write(polynomial_expression + "\n")
            print(f"Fitting exported to '{output_file}'")

            result1 = {}
            result1['widget'] = 'image'
            result1['data'] = np.real(b_field)
            result1['xLabel'] = "xx"
            result1['yLabel'] = "xx"
            result1['title'] = "B0 field"
            result1['row'] = 0
            result1['col'] = 3

            result4 = {}
            result4['widget'] = 'image'
            result4['data'] = np.real(RawPhase1)
            result4['xLabel'] = "xx"
            result4['yLabel'] = "xx"
            result4['title'] = "Raw Phase Image Td1"
            result4['row'] = 0
            result4['col'] = 1

            result5 = {}
            result5['widget'] = 'image'
            result5['data'] = np.real(RawPhase2)
            result5['xLabel'] = "xx"
            result5['yLabel'] = "xx"
            result5['title'] = "Raw Phase Image Td2"
            result5['row'] = 1
            result5['col'] = 1

            result2 = {}
            result2['widget'] = 'image'
            result2['data'] = np.real(i_phase_a)
            result2['xLabel'] = "xx"
            result2['yLabel'] = "xx"
            result2['title'] = "Unwrapped Phase Image Td1"
            result2['row'] = 0
            result2['col'] = 2

            result3 = {}
            result3['widget'] = 'image'
            result3['data'] = np.real(i_phase_b)
            result3['xLabel'] = "xx"
            result3['yLabel'] = "xx"
            result3['title'] = "Unwrapped Phase Image Td2"
            result3['row'] = 1
            result3['col'] = 2

            result6 = {}
            result6['widget'] = 'image'
            result6['data'] = np.abs(i_data_a)
            result6['xLabel'] = "xx"
            result6['yLabel'] = "xx"
            result6['title'] = "Raw Abs Image Td1"
            result6['row'] = 0
            result6['col'] = 0

            result7 = {}
            result7['widget'] = 'image'
            result7['data'] = np.abs(i_data_b)
            result7['xLabel'] = "xx"
            result7['yLabel'] = "xx"
            result7['title'] = "Raw Abs Image Td2"
            result7['row'] = 1
            result7['col'] = 0

            # result8 = {}
            # result8['widget'] = 'image'
            # result8['data'] = np.abs(k_data_aRaw)
            # result8['xLabel'] = "xx"
            # result8['yLabel'] = "xx"
            # result8['title'] = "kSpace 1"
            # result8['row'] = 0
            # result8['col'] = 4
            #
            # result9 = {}
            # result9['widget'] = 'image'
            # result9['data'] = np.abs(k_data_bRaw)
            # result9['xLabel'] = "xx"
            # result9['yLabel'] = "xx"
            # result9['title'] = "kSpace 2"
            # result9['row'] = 1
            # result9['col'] = 4

            self.output = [result1, result2, result3, result4, result5, result6, result7]

        # save data once self.output is created
        self.saveRawData()

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        return self.output

# if __name__=='__main__':
#     seq = SPDS2()
#     seq.sequenceRun()