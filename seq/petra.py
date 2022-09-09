"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: rare sequence class
"""

import os
import sys
import time
import numpy as np
import experiment as ex
import matplotlib.pyplot as plt
import scipy
import scipy.signal as sig
import pdb
import torch
import configs.hw_config as hw # Import the scanner hardware config
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import pyqtgraph as pg              # To plot nice 3d images
import copy
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter import Menu
from sys import exit
from scipy.interpolate import griddata
from plotview.spectrumplot import Spectrum3DPlot # To show nice 2d or 3d images


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class PETRA(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(PETRA, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='PETRAInfo', val='PETRA')
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
        self.addParameter(key='axes', string='Axes', val=[0, 2, 1], field='IM')
        self.addParameter(key='axesEnable', string='Axes enable', val=[1, 1, 0], field='IM')
        self.addParameter(key='drfPhase', string='Phase of excitation pulse (º)', val=0.0, field='RF')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=0, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='gradRiseTime', string='Grad Rise Time (us)', val=1000, field='OTH')
        self.addParameter(key='nStepsGradRise', string='Grad steps', val=5, field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='NyquistOS', string='Radial oversampling', val=1, field='SEQ')



    def sequenceInfo(self):
        print(" ")
        print("3D PETRA sequence")
        print("Under development")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")


    def sequenceTime(self):
        self.sequenceRun(2)
        return self.mapVals['nScans'] * self.mapVals['repetitionTime'] * 1e-3 * self.mapVals['SequenceGradients'].shape[0] / 60

    def sequenceRun(self, plotSeq=0):
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
        axes = self.mapVals['axes']
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
        addRdPoints = 3             # Initial rd points to avoid artifact at the begining of rd
        resolution = fov/nPoints
        self.mapVals['resolution'] = resolution
        self.mapVals['addRdPoints'] = addRdPoints

        # Get cartesian parameters
        dK = 1 / fov
        kMax = nPoints / (2 * fov)  # m-1

        # SetSamplingParameters
        BWoriginal = (np.max(nPoints))*NyquistOS / (2 * acqTime) * 1e-6 # MHz
        samplingPeriodOriginal = 1/BWoriginal
        BWov = BWoriginal * hw.oversamplingFactor  # MHz
        samplingPeriod = 1 / BWov  # us
        self.mapVals['BWinitial'] = BWoriginal
        self.mapVals['BWov'] = BWov
        self.mapVals['kMax'] = kMax
        self.mapVals['dK'] = dK

        gradientAmplitudes = kMax / (hw.gammaB * acqTime)
        if axesEnable[0] == 0:
            gradientAmplitudes[0] = 0
        if axesEnable[1] == 0:
            gradientAmplitudes[1] = 0
        if axesEnable[2] == 0:
            gradientAmplitudes[2] = 0
        print("Gradient strengths are  ", gradientAmplitudes * 1e3, " mT/m")

        nPPL = np.int(np.ceil((1.73205 * acqTime - deadTime - 0.5 * rfExTime) * BWoriginal * 1e6 + 1))
        nLPC = np.int(np.ceil(max(nPoints[0], nPoints[1]) * np.pi / undersampling))
        nLPC = max(nLPC - (nLPC % 2), 1)
        nCir = max(np.int(np.ceil(nPoints[2] * np.pi / 2 / undersampling) + 1), 1)

        if axesEnable[0] == 0 or axesEnable[1] == 0 or axesEnable[2] == 0:
            nCir = 1
        if axesEnable[0] == 0 and axesEnable[1] == 0:
            nLPC = 2
        if axesEnable[0] == 0 and axesEnable[2] == 0:
            nLPC = 2
        if axesEnable[2] == 0 and axesEnable[1] == 0:
            nLPC = 2

        acqTime = nPPL / BWoriginal # us
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
            nRepetitions = nRepetitions + max(np.int(np.ceil(nLPC * np.sin(theta[jj]))), 1)
        self.mapVals['nRepetitions'] = nRepetitions
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
            nLPCjj = max(np.int(np.ceil(nLPC * np.sin(theta[jj]))), 1)
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
        normalizedKRadial[:, :, 0] = (0.5 * rfExTime + deadTime + (0.5 / (BWoriginal*1e6))) * normalizedGradientsRadial
        # Calculate all k-points
        for jj in range(1, nPPL):
            normalizedKRadial[:, :, jj] = normalizedKRadial[:, :, 0] + jj* normalizedGradientsRadial / (BWoriginal*1e6)

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
        tMin = 0.5 * rfExTime + deadTime + 0.5 / (BWoriginal * 1e6)

        # Get the full cartesian points
        kx = np.linspace(-kMax[0] * (nPoints[0] != 1), kMax[0] * (nPoints[0] != 1), nPoints[0])
        ky = np.linspace(-kMax[1] * (nPoints[1] != 1), kMax[1] * (nPoints[1] != 1), nPoints[1])
        kz = np.linspace(-kMax[2] * (nPoints[2] != 1), kMax[2] * (nPoints[2] != 1), nPoints[2])

        kx, ky, kz = np.meshgrid(kx, ky, kz)
        kx = torch.from_numpy(kx)
        kx = kx.permute(2, 0, 1)
        ky = torch.from_numpy(ky)
        ky = ky.permute(2, 0, 1)
        kz = torch.from_numpy(kz)
        kz = kz.permute(2, 0, 1)

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

        gSeq = - np.concatenate((gradientVectors1, gradientVectors2), axis=0)
        self.mapVals['SequenceGradients'] = gSeq

        def createSequence():
            nRep = gSeq.shape[0]
            Grisetime = gradRiseTime * 1e6
            tr = repetitionTime * 1e6
            delayGtoRF = gapGtoRF * 1e6
            RFpulsetime = rfExTime * 1e6
            TxRxtime = deadTime * 1e6
            repeIndex = 0
            ii = 1
            tInit = 20
            print(nRep)
            # Set shimming
            self.iniSequence(tInit, shimming)

            while repeIndex < nRep:
                # Initialize time
                t0 = tInit + tr * (repeIndex + 1)

                # Set gradients
                if repeIndex == 0:
                    ginit = np.array([0, 0, 0])
                    self.setGradientRamp(t0, Grisetime, nStepsGradRise, ginit[0], gSeq[0, 0], axes[0], shimming)
                    self.setGradientRamp(t0, Grisetime, nStepsGradRise, ginit[1], gSeq[0, 1], axes[1], shimming)
                    self.setGradientRamp(t0, Grisetime, nStepsGradRise, ginit[2], gSeq[0, 2], axes[2], shimming)
                elif repeIndex > 0:
                    if gSeq[repeIndex-1, 0] != gSeq[repeIndex, 0]:
                        self.setGradientRamp(t0, Grisetime, nStepsGradRise, gSeq[repeIndex-1, 0], gSeq[repeIndex, 0], axes[0], shimming)
                    if gSeq[repeIndex-1, 1] != gSeq[repeIndex, 1]:
                        self.setGradientRamp(t0, Grisetime, nStepsGradRise, gSeq[repeIndex-1, 1], gSeq[repeIndex, 1], axes[1], shimming)
                    if gSeq[repeIndex-1, 2] != gSeq[repeIndex, 2]:
                        self.setGradientRamp(t0, Grisetime, nStepsGradRise, gSeq[repeIndex-1, 2], gSeq[repeIndex, 2], axes[2], shimming)

                # Excitation pulse
                trf0 = t0 + Grisetime + delayGtoRF
                self.rfRecPulse(trf0, RFpulsetime, rfExAmp, drfPhase * np.pi / 180, txChannel=txChannel)

                if repeIndex < gradientVectors1.shape[0]:
                    tACQ = acqTimeSeq + addRdPoints / BWreal
                if repeIndex >= gradientVectors1.shape[0]:
                    tACQ = addRdPoints / BWreal + 1 / BWreal

                # Rx gate
                t0rx = trf0 + hw.blkTime + RFpulsetime + TxRxtime - addRdPoints / BWreal
                self.rxGate(t0rx, tACQ, rxChannel=rxChannel)

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
            samplingPeriod = self.expt.get_rx_ts()[0]
            BWreal = 1 / samplingPeriod / hw.oversamplingFactor
            acqTimeSeq = nPPL / BWreal  # us
            self.mapVals['BW-real'] = BWreal
            self.mapVals['acqTimeSeq'] = acqTimeSeq
            createSequence()

            if plotSeq == 0:
                # Warnings before run sequence
                if axes[0] == axes[1] or axes[0] == axes[2] or axes[2] == axes[1]:
                    print("Two different gradient coils has been introduced as the same")
                if gradientAmplitudes[0] * 1e3 > 30 or gradientAmplitudes[1] * 1e3 > 30 or gradientAmplitudes[2] * 1e3 > 30:
                    print("So demanding current for gradient coils")
                    messagebox.showinfo(message="So demanding current for gradient coils", title="Warning high currents")
                if gradRiseTime + gapGtoRF + rfExTime + deadTime + acqTimeSeq*1e-6 >= repetitionTime:
                    print("So short TR")
                    messagebox.showinfo(message="So short TR. Enlarge it!", title="Warning TR short")


                for ii in range(nScans):
                    print('Running...')
                    rxd, msgs = self.expt.run()
                    rxd['rx0'] = rxd['rx0'] * 13.788  # Here I normalize to get the result in mV
                    print('PETRA sequence finished!')
                    # Get data
                    overData = np.concatenate((overData, rxd['rx0']), axis=0)

                overData = np.reshape(overData, (rxd['rx0'].shape[0], nScans))
                overData = np.average(overData, axis=1)
                dataFull = sig.decimate(overData, hw.oversamplingFactor, ftype='fir', zero_phase=True)
                RadialSampledPointsRaw = dataFull[0:(nPPL + addRdPoints) * gradientVectors1.shape[0]]
                RadialSampledPointsReshaped = np.reshape(RadialSampledPointsRaw, (gradientVectors1.shape[0], nPPL+addRdPoints))
                RadialSampledPointsFilt = np.delete(RadialSampledPointsReshaped, np.s_[0:addRdPoints], axis=1)
                RadialSampledList = np.reshape(RadialSampledPointsFilt, (nPPL*gradientVectors1.shape[0], 1))

                CartesianSampledPointsRaw = dataFull[(nPPL + addRdPoints) * gradientVectors1.shape[0]:dataFull.shape[0]]
                CartesianSampledPointsReshaped = np.reshape(CartesianSampledPointsRaw, (gradientVectors2.shape[0], 1 + addRdPoints))
                CartesianSampledPointsFilt = np.delete(CartesianSampledPointsReshaped, np.s_[0:addRdPoints], axis=1)
                CartesianSampledList = np.reshape(CartesianSampledPointsFilt, (1*gradientVectors2.shape[0], 1))

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
                    print('3D regridding')
                    valCartesian = griddata((kxOriginal, kyOriginal, kzOriginal), np.reshape(kSpace[:, 3], -1), (kxTarget, kyTarget, kzTarget), method='linear', fill_value=0, rescale=False)

                    DELX = dfov[0]
                    DELY = dfov[1]
                    DELZ = dfov[2]
                    phase = np.exp(-2 * np.pi * 1j * (DELX * kCartesian[:, 0] + DELY * kCartesian[:, 1]+DELZ * kCartesian[:, 2]))
                    valCartesian = valCartesian * phase

                if (nCir == 1) and (nLPC > 2):
                    kxOriginal = np.reshape(np.real(kSpace[:, 0]), -1)
                    kyOriginal = np.reshape(np.real(kSpace[:, 1]), -1)
                    kxTarget = np.reshape(kCartesian[:, 0], -1)
                    kyTarget = np.reshape(kCartesian[:, 1], -1)
                    print('2D regridding')
                    valCartesian = griddata((kxOriginal, kyOriginal), np.reshape(kSpace[:, 3], -1), (kxTarget, kyTarget), method='linear', fill_value=0, rescale=False)

                DELX = dfov[0]
                DELY = dfov[1]
                DELZ = dfov[2]
                phase = np.exp(-2 * np.pi * 1j * (DELX * kCartesian[:, 0] + DELY * kCartesian[:, 1]+DELZ * kCartesian[:, 2]))
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


    def sequenceAnalysis(self, obj=''):
        self.saveRawData()
        axesEnable = self.mapVals['axesEnable']
        kSpace = self.mapVals['kSpaceArray']
        imagenFFT = self.mapVals['ImageFFT']

        image = Spectrum3DPlot(np.abs(imagenFFT),
                               title='Image magnitude',
                               xLabel= " Axis",
                               yLabel= " Axis")
        imageWidget = image.getImageWidget()

        kSpace = Spectrum3DPlot(np.log10(np.abs(kSpace)),
                                title='k-Space',
                                xLabel="k",
                                yLabel="k")
        kSpaceWidget = kSpace.getImageWidget()

        return ([imageWidget, kSpaceWidget])


# if __name__=='__main__':
#     seq = PETRA()
#     seq.sequenceRun()