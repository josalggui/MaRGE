"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: mri blank sequence with common methods that will be inherited by any sequence
"""

import os
import numpy as np
import configs.hw_config as hw
from datetime import date,  datetime
from scipy.io import savemat
import experiment as ex
import scipy.signal as sig
import csv

class MRIBLANKSEQ:
    # Properties
    mapKeys = []         # keys for the maps
    mapNmspc = {}        # name to show in the gui
    mapVals = {}         # values to show in the gui
    mapFields = {}       # fields to classify the input parameter
    mapLen = {}
    plotSeq = 1          # it plots the sequence

    def __init__(self):
        self.mapKeys = []
        self.mapNmspc = {}
        self.mapVals = {}
        self.mapFields = {}
        self.mapLen = {}

    # *********************************************************************************
    # *********************************************************************************
    # *********************************************************************************

    # Create dictionaries of inputs classified by field (RF, SEQ, IM or OTH)

    @property
    def RFproperties(self):
        # Automatically select the inputs related to RF fields
        out = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'RF':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
        return out

    @property
    def IMproperties(self) -> dict:
        # Automatically select the inputs related to IM fields
        out = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'IM':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
        return out

    @property
    def SEQproperties(self) -> dict:
        # Automatically select the inputs related to SEQ fields
        out = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'SEQ':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
        return out

    @property
    def OTHproperties(self) -> dict:
        # Automatically select the inputs related to OTH fields
        out = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'OTH':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
        return out

    def saveParams(self):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Save sequence input parameters into csv
        """
        self.resetMapVals()
        if not os.path.exists('../experiments/parameterisations'):
            os.makedirs('../experiments/parameterisations')
        with open('../experiments/parameterisations/%s_last_parameters.csv' % self.mapVals['seqName'], 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.mapKeys)
            writer.writeheader()
            writer.writerows([self.mapVals])

    def loadParams(self):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Load sequence parameters from csv
        """
        mapValsOld = self.mapVals
        with open('../experiments/parameterisations/%s_last_parameters.csv' % self.mapVals['seqName'],'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for l in reader:
                mapValsNew = l
        self.mapVals = {}

        # Get key for corresponding modified parameter
        for key in self.mapKeys:
            dataLen = self.mapLen[key]
            valOld = mapValsOld[key]
            valNew = mapValsNew[key]
            valNew = valNew.replace('[', '')
            valNew = valNew.replace(']', '')
            valNew = valNew.split(',')
            if type(valOld) == str:
                valOld = [valOld]
            elif dataLen == 1:
                valOld = [valOld]
            dataType = type(valOld[0])

            inputNum = []
            for ii in range(dataLen):
                if dataType == float or dataType == np.float64:
                    try:
                        inputNum.append(float(valNew[ii]))
                    except:
                        inputNum.append(float(valOld[ii]))
                elif dataType == int:
                    try:
                        inputNum.append(int(valNew[ii]))
                    except:
                        inputNum.append(int(valOld[ii]))
                else:
                    try:
                        inputNum.append(str(valNew[0]))
                        break
                    except:
                        inputNum.append(str(valOld[0]))
                        break
            if dataType == str:
                self.mapVals[key] = inputNum[0]
            else:
                if dataLen == 1:  # Save value into mapVals
                    self.mapVals[key] = inputNum[0]
                else:
                    self.mapVals[key] = inputNum

    def resetMapVals(self):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Delete all new parameters in mapVals
        """
        mapVals2 = {}
        for key in self.mapKeys:
            mapVals2[key] = self.mapVals[key]
        self.mapVals = mapVals2

    def sequencePlot(self):
        """ axes: 4-element tuple of axes upon which the TX, gradients, RX and digital I/O plots will be drawn.
        If not provided, plot_sequence() will create its own. """
        # if axes is None:
        #     _, axes = plt.subplots(4, 1, figsize=(12,8), sharex='col')

        # (txs, grads, rxs, ios) = axes

        fd = self.expt.get_flodict()

        def getStepData(data):
            t = data[0]
            s = data[1]
            n = np.size(t)
            tStep = np.zeros(2*n-1)
            sStep = np.zeros(2*n-1)
            tStep[0::2] = t
            tStep[1::2] = t[1::]
            sStep[0::2] = s
            sStep[1::2] = s[0:-1]
            return [tStep, sStep]

        # Plot TX channels
        xData = []
        yData = []
        legend = []
        for txl in ['tx0_i', 'tx0_q', 'tx1_i', 'tx1_q']:
            try:
                dataStep = getStepData(fd[txl])
                xData.append(dataStep[0]*1e-3)
                yData.append(dataStep[1])
                legend.append(txl)
            except KeyError:
                continue
        plotTx = [xData, yData, legend, 'Rx gate']

        # Plot gradient channels
        xData = []
        yData = []
        legend = []
        for gradl in self.expt.gradb.keys():
            try:
                dataStep = getStepData(fd[gradl])
                xData.append(dataStep[0]*1e-3)
                yData.append(dataStep[1])
                legend.append(gradl)
            except KeyError:
                continue
        plotGrad = [xData, yData, legend, 'Gradients']

        # Plot RX enable channels
        xData = []
        yData = []
        legend = []
        for rxl in ['rx0_en', 'rx1_en']:
            try:
                dataStep = getStepData(fd[rxl])
                xData.append(dataStep[0]*1e-3)
                yData.append(dataStep[1])
                legend.append(rxl)
            except KeyError:
                continue
        plotRx = [xData, yData, legend, 'Rx gate']

        # Plot digital outputs
        xData = []
        yData = []
        legend = []
        for iol in ['tx_gate', 'rx_gate', 'trig_out', 'leds']:
            try:
                dataStep = getStepData(fd[iol])
                xData.append(dataStep[0]*1e-3)
                yData.append(dataStep[1])
                legend.append(iol)
            except KeyError:
                continue
        plotDigital = [xData, yData, legend, 'Digital']

        return([plotTx, plotGrad, plotRx, plotDigital])

    def getIndex(self, etl=1, nPH=1, sweepMode=1):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Create 'ind' array that give you the order to sweep the k-space phase lines along an echo train length.
        sweepMode = 0: -kMax to kMax
        sweepMode = 1: 0 to kMax
        sweepMode = 2: kMax to 0
        sweepMode = 3: Niquist modulated method
        """
        n2ETL = int(nPH / 2 / etl)
        ind = []
        if nPH == 1:
            ind = np.array([0])
        else:
            if sweepMode == 0:  # Sequential for T2 contrast
                for ii in range(int(nPH / etl)):
                    ind = np.concatenate((ind, np.linspace(ii, nPH + ii, num=etl, endpoint=False)), axis=0)
                ind = ind[::-1]
            elif sweepMode == 1:  # Center-out for T1 contrast
                if etl == nPH:
                    ind = np.zeros(nPH)
                    ind[0::2] = np.linspace(int(nPH / 2), nPH, num=int(nPH / 2), endpoint=False)
                    ind[1::2] = np.linspace(int(nPH / 2) - 1, -1, num=int(nPH / 2), endpoint=False)
                else:
                    for ii in range(n2ETL):
                        ind = np.concatenate((ind, np.linspace(int(nPH / 2) + ii, nPH + ii, num=etl, endpoint=False)),
                                             axis=0)
                        ind = np.concatenate(
                            (ind, np.linspace(int(nPH / 2) - ii - 1, -ii - 1, num=etl, endpoint=False)), axis=0)
            elif sweepMode == 2:  # Out-to-center for T2 contrast
                if etl == nPH:
                    ind = np.zeros(nPH)
                    ind[0::2] = np.linspace(int(nPH / 2), nPH, num=int(nPH / 2), endpoint=False)
                    ind[1::2] = np.linspace(int(nPH / 2) - 1, -1, num=int(nPH / 2), endpoint=False)
                else:
                    for ii in range(n2ETL):
                        ind = np.concatenate((ind, np.linspace(int(nPH / 2) + ii, nPH + ii, num=etl, endpoint=False)),
                                             axis=0)
                        ind = np.concatenate(
                            (ind, np.linspace(int(nPH / 2) - ii - 1, -ii - 1, num=etl, endpoint=False)), axis=0)
                ind = ind[::-1]
            elif sweepMode == 3:  # Niquist modulated to reduce ghosting artifact
                if etl == nPH:
                    ind = np.arange(0, nPH, 1)
                else:
                    for ii in range(int(n2ETL)):
                        ind = np.concatenate((ind, np.arange(0, nPH, 2 * n2ETL) + 2 * ii), axis=0)
                        ind = np.concatenate((ind, np.arange(nPH - 1, 0, -2 * n2ETL) - 2 * ii), axis=0)

        return np.int32(ind)

    def fixEchoPosition(self, echoes, data0):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Oversampled data obtained with a given echo train length and readout gradient only is used here to determine the true position of k=0.
        After getting the position of k = 0 for each gradient-spin-echo, it shift the sampled data to place k = 0 at the center of each acquisition window.
        """
        etl = np.size(echoes, axis=0)
        n = np.size(echoes, axis=1)
        idx = np.argmax(np.abs(echoes), axis=1)
        idx = idx - int(n / 2)
        data1 = data0 * 0
        for ii in range(etl):
            if idx[ii] > 0:
                idx[ii] = 0
            data1[:, ii, -idx[ii]::] = data0[:, ii, 0:n + idx[ii]]
        return (data1)

    def rfSincPulse(self, tStart, rfTime, rfAmplitude, rfPhase=0, nLobes=7, rewrite=True):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Rf pulse with sinc pulse shape. I use a Hanning window to reduce the banding of the frequency profile.
        """
        txTime = np.linspace(tStart, tStart + rfTime, num=100, endpoint=True) + hw.blkTime
        nZeros = (nLobes + 1)
        tx = np.linspace(-nZeros / 2, nZeros / 2, num=100, endpoint=True)
        hanning = 0.5 * (1 + np.cos(2 * np.pi * tx / nZeros))
        txAmp = rfAmplitude * np.exp(1j * rfPhase) * hanning * np.abs(np.sinc(tx))
        txGateTime = np.array([tStart, tStart + hw.blkTime + rfTime])
        txGateAmp = np.array([1, 0])
        self.expt.add_flodict({
            'tx0': (txTime, txAmp),
            'tx_gate': (txGateTime, txGateAmp)
        }, rewrite)

    def rfRecPulse(self, tStart, rfTime, rfAmplitude, rfPhase=0, rewrite=True):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Rf pulse with square pulse shape
        """
        txTime = np.array([tStart + hw.blkTime, tStart + hw.blkTime + rfTime])
        txAmp = np.array([rfAmplitude * np.exp(1j * rfPhase), 0.])
        txGateTime = np.array([tStart, tStart + hw.blkTime + rfTime])
        txGateAmp = np.array([1, 0])
        self.expt.add_flodict({
            'tx0': (txTime, txAmp),
            'tx_gate': (txGateTime, txGateAmp)
        }, rewrite)

    def rxGate(self, tStart, gateTime, rewrite=True):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        """
        rxGateTime = np.array([tStart, tStart + gateTime])
        rxGateAmp = np.array([1, 0])
        self.expt.add_flodict({
            'rx0_en': (rxGateTime, rxGateAmp),
            'rx_gate': (rxGateTime, rxGateAmp),
        })

    def gradTrap(self, tStart, gRiseTime, gFlattopTime, gAmp, gSteps, gAxis, shimming, rewrite=True):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        gradient pulse with trapezoidal shape. Use 1 step to generate a square pulse.
        Time inputs in us
        Amplitude inputs in T/m
        """
        tUp = np.linspace(tStart, tStart + gRiseTime, num=gSteps, endpoint=False)
        tDown = tUp + gRiseTime + gFlattopTime
        t = np.concatenate((tUp, tDown), axis=0)
        dAmp = gAmp / gSteps
        aUp = np.linspace(dAmp, gAmp, num=gSteps)
        aDown = np.linspace(gAmp - dAmp, 0, num=gSteps)
        a = np.concatenate((aUp, aDown), axis=0) / hw.gFactor[gAxis]
        if gAxis == 0:
            self.expt.add_flodict({'grad_vx': (t, a + shimming[0])}, rewrite)
        elif gAxis == 1:
            self.expt.add_flodict({'grad_vy': (t, a + shimming[1])}, rewrite)
        elif gAxis == 2:
            self.expt.add_flodict({'grad_vz': (t, a + shimming[2])}, rewrite)

    def gradTrapMomentum(self, tStart, kMax, gTotalTime, gAxis, shimming, rewrite=True):
        """"
        @author: T. Guallart-Naval, MRILab, Tesoro Imaging S.L., Valencia, Spain
        @email: teresa.guallart@tesoroimaging.com
        Gradient pulse with trapezoidal shape according to slewrate.
        Time inputs in us
        kMax inputs in 1/m

        """
        kMax = kMax / hw.gammaB * 1e6

        # Changing from Ocra1 units
        slewRate = hw.slewRate / hw.gFactor[gAxis]  # Convert to units [s*m/T]
        stepsRate = hw.stepsRate / hw.gFactor[gAxis]  # Convert to units [steps*m/T]

        # Calculating amplitude
        gAmplitude = (gTotalTime - np.sqrt(gTotalTime ** 2 - 4 * slewRate * kMax)) / (2 * slewRate)

        # Trapezoid characteristics
        gRiseTime = gAmplitude * slewRate
        nSteps = int(np.ceil(gAmplitude * stepsRate))

        # # Creating trapezoid
        tRise = np.linspace(tStart, tStart + gRiseTime, nSteps, endpoint=True)
        aRise = np.linspace(0, gAmplitude, nSteps, endpoint=True)
        tDown = np.linspace(tStart + gTotalTime - gRiseTime, tStart + gTotalTime, nSteps, endpoint=True)
        aDown = np.linspace(gAmplitude, 0, nSteps, endpoint=True)
        gTime = np.concatenate((tRise, tDown), axis=0)
        gAmp = np.concatenate((aRise, aDown), axis=0) / hw.gFactor[gAxis]
        if gAxis == 0:
            self.expt.add_flodict({'grad_vx': (gTime, gAmp + shimming[0])}, rewrite)
        elif gAxis == 1:
            self.expt.add_flodict({'grad_vy': (gTime, gAmp + shimming[1])}, rewrite)
        elif gAxis == 2:
            self.expt.add_flodict({'grad_vz': (gTime, gAmp + shimming[2])}, rewrite)

    def setGradientRamp(self, tStart, gradRiseTime, nStepsGradRise, g0, gf, gAxes, shimming, rewrite=True):
        tRamp = np.zeros(nStepsGradRise)
        gAmp = np.zeros(nStepsGradRise)
        for kk in range(nStepsGradRise):
            tRamp[kk] = tStart + gradRiseTime * kk / nStepsGradRise
            gAmp[kk] = (g0 + ((gf - g0) * (kk + 1) / nStepsGradRise)) / hw.gFactor[gAxes]

            if gAxes == 0:
                self.expt.add_flodict({'grad_vx': (tRamp[kk], gAmp[kk] + shimming[0])}, rewrite)
            elif gAxes == 1:
                self.expt.add_flodict({'grad_vy': (tRamp[kk], gAmp[kk] + shimming[1])}, rewrite)
            elif gAxes == 2:
                self.expt.add_flodict({'grad_vz': (tRamp[kk], gAmp[kk] + shimming[2])}, rewrite)


    def gradTrapAmplitude(self, tStart, gAmplitude, gTotalTime, gAxis, shimming, orders, rewrite=True):
        """"
        @author: T. Guallart-Naval, MRILab, Tesoro Imaging S.L., Valencia, Spain
        @email: teresa.guallart@tesoroimaging.com
        Gradient pulse with trapezoidal shape according to slewrate.
        Time inputs in us
        gAmplitude inputs in T/m

        """
        # Changing from Ocra1 units
        slewRate = hw.slewRate / hw.gFactor[gAxis]  # Convert to units [s*m/T]
        stepsRate = hw.stepsRate / hw.gFactor[gAxis]  # Convert to units [steps*m/T]

        # Trapezoid characteristics
        gRiseTime = np.abs(gAmplitude * slewRate)
        nSteps = int(np.ceil(np.abs(gAmplitude * stepsRate)))
        orders = orders + 2 * nSteps

        # # Creating trapezoid
        tRise = np.linspace(tStart, tStart + gRiseTime, nSteps, endpoint=True)
        aRise = np.linspace(0, gAmplitude, nSteps, endpoint=True)
        tDown = np.linspace(tStart + gTotalTime - gRiseTime, tStart + gTotalTime, nSteps, endpoint=True)
        aDown = np.linspace(gAmplitude, 0, nSteps, endpoint=True)
        gTime = np.concatenate((tRise, tDown), axis=0)
        gAmp = np.concatenate((aRise, aDown), axis=0) / hw.gFactor[gAxis]
        if gAxis == 0:
            self.expt.add_flodict({'grad_vx': (gTime, gAmp + shimming[0])}, rewrite)
        elif gAxis == 1:
            self.expt.add_flodict({'grad_vy': (gTime, gAmp + shimming[1])}, rewrite)
        elif gAxis == 2:
            self.expt.add_flodict({'grad_vz': (gTime, gAmp + shimming[2])}, rewrite)

    def endSequence(self, tEnd):
        self.expt.add_flodict({
            'grad_vx': (np.array([tEnd]), np.array([0])),
            'grad_vy': (np.array([tEnd]), np.array([0])),
            'grad_vz': (np.array([tEnd]), np.array([0])),
            'rx0_en': (np.array([tEnd]), np.array([0])),
            'rx_gate': (np.array([tEnd]), np.array([0])),
            'tx0': (np.array([tEnd]), np.array([0*np.exp(0)])),
            'tx_gate': (np.array([tEnd]), np.array([0]))
        })

    def iniSequence(self, t0, shimming, rewrite=True):
        self.expt.add_flodict({
            'grad_vx': (np.array([t0]), np.array([shimming[0]])),
            'grad_vy': (np.array([t0]), np.array([shimming[1]])),
            'grad_vz': (np.array([t0]), np.array([shimming[2]])),
            'rx0_en': (np.array([t0]), np.array([0])),
            'rx_gate': (np.array([t0]), np.array([0])),
            'tx0': (np.array([t0]), np.array([0])),
            'tx_gate': (np.array([t0]), np.array([0]))
        }, rewrite)

    def setGradient(self, t0, gAmp, gAxis, rewrite=True):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Set the one gradient to a given value
        Time inputs in us
        Amplitude inputs in Ocra1 units
        """
        if gAxis == 0:
            self.expt.add_flodict({'grad_vx': (np.array([t0]), np.array([gAmp]))}, rewrite)
        elif gAxis == 1:
            self.expt.add_flodict({'grad_vy': (np.array([t0]), np.array([gAmp]))}, rewrite)
        elif gAxis == 2:
            self.expt.add_flodict({'grad_vz': (np.array([t0]), np.array([gAmp]))}, rewrite)

    def saveRawData(self):
        """"
        @author: T. Guallart-Naval, MRILab, i3M, CSIC, Valencia, Spain
        @email: teresa.guallart@tesoroimaging.com
        @modified: J.M. Algarín, MRILab, i3M, CSIC, Spain
        Save the rawData
        """
        # Save data
        dt = datetime.now()
        dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3]
        dt2 = date.today()
        dt2_string = dt2.strftime("%Y.%m.%d")
        if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('experiments/acquisitions/%s' % (dt2_string))
        # if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
        #     os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string))
        self.mapVals['fileName'] = "%s.%s.mat" % (self.mapVals['seqName'], dt_string)
        savemat("experiments/acquisitions/%s/%s.%s.mat" % (dt2_string, self.mapVals['seqName'],
            dt_string), self.mapVals)
        savemat("experiments/acquisitions/%s/%s.%s.mat" % (dt2_string, self.mapNmspc['seqName'],
            dt_string), self.mapNmspc)

    def freqCalibration(self, bw=0.05, dbw = 0.0001):
        """
        @author: J.M. ALgarín
        @contact: josalggui@i3m.upv.es
        :param bw: acquisition bandwdith
        :param dbw: frequency resolution
        :return: the central frequency of the acquired data
        """

        # Create custom inputs
        nPoints = int(bw / dbw)
        larmorFreq = self.mapVals['larmorFreq']
        ov = 10
        bw = bw*ov
        samplingPeriod = 1/bw
        acqTime = 1/dbw
        addRdPoints = 5

        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=False,
                             gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1/samplingPeriod/ov
        acqTime = nPoints/bw  # us
        self.createFreqCalSequence(bw, acqTime)
        rxd, msgs = self.expt.run()
        dataFreqCal = sig.decimate(rxd['rx0'] * 13.788, ov, ftype='fir', zero_phase=True)
        dataFreqCal = dataFreqCal[addRdPoints:nPoints+addRdPoints]
        # Get phase
        angle = np.unwrap(np.angle(dataFreqCal))
        idx = np.argmax(np.abs(dataFreqCal))
        dPhase = angle[idx]
        self.mapVals['drfPhase'] = dPhase
        # Get larmor frequency through fft
        spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataFreqCal))))
        fVector = np.linspace(-bw / 2, bw / 2, num=nPoints, endpoint=False)
        idx = np.argmax(spectrum)
        dfFFT = -fVector[idx]
        larmorFreq += dfFFT
        self.mapVals['larmorFreq'] = larmorFreq # MHz
        print("f0 = %s MHz" % (round(larmorFreq, 5)))
        self.expt.__del__()

        return(larmorFreq)

    def createFreqCalSequence(self, bw, acqTime):
        # Def variables
        shimming = np.array(self.mapVals['shimming'])*1e-4
        rfExTime = self.mapVals['rfExTime'] # us
        rfExAmp = self.mapVals['rfExAmp']
        repetitionTime = self.mapVals['repetitionTime']*1e3 # us
        addRdPoints = 5

        t0 = 20
        tEx = 200

        # Shimming
        self.iniSequence(t0, shimming)

        # Excitation pulse
        t0 = tEx-hw.blkTime-rfExTime/2
        self.rfRecPulse(t0, rfExTime, rfExAmp*np.exp(0.))

        # Rx
        t0 = tEx+rfExTime/2+hw.deadTime
        self.rxGate(t0, acqTime+2*addRdPoints/bw)

        # Finalize sequence
        self.endSequence(repetitionTime)

    def addParameter(self, key='', string='', val=0, field=''):
        if key is not self.mapVals.keys(): self.mapKeys.append(key)
        self.mapNmspc[key] = string
        self.mapVals[key] = val
        self.mapFields[key] = field
        try: self.mapLen[key] = len(val)
        except: self.mapLen[key] = 1

    def getParameter(self, key):
        return(self.mapVals[key])

    def setParameter(self, key, val, unit):
        self.mapVals[key] = val
        self.mapUnits[key] = unit