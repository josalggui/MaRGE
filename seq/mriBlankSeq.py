"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: mri blank sequence with common methods that will be inherited by any sequence
"""

import os
import numpy as np
import configs.hw_config as hw
from datetime import date, datetime
from scipy.io import savemat, loadmat
import experiment as ex
import scipy.signal as sig
import csv
import matplotlib.pyplot as plt

# Import dicom saver
from manager.dicommanager import DICOMImage

class MRIBLANKSEQ:
    # Properties
    mapKeys = []  # keys for the maps
    mapNmspc = {}  # name to show in the gui
    mapVals = {}  # values to show in the gui
    mapFields = {}  # fields to classify the input parameter
    mapLen = {}
    plotSeq = 1  # it plots the sequence
    meta_data = {} # Dictionary to save meta data for dicom file
    # output = []

    def __init__(self):
        self.mapKeys = []
        self.mapNmspc = {}
        self.mapVals = {}
        self.mapFields = {}
        self.mapLen = {}
        self.mapTips = {}
        self.map_units = {}
        self.meta_data = {}
        self.rotations = []
        self.dfovs = []
        self.fovs = []
        self.session = {}
        self.demo = None
        self.mode = None
        self.flo_dict = {'g0': [[],[]],
                         'g1': [[],[]],
                         'g2': [[],[]],
                         'rx0': [[],[]],
                         'rx1': [[],[]],
                         'tx0': [[],[]],
                         'tx1': [[],[]],
                         'ttl0': [[],[]],
                         'ttl1': [[],[]],}



    # *********************************************************************************
    # *********************************************************************************
    # *********************************************************************************

    # Create dictionaries of inputs classified by field (RF, SEQ, IM or OTH)

    @property
    def RFproperties(self):
        # Automatically select the inputs related to RF fields
        out = {}
        tips = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'RF':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
                tips[self.mapNmspc[key]] = [self.mapTips[key]]
        return out, tips

    @property
    def IMproperties(self) -> dict:
        # Automatically select the inputs related to IM fields
        out = {}
        tips = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'IM':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
                tips[self.mapNmspc[key]] = [self.mapTips[key]]
        return out, tips

    @property
    def SEQproperties(self) -> dict:
        # Automatically select the inputs related to SEQ fields
        out = {}
        tips = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'SEQ':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
                tips[self.mapNmspc[key]] = [self.mapTips[key]]
        return out, tips

    @property
    def OTHproperties(self) -> dict:
        # Automatically select the inputs related to OTH fields
        out = {}
        tips = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'OTH':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
                tips[self.mapNmspc[key]] = [self.mapTips[key]]
        return out, tips

    def getFovDisplacement(self):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Get the displacement to apply in the fft reconstruction
        """
        def rotationMatrix(rotation):
            theta = rotation[3]*np.pi/180
            ux = rotation[0]
            uy = rotation[1]
            uz = rotation[2]
            out = np.zeros((3, 3))
            out[0, 0] = np.cos(theta) + ux ** 2 * (1 - np.cos(theta))
            out[0, 1] = ux * uy * (1 - np.cos(theta)) - uz * np.sin(theta)
            out[0, 2] = ux * uz * (1 - np.cos(theta)) + uy * np.sin(theta)
            out[1, 0] = uy * ux * (1 - np.cos(theta)) + uz * np.sin(theta)
            out[1, 1] = np.cos(theta) + uy ** 2 * (1 - np.cos(theta))
            out[1, 2] = uy * uz * (1 - np.cos(theta)) - ux * np.sin(theta)
            out[2, 0] = uz * ux * (1 - np.cos(theta)) - uy * np.sin(theta)
            out[2, 1] = uz * uy * (1 - np.cos(theta)) + ux * np.sin(theta)
            out[2, 2] = np.cos(theta) + uz ** 2 * (1 - np.cos(theta))

            return out
        
        dr = np.reshape(np.array([0, 0, 0]), (3, 1))
        for ii in range(1, len(self.dfovs)):
            Mii = rotationMatrix(self.rotations[ii])
            rii = np.reshape(np.array(self.dfovs[ii]), (3, 1))
            dr = np.dot(Mii, (dr + rii))

        return dr

    def getRotationMatrix(self):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Matrix to rotate through an arbitrary axis
        """
        def rotationMatrix(rotation):
            theta = rotation[3]
            ux = rotation[0]
            uy = rotation[1]
            uz = rotation[2]
            out = np.zeros((3, 3))
            out[0, 0] = np.cos(theta) + ux ** 2 * (1 - np.cos(theta))
            out[0, 1] = ux * uy * (1 - np.cos(theta)) - uz * np.sin(theta)
            out[0, 2] = ux * uz * (1 - np.cos(theta)) + uy * np.sin(theta)
            out[1, 0] = uy * ux * (1 - np.cos(theta)) + uz * np.sin(theta)
            out[1, 1] = np.cos(theta) + uy ** 2 * (1 - np.cos(theta))
            out[1, 2] = uy * uz * (1 - np.cos(theta)) - ux * np.sin(theta)
            out[2, 0] = uz * ux * (1 - np.cos(theta)) - uy * np.sin(theta)
            out[2, 1] = uz * uy * (1 - np.cos(theta)) + ux * np.sin(theta)
            out[2, 2] = np.cos(theta) + uz ** 2 * (1 - np.cos(theta))

            return out

        rotations = []
        for rotation in self.rotations:
            rotations.append(rotationMatrix(rotation))

        l = len(self.rotations)
        rotation = rotations[-1]
        if l>1:
            for ii in range(l-1):
                rotation = np.dot(rotations[-2-ii], rotation)

        return rotation

    def deleteOutput(self):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Delete the output if it exists in the sequence
        """
        # Delete the out attribute if exist
        if hasattr(self, 'output'): delattr(self, 'output')

    def saveParams(self):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Save sequence input parameters into csv
        """
        # Reset the mapVals variable
        self.resetMapVals()

        # Create directory if it does not exist
        if not os.path.exists('experiments/parameterization'):
            os.makedirs('experiments/parameterization')

        # Save csv file with mapVals
        with open('experiments/parameterization/%s_last_parameters.csv' % self.mapVals['seqName'], 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.mapKeys)
            writer.writeheader()
            writer.writerows([self.mapNmspc, self.mapVals])

    def loadParams(self, directory='experiments/parameterization', file=None):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Load sequence parameters from csv
        """
        mapValsOld = self.mapVals
        try:
            if file is None:
                with open('%s/%s_last_parameters.csv' % (directory, self.mapVals['seqName']), 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for l in reader:
                        mapValsNew = l
            else:
                try:
                    if directory == 'calibration':
                        with open('%s/%s_last_parameters.csv' % (directory, file), 'r') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for l in reader:
                                mapValsNew = l
                    else:
                        with open('%s/%s' % (directory, file), 'r') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for l in reader:
                                mapValsNew = l
                except:
                    print("File %s/%s does not exist" % (directory, file))
                    print("File %s/%s loaded" % ("experiments/parameterization", self.mapVals['seqName']))
                    with open('%s/%s_last_parameters.csv' % ("experiments/parameterization", self.mapVals['seqName']), 'r') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for l in reader:
                            mapValsNew = l

            self.mapVals = {}

            # Get key for corresponding modified parameter
            for key in self.mapKeys:
                dataLen = self.mapLen[key]
                valOld = mapValsOld[key]
                try:
                    valNew = mapValsNew[key]
                except:
                    valNew = str(valOld)
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
        except:
            self.mapVals = self.mapVals
            print("\n Warning: no loaded parameters")

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

        def getStepData(data):
            t = data[0]
            s = data[1]
            n = np.size(t)
            tStep = np.zeros(2 * n - 1)
            sStep = np.zeros(2 * n - 1)
            tStep[0::2] = t
            tStep[1::2] = t[1::]
            sStep[0::2] = s
            sStep[1::2] = s[0:-1]
            return [tStep, sStep]

        # Plots
        if self.demo:
            # Plot tx channels
            xData = []
            yData = []
            legend = []

            # tx0_i
            x = self.flo_dict['tx0'][0]
            y = np.real(self.flo_dict['tx0'][1])
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('tx0_i')

            # tx0_q
            x = self.flo_dict['tx0'][0]
            y = np.imag(self.flo_dict['tx0'][1])
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('tx0_q')

            # tx1_i
            x = self.flo_dict['tx1'][0]
            y = np.real(self.flo_dict['tx1'][1])
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('tx1_i')

            # tx1_q
            x = self.flo_dict['tx1'][0]
            y = np.imag(self.flo_dict['tx1'][1])
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('tx1_q')

            plotTx = [xData, yData, legend, 'Tx gate']

            # Plot gradients
            xData = []
            yData = []
            legend = []

            # g0
            x = self.flo_dict['g0'][0]
            y = self.flo_dict['g0'][1]
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('g0')

            # g1
            x = self.flo_dict['g1'][0]
            y = self.flo_dict['g1'][1]
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('g1')

            # g0
            x = self.flo_dict['g2'][0]
            y = self.flo_dict['g2'][1]
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('g2')

            plotGrad = [xData, yData, legend, 'Gradients']

            # Plot readouts
            xData = []
            yData = []
            legend = []

            # rx_0
            x = self.flo_dict['rx0'][0]
            y = self.flo_dict['rx0'][1]
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('rx0_en')

            # rx_1
            x = self.flo_dict['rx1'][0]
            y = self.flo_dict['rx1'][1]
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('rx1_en')

            plotRx = [xData, yData, legend, 'Rx gate']

            return ([plotTx, plotGrad, plotRx])
        else:
            # Get instructions from experiment object
            fd = self.expt.get_flodict()

            # Plot tx channels
            xData = []
            yData = []
            legend = []
            for txl in ['tx0_i', 'tx0_q', 'tx1_i', 'tx1_q']:
                try:
                    dataStep = getStepData(fd[txl])
                    xData.append(dataStep[0] * 1e-3)
                    yData.append(dataStep[1])
                    legend.append(txl)
                except KeyError:
                    continue
            plotTx = [xData, yData, legend, 'Tx gate']

            # Plot gradient channels
            xData = []
            yData = []
            legend = []
            for gradl in self.expt.gradb.keys():
                try:
                    dataStep = getStepData(fd[gradl])
                    xData.append(dataStep[0] * 1e-3)
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
                    xData.append(dataStep[0] * 1e-3)
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
                    xData.append(dataStep[0] * 1e-3)
                    yData.append(dataStep[1])
                    legend.append(iol)
                except KeyError:
                    continue
            plotDigital = [xData, yData, legend, 'Digital']

            return ([plotTx, plotGrad, plotRx, plotDigital])

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

    def decimate(self, dataOver, nRdLines, option='PETRA'):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        This code:
            - deletes the added points that account by the time shift and ramp of the CIC filter
            - preprocess data to avoid oscillations due to 'fir' filter in the decimation
        It must be used if the sequence uses "rxGateSync" to acquire data
        """
        # Preprocess the signal to avoid oscillations due to decimation
        if option=='PETRA':
            dataOver = np.reshape(dataOver, (nRdLines, -1))
            for line in range(nRdLines):
                dataOver[line, 0:hw.addRdPoints * hw.oversamplingFactor] = dataOver[line, hw.addRdPoints * hw.oversamplingFactor]
            dataOver = np.reshape(dataOver, -1)
        elif option=='Normal':
            pass
        self.mapVals['dataOver'] = dataOver

        # Decimate the signal after 'fir' filter
        dataFull = sig.decimate(dataOver[int((hw.oversamplingFactor-1)/2)::], hw.oversamplingFactor, ftype='fir', zero_phase=True)

        # Remove addRdPoints
        nPoints = int(dataFull.shape[0]/nRdLines)-2*hw.addRdPoints
        dataFull = np.reshape(dataFull, (nRdLines, -1))
        dataFull = dataFull[:, hw.addRdPoints:hw.addRdPoints+nPoints]
        dataFull = np.reshape(dataFull, -1)

        return dataFull

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
        txGateTime = np.array(tStart, tStart + hw.blkTime + rfTime)
        txGateAmp = np.array([1, 0])
        self.flo_dict['tx0'][0] = np.concatenate((self.flo_dict['tx0'][0], txTime), axis=0)
        self.flo_dict['tx0'][1] = np.concatenate((self.flo_dict['tx0'][1], txAmp), axis=0)
        self.flo_dict['ttl0' % rxChannel][0] = np.concatenate((self.flo_dict['ttl0' % rxChannel][0], txGateTime), axis=0)
        self.flo_dict['ttl0' % rxChannel][1] = np.concatenate((self.flo_dict['ttl0' % rxChannel][1], txGateAmp), axis=0)

    def rfRecPulse(self, tStart, rfTime, rfAmplitude, rfPhase=0, channel=0):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Rf pulse with square pulse shape
        """
        txTime = np.array([tStart + hw.blkTime, tStart + hw.blkTime + rfTime])
        txAmp = np.array([rfAmplitude * np.exp(1j * rfPhase), 0.])
        txGateTime = np.array([tStart, tStart + hw.blkTime + rfTime])
        txGateAmp = np.array([1, 0])
        self.flo_dict['tx%i' % channel][0] = np.concatenate((self.flo_dict['tx%i' % channel][0], txTime), axis=0)
        self.flo_dict['tx%i' % channel][1] = np.concatenate((self.flo_dict['tx%i' % channel][1], txAmp), axis=0)
        self.flo_dict['ttl0'][0] = np.concatenate((self.flo_dict['ttl0'][0], txGateTime), axis=0)
        self.flo_dict['ttl0'][1] = np.concatenate((self.flo_dict['ttl0'][1], txGateAmp), axis=0)

    def rfRawPulse(self, tStart, rfTime, rfAmplitude, rfPhase=0, channel=0):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        """
        txTime = np.array([tStart, tStart + rfTime])
        txAmp = np.array([rfAmplitude * np.exp(1j * rfPhase), 0.])
        self.flo_dict['tx%i' % channel][0] = np.concatenate((self.flo_dict['tx%i' % channel][0], txTime), axis=0)
        self.flo_dict['tx%i' % channel][1] = np.concatenate((self.flo_dict['tx%i' % channel][1], txAmp), axis=0)

    def rxGate(self, tStart, gateTime, channel=0):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        """
        self.flo_dict['rx%i' % channel][0] = \
            np.concatenate((self.flo_dict['rx%i' % channel][0], np.array([tStart, tStart + gateTime])), axis=0)
        self.flo_dict['rx%i' % channel][1] = \
            np.concatenate((self.flo_dict['rx%i' % channel][1], np.array([1, 0])), axis=0)

    def rxGateSync(self, tStart, gateTime, channel=0):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        This code open the rx channel with additional points to take into account the time shift and ramp of the CIC filter
        It only works with the Experiment class in controller, that inherits from Experiment in marcos_client
        """
        # Generate instructions taking into account the cic filter delay and addRdPoints
        try:
            samplingRate = self.expt.getSamplingRate() / hw.oversamplingFactor # us
        except:
            samplingRate = self.mapVals['samplingPeriod']*1e3 / hw.oversamplingFactor
        t0 = tStart - (hw.addRdPoints * hw.oversamplingFactor - hw.cic_delay_points) * samplingRate # us
        t1 = tStart + (hw.addRdPoints * hw.oversamplingFactor + hw.cic_delay_points) * samplingRate + gateTime # us
        self.flo_dict['rx%i' % channel][0] = \
            np.concatenate((self.flo_dict['rx%i' % channel][0], np.array([t0, t1])), axis=0)
        self.flo_dict['rx%i' % channel][1] = \
            np.concatenate((self.flo_dict['rx%i' % channel][1], np.array([1, 0])), axis=0)

    def ttl(self, tStart, ttlTime, channel=0):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        """
        self.flo_dict['ttl%i' % channel][0] = \
            np.concatenate((self.flo_dict['ttl%i' % channel][0], np.array([tStart, tStart + ttlTime])), axis=0)
        self.flo_dict['ttl%i' % channel][1] = \
            np.concatenate((self.flo_dict['ttl%i' % channel][1], np.array([1, 0])), axis=0)

    def gradTrap(self, tStart, gRiseTime, gFlattopTime, gAmp, gSteps, gAxis, shimming):
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
        a = np.squeeze(np.concatenate((aUp, aDown), axis=0)) / hw.gFactor[gAxis] + shimming[gAxis]

        self.flo_dict['g%i' % gAxis][0] = np.concatenate((self.flo_dict['g%i' % gAxis][0], t), axis=0)
        self.flo_dict['g%i' % gAxis][1] = np.concatenate((self.flo_dict['g%i' % gAxis][1], a), axis=0)

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

    def setGradientRamp(self, tStart, gradRiseTime, nStepsGradRise, g0, gf, gAxis, shimming, rewrite=True):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        gradient ramp from 'g0' to 'gf'
        Time inputs in us
        Amplitude inputs in T/m
        """
        for kk in range(nStepsGradRise):
            tRamp = tStart + gradRiseTime * kk / nStepsGradRise
            gAmp = (g0 + ((gf - g0) * (kk + 1) / nStepsGradRise)) / hw.gFactor[gAxis] + shimming[gAxis]
            self.flo_dict['g%i' % gAxis][0] = np.concatenate((self.flo_dict['g%i' % gAxis][0], np.array([tRamp])), axis=0)
            self.flo_dict['g%i' % gAxis][1] = np.concatenate((self.flo_dict['g%i' % gAxis][1], np.array([gAmp])), axis=0)

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
        self.flo_dict['g0'][0] = np.concatenate((self.flo_dict['g0'][0], np.array([tEnd])), axis=0)
        self.flo_dict['g0'][1] = np.concatenate((self.flo_dict['g0'][1], np.array([0])), axis=0)
        self.flo_dict['g1'][0] = np.concatenate((self.flo_dict['g1'][0], np.array([tEnd])), axis=0)
        self.flo_dict['g1'][1] = np.concatenate((self.flo_dict['g1'][1], np.array([0])), axis=0)
        self.flo_dict['g2'][0] = np.concatenate((self.flo_dict['g2'][0], np.array([tEnd])), axis=0)
        self.flo_dict['g2'][1] = np.concatenate((self.flo_dict['g2'][1], np.array([0])), axis=0)
        self.flo_dict['rx0'][0] = np.concatenate((self.flo_dict['rx0'][0], np.array([tEnd])), axis=0)
        self.flo_dict['rx0'][1] = np.concatenate((self.flo_dict['rx0'][1], np.array([0])), axis=0)
        self.flo_dict['rx1'][0] = np.concatenate((self.flo_dict['rx1'][0], np.array([tEnd])), axis=0)
        self.flo_dict['rx1'][1] = np.concatenate((self.flo_dict['rx1'][1], np.array([0])), axis=0)
        self.flo_dict['tx0'][0] = np.concatenate((self.flo_dict['tx0'][0], np.array([tEnd])), axis=0)
        self.flo_dict['tx0'][1] = np.concatenate((self.flo_dict['tx0'][1], np.array([0])), axis=0)
        self.flo_dict['tx1'][0] = np.concatenate((self.flo_dict['tx1'][0], np.array([tEnd])), axis=0)
        self.flo_dict['tx1'][1] = np.concatenate((self.flo_dict['tx1'][1], np.array([0])), axis=0)
        self.flo_dict['ttl0'][0] = np.concatenate((self.flo_dict['ttl0'][0], np.array([tEnd])), axis=0)
        self.flo_dict['ttl0'][1] = np.concatenate((self.flo_dict['ttl0'][1], np.array([0])), axis=0)
        self.flo_dict['ttl1'][0] = np.concatenate((self.flo_dict['ttl1'][0], np.array([tEnd])), axis=0)
        self.flo_dict['ttl1'][1] = np.concatenate((self.flo_dict['ttl1'][1], np.array([0])), axis=0)

    def iniSequence(self, t0, shimming):
        self.flo_dict['g0'][0] = np.array([t0])
        self.flo_dict['g0'][1] = np.array([shimming[0]])
        self.flo_dict['g1'][0] = np.array([t0])
        self.flo_dict['g1'][1] = np.array([shimming[1]])
        self.flo_dict['g2'][0] = np.array([t0])
        self.flo_dict['g2'][1] = np.array([shimming[2]])
        self.flo_dict['rx0'][0] = np.array([t0])
        self.flo_dict['rx0'][1] = np.array([0])
        self.flo_dict['rx1'][0] = np.array([t0])
        self.flo_dict['rx1'][1] = np.array([0])
        self.flo_dict['tx0'][0] = np.array([t0])
        self.flo_dict['tx0'][1] = np.array([0])
        self.flo_dict['tx1'][0] = np.array([t0])
        self.flo_dict['tx1'][1] = np.array([0])
        self.flo_dict['ttl0'][0] = np.array([t0])
        self.flo_dict['ttl0'][1] = np.array([0])
        self.flo_dict['ttl1'][0] = np.array([t0])
        self.flo_dict['ttl1'][1] = np.array([0])

    def setGradient(self, t0, gAmp, gAxis, rewrite=True):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Set the one gradient to a given value
        Time inputs in us
        Amplitude inputs in Ocra1 units
        """
        self.flo_dict['g%i' % gAxis][0] = np.concatenate((self.flo_dict['g%i' % gAxis][0], np.array([t0])), axis=0)
        self.flo_dict['g%i' % gAxis][1] = np.concatenate((self.flo_dict['g%i' % gAxis][1], np.array([gAmp])), axis=0)

    def floDict2Exp(self, rewrite=True):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Check for errors and add instructions to red pitaya if no errors are found
        """
        # Check errors:
        for key in self.flo_dict.keys():
            item = self.flo_dict[key]
            dt = item[0][1::]-item[0][0:-1]
            if (dt<=0).any():
                print("\n%s timing error" % key)
                return False
            if (item[1]>1).any() or (item[1]<-1).any():
                print("\n%s amplitude error" % key)
                return False

        self.expt.add_flodict({'grad_vx': (self.flo_dict['g0'][0], self.flo_dict['g0'][1]),
                               'grad_vy': (self.flo_dict['g1'][0], self.flo_dict['g1'][1]),
                               'grad_vz': (self.flo_dict['g2'][0], self.flo_dict['g2'][1]),
                               'rx0_en': (self.flo_dict['rx0'][0], self.flo_dict['rx0'][1]),
                               'rx1_en': (self.flo_dict['rx1'][0], self.flo_dict['rx1'][1]),
                               'tx0': (self.flo_dict['tx0'][0], self.flo_dict['tx0'][1]),
                               'tx1': (self.flo_dict['tx1'][0], self.flo_dict['tx1'][1]),
                               'tx_gate': (self.flo_dict['ttl0'][0], self.flo_dict['ttl0'][1]),
                               'rx_gate': (self.flo_dict['ttl1'][0], self.flo_dict['ttl1'][1]),
                               }, rewrite)
        return True

    def saveRawData(self):
        """"
        @author: T. Guallart-Naval, Tesoro Imaging S.L., Valencia, Spain
        @email: teresa.guallart@tesoroimaging.com
        @modified: J.M. Algarín, MRILab, i3M, CSIC, Spain
        Save the rawData
        """
        # Get directory
        if 'directory' in self.session.keys():
            directory = self.session['directory']
        else:
            dt2 = date.today()
            date_string = dt2.strftime("%Y.%m.%d")
            directory = 'experiments/acquisitions/%s' % (date_string)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # generate directories for mat, csv and dcm files
        directory_mat = directory + '/mat'
        directory_csv = directory + '/csv'
        directory_dcm = directory + '/dcm'
        if not os.path.exists(directory+'/mat'):
            os.makedirs(directory_mat)
        if not os.path.exists(directory+'/csv'):
            os.makedirs(directory_csv)
        if not os.path.exists(directory+'/dcm'):
            os.makedirs(directory_dcm)

        # Generate filename
        name = datetime.now()
        name_string = name.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3]
        self.mapVals['name_string'] = name_string
        if hasattr(self, 'raw_data_name'):
            file_name = "%s.%s" % (self.raw_data_name, name_string)
        else:
            file_name = "%s.%s" % (self.mapVals['seqName'], name_string)
        self.mapVals['fileName'] = "%s.mat" % file_name

        # Save mat file with the outputs
        savemat("%s/%s.mat" % (directory_mat, file_name), self.mapVals)

        # Save csv with input parameters
        with open('%s/%s.csv' % (directory_csv, file_name), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.mapKeys)
            writer.writeheader()
            mapVals = {}
            for key in self.mapKeys:  # take only the inputs from mapVals
                mapVals[key] = self.mapVals[key]
            writer.writerows([self.mapNmspc, mapVals])

        # Save dcm with the final image
        if (len(self.output) > 0) and (self.output[0]['widget'] == 'image') and (self.mode==None):
            self.image2Dicom(fileName = "%s/%s.dcm" % (directory_dcm, file_name))

    def image2Dicom(self, fileName):
        """"
        @author: F. Juan-Llorís, PhysioMRI S.L., Valencia, Spain
        @email: franc.juan@physiomri.com
        @modified: J.M. Algarín, MRILab, i3M, CSIC, Spain
        @modified: T. Guallart-Naval, MRILab, i3M, CSIC, Spain
        Save the dicom
        """

        # Create dicom object
        dicom_image = DICOMImage()

        # Save image into dicom object
        try:
            dicom_image.meta_data["PixelData"] = self.meta_data["PixelData"]
        except KeyError:
            image = self.output[0]['data']
            dicom_image.meta_data["PixelData"] = image.astype(np.int16).tobytes()
            # If it is a 3d image
            if len(image.shape) > 2:
                # Obtener dimensiones
                slices, rows, columns = image.shape
                dicom_image.meta_data["Columns"] = columns
                dicom_image.meta_data["Rows"] = rows
                dicom_image.meta_data["NumberOfSlices"] = slices
                dicom_image.meta_data["NumberOfFrames"] = slices
            # if it is a 2d image
            else:
                # Obtener dimensiones
                rows, columns = image.shape
                dicom_image.meta_data["Columns"] = columns
                dicom_image.meta_data["Rows"] = rows
                dicom_image.meta_data["NumberOfSlices"] = 1
                dicom_image.meta_data["NumberOfFrames"] = 1

        # Date and time
        current_time = datetime.now()
        self.meta_data["StudyDate"] = current_time.strftime("%Y%m%d")
        self.meta_data["StudyTime"] = current_time.strftime("%H%M%S")

        # More DICOM tags
        self.meta_data["PatientName"] = self.session["subject_id"]
        self.meta_data["PatientSex"] = " "
        self.meta_data["StudyID"] = self.session["subject_id"]
        self.meta_data["InstitutionName"] = self.session["scanner"]
        self.meta_data["ImageComments"] = " "
        self.meta_data["PatientID"] = self.session["subject_id"]
        self.meta_data["SOPInstanceUID"] = self.mapVals['name_string']
        self.meta_data["SeriesDescription"] = self.raw_data_name
        self.session['seriesNumber'] = self.session['seriesNumber'] + 1
        self.meta_data["SeriesNumber"] = self.session['seriesNumber']
        # Full dinamic window
        #self.meta_data["WindowWidth"] = 26373
        #self.meta_data["WindowCenter"] = 13194


        dicom_image.meta_data = dicom_image.meta_data | self.meta_data

        # Save meta_data dictionary into dicom object metadata (Standard DICOM 3.0)
        dicom_image.image2Dicom()

        # Save dicom file
        dicom_image.save(fileName)

    def freqCalibration(self, bw=0.05, dbw=0.0001):
        """
        @author: J.M. ALgarín
        @contact: josalggui@i3m.upv.es
        :param bw: acquisition bandwdith
        :param dbw: frequency resolution
        :return: the central frequency of the acquired data
        """

        # Create custom inputs
        nPoints = int(bw / dbw)
        ov = 10
        bw = bw * ov
        samplingPeriod = 1 / bw
        acqTime = 1 / dbw
        addRdPoints = 5

        self.expt = ex.Experiment(lo_freq=hw.larmorFreq, rx_t=samplingPeriod, init_gpa=False,
                                  gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1 / samplingPeriod / ov
        acqTime = nPoints / bw  # us
        self.createFreqCalSequence(bw, acqTime)
        rxd, msgs = self.expt.run()
        dataFreqCal = sig.decimate(rxd['rx0'] * 13.788, ov, ftype='fir', zero_phase=True)
        dataFreqCal = dataFreqCal[addRdPoints:nPoints + addRdPoints]
        # Get larmor frequency through fft
        spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataFreqCal))))
        fVector = np.linspace(-bw / 2, bw / 2, num=nPoints, endpoint=False)
        idx = np.argmax(spectrum)
        dfFFT = -fVector[idx]
        hw.larmorFreq += dfFFT
        self.mapVals['larmorFreq'] = np.round(hw.larmorFreq, decimals=6)  # MHz
        print("f0 = %s MHz" % (round(hw.larmorFreq, 5)))
        self.expt.__del__()

        return (hw.larmorFreq)

    def createFreqCalSequence(self, bw, acqTime):
        # Def variables
        shimming = np.array(self.mapVals['shimming']) * 1e-4
        rfExTime = self.mapVals['rfExTime']  # us
        rfExAmp = self.mapVals['rfExAmp']
        repetitionTime = self.mapVals['repetitionTime'] * 1e3  # us
        addRdPoints = 5

        t0 = 20
        tEx = 200

        # Shimming
        self.iniSequence(t0, shimming)

        # Excitation pulse
        t0 = tEx - hw.blkTime - rfExTime / 2
        self.rfRecPulse(t0, rfExTime, rfExAmp * np.exp(0.))

        # Rx
        t0 = tEx + rfExTime / 2 + hw.deadTime
        self.rxGate(t0, acqTime + 2 * addRdPoints / bw)

        # Finalize sequence
        self.endSequence(repetitionTime)

    def addParameter(self, key='', string='', val=0, units=True, field='', tip=None):
        if key is not self.mapVals.keys(): self.mapKeys.append(key)
        self.mapNmspc[key] = string
        self.mapVals[key] = val
        self.mapFields[key] = field
        self.mapTips[key] = tip
        self.map_units[key] = units
        try:
            self.mapLen[key] = len(val)
        except:
            self.mapLen[key] = 1

    def sequenceAtributes(self):
        # Add input parameters to the self
        for key in self.mapKeys:
            if isinstance(self.mapVals[key], list):
                setattr(self, key, np.array([element * self.map_units[key] for element in self.mapVals[key]]))
            else:
                setattr(self, key, self.mapVals[key]*self.map_units[key])

            x = 0

    def plotResults(self):
        # Get number of collumns and rows
        cols = 1
        rows = 1
        for item in self.output:
            if item['row'] + 1 > rows:
                rows += 1
            if item['col'] + 1 > cols:
                cols += 1

        # Create plot window
        fig, axes = plt.subplots(rows, cols, figsize=(10, 5))

        # Insert plots
        plot = 0
        for item in self.output:
            if item['widget'] == 'image':
                nz, ny, nx = item['data'].shape
                plt.subplot(rows, cols, plot+1)
                plt.imshow(item['data'][int(nz/2), :, :], cmap='gray')
                plt.title(item['title'])
                plt.xlabel(item['xLabel'])
                plt.ylabel(item['yLabel'])
            elif item['widget'] == 'curve':
                plt.subplot(rows, cols, plot+1)
                n = 0
                for y_data in item['yData']:
                    plt.plot(item['xData'], y_data, label=item['legend'][n])
                    n += 1
                plt.title(item['title'])
                plt.xlabel(item['xLabel'])
                plt.ylabel(item['yLabel'])
            plot += 1

        # Set figure title
        plt.suptitle(self.mapVals['fileName'])

        # Adjust the layout to prevent overlapping titles
        plt.tight_layout()

        # Show the plot
        plt.show()


    def getParameter(self, key):
        return (self.mapVals[key])

    def setParameter(self, key, val, unit):
        self.mapVals[key] = val
        self.mapUnits[key] = unit
