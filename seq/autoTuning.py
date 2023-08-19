"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: code to obtain a good combination of tuning/matching
Specific hardware from MRILab @ i3M is required
"""

import os
import sys

# *****************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char == '\\' or char == '/') and path[ii + 1:ii + 14] == 'PhysioMRI_GUI':
        sys.path.append(path[0:ii + 1] + 'PhysioMRI_GUI')
        sys.path.append(path[0:ii + 1] + 'marcos_client')
    ii += 1
# ******************************************************************************
from PyQt5.QtCore import QThreadPool
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import serial
import time
import copy
import configs.hw_config as hw
import autotuning.autotuning as autotuning


class AutoTuning(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(AutoTuning, self).__init__()
        # Input the parameters
        self.larmorFreq = None
        self.statesXm = None
        self.statesCm = None
        self.statesCt = None
        self.statesXs = None
        self.states = None
        self.statesCs = None
        self.test = None
        self.switch = None
        self.matching = None
        self.tuning = None
        self.series = None
        self.seriesTarget = None
        self.state0 = None
        self.frequencies = None
        self.expt = None
        self.threadpool = None

        # Connect to Arduino and set the initial state
        self.arduino = autotuning.Arduino()
        self.arduino.connect()
        self.arduino.send('1111111111111111')

        # Connect to VNA
        self.vna = autotuning.VNA()
        self.vna.connect()

        # Parameters
        self.addParameter(key='seqName', string='AutoTuningInfo', val='AutoTuning')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.066, field='RF')
        self.addParameter(key='seriesTarget', string='Series target (Ohms)', val=50.0, field='RF')
        self.addParameter(key='iterations', string='Max iterations', val=10, field='RF')
        self.addParameter(key='series', string='Series capacitor', val='00000', field='RF')
        self.addParameter(key='tuning', string='Tuning capacitor', val='00000', field='RF')
        self.addParameter(key='matching', string='Matching capacitor', val='00000', field='RF')
        self.addParameter(key='switch', string='Switch', val='0', field='RF')
        self.addParameter(key='test', string='Test', val=0, field='RF')

        self.arduino.send(self.mapVals['series'] + self.mapVals['tuning'] + self.mapVals['matching'] + "1")

    def sequenceInfo(self):
        print("\nRF automatic impedance matching")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Look for the best combination of tuning/matching.")
        print("Specific hardware from MRILab @ i3M is required.\n")

    def sequenceTime(self):
        return 0  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        self.demo = demo

        if self.arduino.device is None:
            print("\nNo Arduino found for auto-tuning.")
            return False

        if self.vna.device is None:
            print("\nNo nanoVNA found for auto-tuning.")
            print("Only test mode.")
            self.test = 1

        if self.test == 0:
            self.runAutoTuning()
        else:
            self.arduino.send(self.series + self.tuning + self.matching + self.switch)
            if self.vna.device is not None:
                s11, impedance = self.vna.getS11(self.larmorFreq)
                r0 = impedance.real
                x0 = impedance.imag
                print("\nInput impedance:")
                print("R = %0.2f Ohms" % r0)
                print("X = %0.2f Ohms" % x0)
            else:
                return False

        return True

    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        f_vec = self.vna.getFrequency()
        s_vec = self.vna.getData()
        s11 = self.mapVals['s11']

        # Plot signal versus time
        result1 = {'widget': 'curve',
                   'xData': f_vec - self.larmorFreq * 1e-6,
                   'yData': [20 * np.log10(np.abs(s_vec))],
                   'xLabel': 'Frequency (MHz)',
                   'yLabel': 'S11 (dB)',
                   'title': 'Reflection coefficient',
                   'legend': [''],
                   'row': 0,
                   'col': 0}

        result2 = {'widget': 'smith',
                   'xData': [np.real(s11), np.real(s_vec)],
                   'yData': [np.imag(s11), np.imag(s_vec)],
                   'xLabel': 'Real(S11)',
                   'yLabel': 'Imag(S11)',
                   'title': 'Smith chart',
                   'legend': [''],
                   'row': 0,
                   'col': 1}

        self.output = [result1]

        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()

        return self.output

    def runAutoTuning(self):
        start = time.time()
        nCap = 5

        # Combinations
        self.states = [''] * 2 ** nCap
        for state in range(2 ** nCap):
            prov = format(state, f'0{nCap}b')
            self.states[state] = ''.join('1' if bit == '0' else '0' for bit in prov)

        # Series reactance
        cs = np.array([np.Inf, 8, 3.9, 1.8, 1]) * 1e-9
        self.statesCs = np.zeros(2 ** nCap)
        self.statesXs = np.zeros(2 ** nCap)
        for state in range(2 ** nCap):
            for c in range(nCap):
                if int(self.states[state][c]) == 0:
                    self.statesCs[state] += cs[c]
            if self.statesCs[state] == 0:
                self.statesXs[state] = np.Inf
            else:
                self.statesXs[state] = -1 / (2 * np.pi * hw.larmorFreq * 1e6 * self.statesCs[state])

        # Tuning capacitor
        ct = np.array([326, 174, 87, 44, 26]) * 1e-12
        self.statesCt = np.zeros(2 ** nCap)
        for state in range(2 ** nCap):
            for c in range(nCap):
                if int(self.states[state][c]) == 0:
                    self.statesCt[state] += ct[c]

        # Matching capacitors
        cm = np.array([np.Inf, 500, 262, 142, 75]) * 1e-12
        self.statesCm = np.zeros(2 ** nCap)
        self.statesXm = np.zeros(2 ** nCap)
        for state in range(2 ** nCap):
            for c in range(nCap):
                if int(self.states[state][c]) == 0:
                    self.statesCm[state] += cm[c]
            if self.statesCm[state] == 0:
                self.statesXm[state] = np.Inf
            else:
                self.statesXm[state] = -1 / (2 * np.pi * hw.larmorFreq * 1e6 * self.statesCm[state])

        # Get initial impedance
        self.arduino.send('0111111111011110')
        s11, impedance = self.vna.getS11(self.larmorFreq)
        r0 = impedance.real
        x0 = impedance.imag
        print("\nInput impedance:")
        print("R = %0.2f Ohms" % r0)
        print("X = %0.2f Ohms" % x0)

        # if x0 > self.seriesTarget:
        #     stateCs = self.getCs(0, 17, "1")
        # else:
        #     stateCs = 16

        stateCt = self.getCtS(16, 16, 16)

        stateCm = self.getCmS(8, 16, stateCt)

        stateCs = self.getCsS(8, stateCt, stateCm)

        stateCt = self.getCtS(stateCt, stateCs, stateCm)

        stateCm = self.getCmS(stateCm, stateCs, stateCt)

        stateCs, s11 = self.getCsS(stateCs, stateCt, stateCm)

        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + "1")

        self.mapVals['series'] = self.states[stateCs]
        self.mapVals['tuning'] = self.states[stateCt]
        self.mapVals['matching'] = self.states[stateCm]
        self.mapVals['s11'] = s11
        s11r = np.real(s11)
        s11i = np.imag(s11)
        z = 50*(1+s11)/(1-s11)
        r = np.real(z)
        x = np.imag(z)

        print("\nFinal state")
        print(self.states[stateCs] + self.states[stateCt] + self.states[stateCm])
        if s11i >= 0:
            print("S11 = %0.3f + j %0.3f" % (s11r, s11i))
        else:
            print("S11 = %0.3f - j %0.3f" % (s11r, np.abs(s11i)))
        if x >= 0:
            print("Z = %0.1f + j %0.1f Ohms" % (r, x))
        else:
            print("Z = %0.1f - j %0.1f Ohms" % (r, np.abs(x)))

    def getCsS(self, n0, stateCt, stateCm):
        print("\nObtaining series capacitor...")
        n = [n0]

        # First measurement
        self.arduino.send(self.states[n[-1]] + self.states[stateCt] + self.states[stateCm] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        s11dB = [20 * np.log10(np.abs(s11))]
        print("S11 = %0.2f dB" % s11dB[-1])

        # Second measurement
        if n[-1] == 16:
            step = - 1
        else:
            step = + 1
        n.append(n[-1] + step)
        self.arduino.send(self.states[n[-1]] + self.states[stateCt] + self.states[stateCm] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        s11dB.append(20 * np.log10(np.abs(s11)))
        print("S11 = %0.2f dB" % s11dB[-1])

        # Check the direction to follow
        if s11dB[-1] < s11dB[-2]:
            pass
        else:
            step = -step
            n.reverse()
            s11dB.reverse()

        # Sweep until the S11 starts to go up
        while s11dB[-1] < s11dB[-2] and 0 <= n[-1] + step <= 16:
            n.append(n[-1] + step)
            self.arduino.send(self.states[n[-1]] + self.states[stateCt] + self.states[stateCm] + "0")
            s11, impedance = self.vna.getS11(self.larmorFreq)
            s11dB.append(20 * np.log10(np.abs(s11)))
            print("S11 = %0.2f dB" % s11dB[-1])

        # Set the best state
        stateCs = n[np.argmin(np.array(s11dB))]
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        r0 = impedance.real
        x0 = impedance.imag
        print("Best state:")
        print(self.states[stateCs])
        print("%0.0f pF" % (self.statesCs[stateCs] * 1e12))
        print("S11 = %0.2f dB" % (20 * np.log10(np.abs(s11))))
        print("R = %0.2f Ohms" % r0)
        print("X = %0.2f Ohms" % x0)
        return stateCs, s11

    def getCsZ(self, stateCt, stateCm, auto):
        # Sweep series impedances until reactance goes higher than 50 Ohms
        print("\nObtaining series capacitor...")
        n = 0
        x0 = [0.0]
        while x0[-1] < self.seriesTarget and n < 31:
            n += 1
            self.arduino.write((self.states[n] + self.states[stateCt] + self.states[stateCm] + "1").encode())
            while self.arduino.in_waiting == 0:
                time.sleep(0.1)
            result = self.arduino.readline()
            s11 = np.array(
                [float(value) for value in
                 self.vna.readValues("data 0")[self.idf].split(" ")])  # "data 0"->S11, "data 1"->S21
            s11 = s11[0] + s11[1] * 1j
            impedance = 50 * (1. + s11) / (1. - s11)
            x0.append(impedance.imag)
            print("S11 = %0.2f dB" % (20 * np.log10(np.abs(s11))))

        # Select the value with reactance closest to 50 Ohms
        stateCs = np.argmin(np.abs(np.array(x0) - self.seriesTarget))
        self.arduino.write((self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + auto).encode())
        while self.arduino.in_waiting == 0:
            time.sleep(0.1)
        result = self.arduino.readline()
        s11 = np.array(
            [float(value) for value in
             self.vna.readValues("data 0")[self.idf].split(" ")])  # "data 0"->S11, "data 1"->S21
        s11 = s11[0] + s11[1] * 1j
        impedance = 50 * (1. + s11) / (1. - s11)
        r0 = impedance.real
        x0 = impedance.imag
        print("Best state:")
        print(self.states[stateCs])
        print("%0.0f pF" % (self.statesCs[stateCs] * 1e12))
        print("S11 = %0.2f dB" % (20 * np.log10(np.abs(s11))))
        print("R = %0.2f Ohms" % r0)
        print("X = %0.2f Ohms" % x0)

        return stateCs

    def getCtS(self, n0, stateCs, stateCm):
        # Sweep tuning capacitances until find a minimum in S11
        print("\nObtaining tuning capacitor...")
        n = [n0]

        # First measurement
        self.arduino.send(self.states[stateCs] + self.states[n[-1]] + self.states[stateCm] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        s11dB = [20 * np.log10(np.abs(s11))]
        print("S11 = %0.2f dB" % s11dB[-1])

        # Second measurement
        if n[-1] == 31:
            step = - 1
        else:
            step = + 1
        n.append(n[-1] + step)
        self.arduino.send(self.states[stateCs] + self.states[n[-1]] + self.states[stateCm] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        s11dB.append(20 * np.log10(np.abs(s11)))
        print("S11 = %0.2f dB" % s11dB[-1])

        # Check the direction to follow
        if s11dB[-1] < s11dB[-2]:
            pass
        else:
            step = -step
            n.reverse()
            s11dB.reverse()

        # Sweep until the S11 starts to go up
        while s11dB[-1] < s11dB[-2] and n[-1] < 31 and n[-1] > 0:
            n.append(n[-1] + step)
            self.arduino.send(self.states[stateCs] + self.states[n[-1]] + self.states[stateCm] + "0")
            s11, impedance = self.vna.getS11(self.larmorFreq)
            s11dB.append(20 * np.log10(np.abs(s11)))
            print("S11 = %0.2f dB" % s11dB[-1])

        # Set the best state
        stateCt = n[np.argmin(np.array(s11dB))]
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        r0 = impedance.real
        x0 = impedance.imag
        print("Best state:")
        print(self.states[stateCt])
        print("%0.0f pF" % (self.statesCt[stateCt] * 1e12))
        print("S11 = %0.2f dB" % (20 * np.log10(np.abs(s11))))
        print("R = %0.2f Ohms" % r0)
        print("X = %0.2f Ohms" % x0)
        return stateCt, s11

    def getCtZ(self, n0, stateCs, stateCm, auto):
        # Sweep tuning capacitances until resistance goes higher than 50 Ohms
        print("\nObtaining tuning capacitor...")
        n = copy.copy(n0)
        r0 = [0.0]
        while r0[-1] < 50.0 and n < 31:
            n += 1
            self.arduino.send(self.states[stateCs] + self.states[n] + self.states[stateCm] + "1")
            s11 = np.array(
                [float(value) for value in
                 self.vna.readValues("data 0")[self.idf].split(" ")])  # "data 0"->S11, "data 1"->S21
            s11 = s11[0] + s11[1] * 1j
            impedance = 50 * (1. + s11) / (1. - s11)
            r0.append(impedance.real)
            print("S11 = %0.2f dB" % (20 * np.log10(np.abs(s11))))

        # Select the value with reactance closest to 50 Ohms
        stateCt = n0 + np.argmin(np.abs(np.array(r0) - 50.0))
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + auto)
        s11 = np.array(
            [float(value) for value in
             self.vna.readValues("data 0")[self.idf].split(" ")])  # "data 0"->S11, "data 1"->S21
        s11 = s11[0] + s11[1] * 1j
        impedance = 50 * (1. + s11) / (1. - s11)
        r0 = impedance.real
        x0 = impedance.imag
        print("Best state:")
        print(self.states[stateCt])
        print("%0.0f pF" % (self.statesCt[stateCt] * 1e12))
        print("S11 = %0.2f dB" % (20 * np.log10(np.abs(s11))))
        print("R = %0.2f Ohms" % r0)
        print("X = %0.2f Ohms" % x0)

        return stateCt

    def getCmS(self, n0, stateCs, stateCt):
        # Sweep matching capacitances until reactance goes negative
        print("\nObtaining matching capacitor...")
        n = [n0]

        # First measurement
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[n[-1]] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        s11dB = [20 * np.log10(np.abs(s11))]
        print("S11 = %0.2f dB" % s11dB[-1])

        # Second measurement
        if n[-1] == 0:
            step = + 1
        else:
            step = - 1
        n.append(n[-1] + step)
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[n[-1]] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        s11dB.append(20 * np.log10(np.abs(s11)))
        print("S11 = %0.2f dB" % s11dB[-1])

        # Check the direction to follow
        if s11dB[-1] < s11dB[-2]:
            step = step
        else:
            step = -step
            n.reverse()
            s11dB.reverse()

        # Sweep until the S11 starts to go up
        while s11dB[-1] < s11dB[-2] and 0 <= n[-1] + step <= 16:
            n.append(n[-1] + step)
            self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[n[-1]] + "0")
            s11, impedance = self.vna.getS11(self.larmorFreq)
            s11dB.append(20 * np.log10(np.abs(s11)))
            print("S11 = %0.2f dB" % s11dB[-1])

        stateCm = n[np.argmin(np.array(s11dB))]
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        r0 = impedance.real
        x0 = impedance.imag
        print("Best state:")
        print(self.states[stateCm])
        print("%0.0f pF" % (self.statesCm[stateCm] * 1e12))
        print("S11 = %0.2f dB" % (20 * np.log10(np.abs(s11))))
        print("R = %0.2f Ohms" % r0)
        print("X = %0.2f Ohms" % x0)
        return stateCm, s11

    def getCmZ(self, n0, stateCs, stateCt, auto):
        # Sweep matching capacitances until reactance goes negative
        print("\nObtaining matching capacitor...")
        n = copy.copy(n0)
        x0 = [10000.0]
        while x0[-1] > 0.0 and n > 0:
            n -= 1
            self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[n] + "1")
            s11 = np.array(
                [float(value) for value in
                 self.vna.readValues("data 0")[self.idf].split(" ")])  # "data 0"->S11, "data 1"->S21
            s11 = s11[0] + s11[1] * 1j
            impedance = 50 * (1. + s11) / (1. - s11)
            x0.append(impedance.imag)
            print("S11 = %0.2f dB" % (20 * np.log10(np.abs(s11))))

        # Select the value with reactance closest to 50 Ohms
        stateCm = n0 - np.argmin(np.abs(np.array(x0)))
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + "1")
        s11 = np.array(
            [float(value) for value in
             self.vna.readValues("data 0")[self.idf].split(" ")])  # "data 0"->S11, "data 1"->S21
        s11 = s11[0] + s11[1] * 1j
        impedance = 50 * (1. + s11) / (1. - s11)
        r0 = impedance.real
        x0 = impedance.imag
        print("Best state:")
        print(self.states[stateCm])
        print("%0.0f pF" % (self.statesCm[stateCm] * 1e12))
        print("S11 = %0.2f dB" % (20 * np.log10(np.abs(s11))))
        print("R = %0.2f Ohms" % r0)
        print("X = %0.2f Ohms" % x0)

        return stateCm


if __name__ == '__main__':
    seq = AutoTuning()
    seq.sequenceAtributes()
    seq.sequenceRun()
    seq.sequenceAnalysis(mode='Standalone')
