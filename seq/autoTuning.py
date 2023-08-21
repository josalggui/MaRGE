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
        self.seriesTarget = 50
        self.state0 = None
        self.frequencies = None
        self.expt = None
        self.threadpool = None
        self.s11 = []

        # Parameters
        self.addParameter(key='seqName', string='AutoTuningInfo', val='AutoTuning')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.059, field='IM')
        self.addParameter(key='series', string='Series capacitor', val='01111', field='IM')
        self.addParameter(key='tuning', string='Tuning capacitor', val='11111', field='IM')
        self.addParameter(key='matching', string='Matching capacitor', val='01111', field='IM')
        self.addParameter(key='switch', string='Switch', val='0', field='IM')
        self.addParameter(key='test', string='Test', val='auto', field='IM',
                          tip='Choose one option: auto, manual, series, tunin or matching')

        # Connect to Arduino and set the initial state
        self.arduino = autotuning.Arduino()
        self.arduino.connect()
        self.arduino.send(self.mapVals['series'] + self.mapVals['tuning'] + self.mapVals['matching'] + "1")

        # Connect to VNA
        self.vna = autotuning.VNA()
        self.vna.connect()

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
        self.s11 = []

        if self.arduino.device is None:
            print("\nNo Arduino found for auto-tuning.")
            return False

        if self.vna.device is None:
            print("\nNo nanoVNA found for auto-tuning.")
            print("Only test mode.")

        if self.test == 'auto':
            return self.runAutoTuning()
        elif self.test == 'manual':
            return self.runManual()
        elif self.test == 'series':
            return self.sweepSeries()
        elif self.test == 'matching':
            return self.sweepMatching()
        elif self.test == 'tuning':
            return self.sweepTuning()
        else:
            print("\nIncorrect test mode.")
            return False

    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        s11 = np.array(self.s11)
        if self.test == 'series' or self.test == 'matching':
            states = np.linspace(1, 16, 16)
        elif self.test == 'tuning':
            states = np.linspace(0, 31, 32)
        f_vec = self.vna.getFrequency()
        s_vec = self.vna.getData()

        if self.test == 'manual':
            s11 = np.concatenate((s11, s11), axis=0)
        if self.test == 'manual' or self.test == 'auto':

            # Plot smith chart
            result1 = {'widget': 'smith',
                       'xData': [np.real(s11), np.real(s_vec)],
                       'yData': [np.imag(s11), np.imag(s_vec)],
                       'xLabel': 'Real(S11)',
                       'yLabel': 'Imag(S11)',
                       'title': 'Smith chart',
                       'legend': ['', ''],
                       'row': 0,
                       'col': 0}

            # Plot reflection coefficient
            result2 = {'widget': 'curve',
                       'xData': (f_vec - self.larmorFreq) * 1e3,
                       'yData': [20 * np.log10(np.abs(s_vec))],
                       'xLabel': 'Frequency (kHz)',
                       'yLabel': 'S11 (dB)',
                       'title': 'Reflection coefficient',
                       'legend': [''],
                       'row': 0,
                       'col': 1}
        else:
            # Plot smith chart
            result1 = {'widget': 'smith',
                       'xData': [np.real(s11), np.real(s11)],
                       'yData': [np.imag(s11), np.imag(s11)],
                       'xLabel': 'Real(S11)',
                       'yLabel': 'Imag(S11)',
                       'title': 'Smith chart',
                       'legend': ['', ''],
                       'row': 0,
                       'col': 0}

            # Plot reflection coefficient
            result2 = {'widget': 'curve',
                       'xData': states,
                       'yData': [20 * np.log10(np.abs(s11))],
                       'xLabel': 'State',
                       'yLabel': 'S11 (dB)',
                       'title': 'Reflection coefficient',
                       'legend': [''],
                       'row': 0,
                       'col': 1}

        self.output = [result1, result2]

        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()

        return self.output

    def runAutoTuning(self):
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

        # Get initial state index
        stateCs = self.states.index(self.series)
        stateCt = self.states.index(self.tuning)
        stateCm = self.states.index(self.matching)

        # Check current status
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        s11dB = 20 * np.log10(np.abs(s11))

        # Iterate to find the best configuration
        if s11dB > -20:
            iteration = 1
            stateCs = self.getCsZ(16, 0, 16)
            stateCt = self.getCtZ(stateCs, 0, 16)
            stateCm = self.getCmZ(stateCs, stateCt, 16)
            while 20*np.log10(np.abs(self.s11[-1])) > -20 and iteration < 5:
                stateCs = self.getCsZ2(stateCs, stateCt, stateCm)
                stateCt = self.getCtZ(stateCs, stateCt, stateCm)
                stateCm = self.getCmZ(stateCs, stateCt, stateCm)
                iteration += 1

        # Measure the best state
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        s11dB = 20 * np.log10(np.abs(s11))
        r = impedance.real
        x = impedance.imag
        print("\nBest state:")
        print(self.states[stateCs] + self.states[stateCt] + self.states[stateCm])
        print("S11 = %0.2f dB" % s11dB)
        print("R = %0.2f Ohms" % r)
        print("X = %0.2f Ohms" % x)
        self.mapVals['series'] = self.states[stateCs]
        self.mapVals['tuning'] = self.states[stateCt]
        self.mapVals['matching'] = self.states[stateCm]
        self.mapVals['s11'] = s11

        # Connect the system to TxRx switch
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + "1")

        # Data to sweep sequence
        self.mapVals['sampledPoint'] = s11dB

        return True

    def runManual(self):
        self.arduino.send(self.series + self.tuning + self.matching + self.switch)
        if self.vna.device is not None:
            s11, impedance = self.vna.getS11(self.larmorFreq)
            self.mapVals['s11'] = s11
            s11r = np.real(s11)
            s11i = np.imag(s11)
            if s11i >= 0:
                print("S11 = %0.3f + j %0.3f" % (s11r, s11i))
            else:
                print("S11 = %0.3f - j %0.3f" % (s11r, np.abs(s11i)))
            print("R = %0.2f Ohms" % np.real(impedance))
            print("X = %0.2f Ohms" % np.imag(impedance))
            self.arduino.send(self.series + self.tuning + self.matching + "1")
            return True
        else:
            return False

    def sweepSeries(self):
        nCap = 5
        s11_vec = []

        print("\n###########################")
        print("  Sweep series capacitor   ")
        print("###########################")

        # Combinations
        self.states = [''] * 2 ** nCap
        for state in range(2 ** nCap):
            prov = format(state, f'0{nCap}b')
            self.states[state] = ''.join('1' if bit == '0' else '0' for bit in prov)

        # Sweep values
        for state in range(2**(nCap-1)):
            self.arduino.send(self.states[state+1] + self.tuning + self.matching + "0")
            s11, impedance = self.vna.getS11(self.larmorFreq)
            self.s11.append(s11)
            s11r = np.real(s11)
            s11i = np.imag(s11)
            print("\nState %s" % self.states[state+1])
            if s11i >= 0:
                print("S11 = %0.3f + j %0.3f" % (s11r, s11i))
            else:
                print("S11 = %0.3f - j %0.3f" % (s11r, np.abs(s11i)))
            print("R = %0.2f Ohms" % np.real(impedance))
            print("X = %0.2f Ohms" % np.imag(impedance))

        self.mapVals['s11'] = np.array(self.s11)

        return True

    def sweepMatching(self):
        nCap = 5
        s11_vec = []

        print("\n###########################")
        print(" Sweep matching capacitor  ")
        print("###########################")

        # Combinations
        self.states = [''] * 2 ** nCap
        for state in range(2 ** nCap):
            prov = format(state, f'0{nCap}b')
            self.states[state] = ''.join('1' if bit == '0' else '0' for bit in prov)

        # Sweep values
        for state in range(2 ** (nCap - 1)):
            self.arduino.send(self.series + self.tuning + self.states[state + 1] + "0")
            s11, impedance = self.vna.getS11(self.larmorFreq)
            self.s11.append(s11)
            s11r = np.real(s11)
            s11i = np.imag(s11)
            print("\nState %s" % self.states[state + 1])
            if s11i >= 0:
                print("S11 = %0.3f + j %0.3f" % (s11r, s11i))
            else:
                print("S11 = %0.3f - j %0.3f" % (s11r, np.abs(s11i)))
            print("R = %0.2f Ohms" % np.real(impedance))
            print("X = %0.2f Ohms" % np.imag(impedance))

        self.mapVals['s11'] = np.array(self.s11)

        return True

    def sweepTuning(self):
        nCap = 5

        print("\n###########################")
        print("  Sweep tuning capacitor   ")
        print("###########################")

        # Combinations
        self.states = [''] * 2 ** nCap
        for state in range(2 ** nCap):
            prov = format(state, f'0{nCap}b')
            self.states[state] = ''.join('1' if bit == '0' else '0' for bit in prov)

        # Sweep values
        for state in range(2 ** nCap):
            self.arduino.send(self.series + self.states[state] + self.matching + "0")
            s11, impedance = self.vna.getS11(self.larmorFreq)
            self.s11.append(s11)
            s11r = np.real(s11)
            s11i = np.imag(s11)
            print("\nState %s" % self.states[state])
            if s11i >= 0:
                print("S11 = %0.3f + j %0.3f" % (s11r, s11i))
            else:
                print("S11 = %0.3f - j %0.3f" % (s11r, np.abs(s11i)))
            print("R = %0.2f Ohms" % np.real(impedance))
            print("X = %0.2f Ohms" % np.imag(impedance))

        self.mapVals['s11'] = np.array(self.s11)

        return True

    def getCsZ(self, n0, stateCt, stateCm):
        print("\nObtaining series capacitor...")
        n = [n0]

        # First measurement
        self.arduino.send(self.states[n[-1]] + self.states[stateCt] + self.states[stateCm] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        self.s11.append(s11)
        s11dB = 20 * np.log10(np.abs(s11))
        r = impedance.real
        x = impedance.imag
        print("")
        print(self.states[n[-1]])
        print("S11 = %0.2f dB" % s11dB)
        print("R = %0.2f Ohms" % r)
        print("X = %0.2f Ohms" % x)
        x0 = [np.imag(impedance)]

        # Sweep series impedance until reactance goes to 50 Ohms
        if x0[-1] < self.seriesTarget:
            step = 1
            while x0[-1] < self.seriesTarget and n[-1]+step < 16 and s11dB > -20:
                n.append(n[-1]+step)
                self.arduino.send(self.states[n[-1]] + self.states[stateCt] + self.states[stateCm] + "0")
                s11, impedance = self.vna.getS11(self.larmorFreq)
                self.s11.append(s11)
                s11dB = 20 * np.log10(np.abs(s11))
                r = impedance.real
                x = impedance.imag
                print("")
                print(self.states[n[-1]])
                print("S11 = %0.2f dB" % s11dB)
                print("R = %0.2f Ohms" % r)
                print("X = %0.2f Ohms" % x)
                x0.append(impedance.imag)
        elif x0[-1] > self.seriesTarget:
            step = -1
            while x0[-1] > self.seriesTarget and n[-1] + step > 0 and s11dB > -20:
                n.append(n[-1] + step)
                self.arduino.send(self.states[n[-1]] + self.states[stateCt] + self.states[stateCm] + "0")
                s11, impedance = self.vna.getS11(self.larmorFreq)
                self.s11.append(s11)
                s11dB = 20 * np.log10(np.abs(s11))
                r = impedance.real
                x = impedance.imag
                print("")
                print(self.states[n[-1]])
                print("S11 = %0.2f dB" % s11dB)
                print("R = %0.2f Ohms" % r)
                print("X = %0.2f Ohms" % x)
                x0.append(impedance.imag)

        # Select the value with reactance closest to 50 Ohms
        if s11dB <= -20:
            stateCs = n[-1]
        else:
            stateCs = n[np.argmin(np.abs(np.array(x0) - self.seriesTarget))]
        print("\nBest state:")
        print(self.states[stateCs])

        return stateCs

    def getCsZ2(self, n0, stateCt, stateCm):
        print("\nObtaining series capacitor...")
        n = [n0]

        # First measurement
        self.arduino.send(self.states[n[-1]] + self.states[stateCt] + self.states[stateCm] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        self.s11.append(s11)
        s11dB = 20 * np.log10(np.abs(s11))
        r = impedance.real
        x = impedance.imag
        print("")
        print(self.states[n[-1]])
        print("S11 = %0.2f dB" % s11dB)
        print("R = %0.2f Ohms" % r)
        print("X = %0.2f Ohms" % x)
        r0 = [r]

        # Sweep series impedance until reactance goes to 50 Ohms
        if r0[-1] < 50:
            step = 1
            while r0[-1] < 50 and n[-1]+step < 16 and s11dB > -20:
                n.append(n[-1]+step)
                self.arduino.send(self.states[n[-1]] + self.states[stateCt] + self.states[stateCm] + "0")
                s11, impedance = self.vna.getS11(self.larmorFreq)
                self.s11.append(s11)
                s11dB = 20 * np.log10(np.abs(s11))
                r = impedance.real
                x = impedance.imag
                print("")
                print(self.states[n[-1]])
                print("S11 = %0.2f dB" % s11dB)
                print("R = %0.2f Ohms" % r)
                print("X = %0.2f Ohms" % x)
                r0.append(r)
        elif r0[-1] > 50:
            step = -1
            while r0[-1] > 50 and n[-1] + step > 0 and s11dB > -20:
                n.append(n[-1] + step)
                self.arduino.send(self.states[n[-1]] + self.states[stateCt] + self.states[stateCm] + "0")
                s11, impedance = self.vna.getS11(self.larmorFreq)
                self.s11.append(s11)
                s11dB = 20 * np.log10(np.abs(s11))
                r = impedance.real
                x = impedance.imag
                print("")
                print(self.states[n[-1]])
                print("S11 = %0.2f dB" % s11dB)
                print("R = %0.2f Ohms" % r)
                print("X = %0.2f Ohms" % x)
                r0.append(r)

        # Select the value with reactance closest to 50 Ohms
        if s11dB <= -20:
            stateCs = n[-1]
        else:
            stateCs = n[np.argmin(np.abs(np.array(r0) - 50))]
        print("\nBest state:")
        print(self.states[stateCs])

        return stateCs

    def getCtZ(self, stateCs, n0, stateCm):
        # Sweep tuning capacitances until resistance goes higher than 50 Ohms
        print("\nObtaining tuning capacitor...")

        n = [n0]

        # First measurement
        self.arduino.send(self.states[stateCs] + self.states[n[-1]] + self.states[stateCm] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        self.s11.append(s11)
        s11dB = 20 * np.log10(np.abs(s11))
        r = impedance.real
        x = impedance.imag
        print("")
        print(self.states[n[-1]])
        print("S11 = %0.2f dB" % s11dB)
        print("R = %0.2f Ohms" % r)
        print("X = %0.2f Ohms" % x)
        r0 = [np.real(impedance)]

        if r0[-1] < 50:
            step = 1
            while r0[-1] < 50.0 and n[-1]+step < 32 and s11dB > -20:
                n.append(n[-1]+step)
                self.arduino.send(self.states[stateCs] + self.states[n[-1]] + self.states[stateCm] + "0")
                s11, impedance = self.vna.getS11(self.larmorFreq)
                self.s11.append(s11)
                s11dB = 20 * np.log10(np.abs(s11))
                r = impedance.real
                x = impedance.imag
                print("")
                print(self.states[n[-1]])
                print("S11 = %0.2f dB" % s11dB)
                print("R = %0.2f Ohms" % r)
                print("X = %0.2f Ohms" % x)
                r0.append(r)
        else:
            step = -1
            while r0[-1] > 50.0 and n[-1]+step >= 0 and s11dB > -20:
                n.append(n[-1] + step)
                self.arduino.send(self.states[stateCs] + self.states[n[-1]] + self.states[stateCm] + "0")
                s11, impedance = self.vna.getS11(self.larmorFreq)
                self.s11.append(s11)
                s11dB = 20 * np.log10(np.abs(s11))
                r = impedance.real
                x = impedance.imag
                print("")
                print(self.states[n[-1]])
                print("S11 = %0.2f dB" % s11dB)
                print("R = %0.2f Ohms" % r)
                print("X = %0.2f Ohms" % x)
                r0.append(r)

        # Select the value with reactance closest to 50 Ohms
        if s11dB <= -20:
            stateCt = n[-1]
        else:
            stateCt = n[np.argmin(np.abs(np.array(r0) - 50.0))]
        print("\nBest state:")
        print(self.states[stateCt])

        return stateCt

    def getCmZ(self, stateCs, stateCt, n0):
        print("\nObtaining matching capacitor...")

        n = [n0]

        # First measurement
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[n[-1]] + "0")
        s11, impedance = self.vna.getS11(self.larmorFreq)
        self.s11.append(s11)
        s11dB = 20 * np.log10(np.abs(s11))
        r = impedance.real
        x = impedance.imag
        print("")
        print(self.states[n[-1]])
        print("S11 = %0.2f dB" % s11dB)
        print("R = %0.2f Ohms" % r)
        print("X = %0.2f Ohms" % x)
        x0 = [x]

        # Sweep series impedance until reactance goes to 50 Ohms
        if x0[-1] < 0.0:
            step = 1
            while x0[-1] < 0.0 and n[-1] + step < 16 and s11dB > -20:
                n.append(n[-1] + step)
                self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[n[-1]] + "0")
                s11, impedance = self.vna.getS11(self.larmorFreq)
                self.s11.append(s11)
                s11dB = 20 * np.log10(np.abs(s11))
                r = impedance.real
                x = impedance.imag
                print("")
                print(self.states[n[-1]])
                print("S11 = %0.2f dB" % s11dB)
                print("R = %0.2f Ohms" % r)
                print("X = %0.2f Ohms" % x)
                x0.append(impedance.imag)
        elif x0[-1] > 0.0:
            step = -1
            while x0[-1] > 0.0 and n[-1] + step > 0 and s11dB > -20:
                n.append(n[-1] + step)
                self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[n[-1]] + "0")
                s11, impedance = self.vna.getS11(self.larmorFreq)
                self.s11.append(s11)
                s11dB = 20 * np.log10(np.abs(s11))
                r = impedance.real
                x = impedance.imag
                print("")
                print(self.states[n[-1]])
                print("S11 = %0.2f dB" % s11dB)
                print("R = %0.2f Ohms" % r)
                print("X = %0.2f Ohms" % x)
                x0.append(impedance.imag)

        # Select the value with reactance closest to 50 Ohms
        if s11dB <= -20:
            stateCm = n[-1]
        else:
            stateCm = n[np.argmin(np.abs(np.array(x0)))]
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + "0")
        print("\nBest state:")
        print(self.states[stateCm])

        return stateCm


if __name__ == '__main__':
    seq = AutoTuning()
    seq.sequenceAtributes()
    seq.sequenceRun()
    seq.sequenceAnalysis(mode='Standalone')
