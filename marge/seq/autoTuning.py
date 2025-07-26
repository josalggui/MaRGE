"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: code to obtain a good combination of tuning/matching
Specific hardware from MRILab @ i3M is required
"""
import copy
import os
import sys
import time

from scipy.interpolate import interp1d

import numpy as np
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import marge.configs.hw_config as hw
import marge.autotuning.autotuning as autotuning
import marge.configs.units as units


class AutoTuning(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(AutoTuning, self).__init__()
        # Input the parameters
        self.freqOffset = None
        self.frequency = None
        self.statesXm = None
        self.statesCm = None
        self.statesCt = None
        self.statesXs = None
        self.states = None
        self.statesCs = None
        self.test = None
        self.matching = None
        self.tuning = None
        self.series = None
        self.seriesTarget = 80
        self.tuningTarget = 55
        self.state0 = None
        self.frequencies = None
        self.expt = None
        self.threadpool = None
        self.s11_hist = []
        self.s11_db_hist = []
        self.states_hist = [[], [], []]
        self.n_aux = [[], [], []]

        # Parameters
        self.addParameter(key='seqName', string='AutoTuningInfo', val='AutoTuning')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='freqOffset', string='Frequency offset(kHz)', val=0.0, units=units.kHz, field='IM')
        self.addParameter(key='series', string='Series capacitor', val='11011', field='IM')
        self.addParameter(key='tuning', string='Tuning capacitor', val='10000', field='IM')
        self.addParameter(key='matching', string='Matching capacitor', val='10011', field='IM')
        self.addParameter(key='test', string='Test', val='auto', field='IM',
                          tip='Choose one option: auto, manual')
        self.addParameter(key='xyz', string='xyz', val=0.0, field='IM')

        # Connect to Arduino and set the initial state
        self.arduino = autotuning.Arduino(name="auto-tuning", serial_number=hw.ard_sn_autotuning)
        self.arduino.connect()
        self.arduino.send(self.mapVals['series'] + self.mapVals['tuning'] + self.mapVals['matching'] + "11")

    def sequenceInfo(self):
        print("RF automatic impedance matching")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Look for the best combination of tuning/matching.")
        print("Specific hardware from MRILab @ i3M is required.\n")

    def sequenceTime(self):
        return 0  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        self.demo = demo
        self.s11_hist = []
        self.s11_db_hist = []
        self.states_hist = [[], [], []]
        self.n_aux = [[], [], []]
        self.frequency = hw.larmorFreq + self.freqOffset * 1e-6

        if self.arduino.device is None:
            print("WARNING: No Arduino found for auto-tuning.")
            return False
        else:
            counter = 0
            while counter < 10:
                # Turn OFF vna.
                self.arduino.send(self.mapVals['series'] + self.mapVals['tuning'] + self.mapVals['matching'] + "11")
                print("nanoVNA OFF")
                time.sleep(0.5)

                # Turn ON vna.
                self.arduino.send(self.mapVals['series'] + self.mapVals['tuning'] + self.mapVals['matching'] + "00")
                print("nanoVNA ON")
                time.sleep(0.5)

                # Connect to VNA
                print("Linking to nanoVNA...")
                self.vna = autotuning.VNA()
                if self.vna.connect():
                    break
                else:
                    counter += 1
                    print("No nanoVNA found for auto-tuning....\n")

            # Check connection with nanoVNA
            if self.vna.device is None:
                print("No nanoVNA found for auto-tuning. \n")
                return False

        if self.test == 'auto':
            return self.runAutoTuning()
        elif self.test == 'manual':
            return self.runManual()
        else:
            print("Incorrect test mode.")
            return False

    def sequenceAnalysis(self, mode=None):
        self.mode = mode

        # Get results
        s11 = np.array(self.s11_hist)
        s11_opt = self.mapVals['s11']
        f_vec = self.vna.getFrequency()
        s_vec = self.vna.getData()

        # Interpolate s_vec
        interp_func = interp1d(f_vec, s_vec, kind='cubic')
        f_vec_t = np.linspace(np.min(f_vec), np.max(f_vec), 1000)
        s_vec_t = interp_func(f_vec_t)

        # Insert s11 into s_vec
        index = np.searchsorted(f_vec_t, self.frequency)
        f_vec_t = np.insert(f_vec_t, index, self.frequency)
        s_vec_t = np.insert(s_vec_t, index, s11_opt)

        # Get s in dB
        s_vec_db = 20 * np.log10(np.abs(s_vec_t))

        # Get quality factor
        try:
            idx = np.argmin(s_vec_db)
            f0 = f_vec_t[idx]
            f1 = f_vec_t[np.argmin(np.abs(s_vec_db[0:idx] + 3))]
            f2 = f_vec_t[idx + np.argmin(np.abs(s_vec_db[idx::] + 3))]
            q = f0 / (f2 - f1)
            print("Q = %0.0f" % q)
            print("BW @ -3 dB = %0.0f kHz" % ((f2 - f1) * 1e3))
        except:
            pass

        # Create data array in case single point is acquired
        if self.test == 'manual':
            s11 = np.concatenate((s11, s11), axis=0)

        # Plot smith chart
        result1 = {'widget': 'smith',
                   'xData': [np.real(s11), np.real(s_vec_t)],
                   'yData': [np.imag(s11), np.imag(s_vec_t)],
                   'xLabel': 'Real(S11)',
                   'yLabel': 'Imag(S11)',
                   'title': 'Smith chart',
                   'legend': ['', ''],
                   'row': 0,
                   'col': 0}

        # Plot reflection coefficient
        result2 = {'widget': 'curve',
                   'xData': (f_vec_t - self.frequency) * 1e3,
                   'yData': [s_vec_db],
                   'xLabel': 'Frequency (kHz)',
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
                self.statesXs[state] = -1 / (2 * np.pi * self.frequency * 1e6 * self.statesCs[state])

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
                self.statesXm[state] = -1 / (2 * np.pi * self.frequency * 1e6 * self.statesCm[state])

        # Get initial state index
        stateCs = self.states.index(self.series)
        stateCt = self.states.index(self.tuning)
        stateCm = self.states.index(self.matching)

        # Check current status
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + "00")
        s11, impedance = self.vna.getS11(self.frequency)
        self.addValues(s11, self.series, self.tuning, self.matching, stateCs, stateCt, stateCm)
        s11_db = 20 * np.log10(np.abs(s11))

        # Search from input impedance in cas initial state is far from matching
        if s11_db > -20:
            stateCs = self.getCsZ(16, 0, 16)
            stateCt = self.getCtZ(stateCs, 0, 16)
            stateCm = self.getCmZ(stateCs, stateCt, 16)

            # Get the best state from all measured states
            idx = np.argmin(self.s11_db_hist)
            stateCs = self.n_aux[0][idx]
            stateCt = self.n_aux[1][idx]
            stateCm = self.n_aux[2][idx]
            s11_db = self.s11_db_hist[idx]

            # Move one state the series capacitor in case there is not matching.
            if s11_db > -20:
                stateCs = stateCs + 1
                stateCt = self.getCtZ(stateCs, 0, 16)
                stateCm = self.getCmZ(stateCs, stateCt, 16)

                # Get the best state from all measured states
                idx = np.argmin(self.s11_db_hist)
                stateCs = self.n_aux[0][idx]
                stateCt = self.n_aux[1][idx]
                stateCm = self.n_aux[2][idx]
                s11_db = self.s11_db_hist[idx]

        # Once s11 < -20 dB, do a final optimization based on s11 in dB
        self.finalOptimization2D(stateCs, stateCt, stateCm)

        # Get the best state from all measured states
        idx = np.argmin(self.s11_db_hist)
        stateCs = self.n_aux[0][idx]
        stateCt = self.n_aux[1][idx]
        stateCm = self.n_aux[2][idx]
        s11_db = self.s11_db_hist[idx]

        # Check final status
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + "00")
        s11, impedance = self.vna.getS11(self.frequency)
        self.addValues(s11, self.series, self.tuning, self.matching, stateCs, stateCt, stateCm)

        # Print results
        print("S11 = %0.1f dB" % s11_db)

        # Save parameters to source sequence
        try:
            sequence = self.sequence_list[self.mapVals['seqName']]
            sequence.mapVals['matching'] = self.states[stateCm]
            sequence.mapVals['tuning'] = self.states[stateCt]
            sequence.mapVals['series'] = self.states[stateCs]
        except:
            pass

        # Save result into the mapVals to save the rawData
        self.mapVals['series'] = self.states[stateCs]
        self.mapVals['tuning'] = self.states[stateCt]
        self.mapVals['matching'] = self.states[stateCm]
        self.mapVals['s11'] = self.s11_hist[-1]
        self.mapVals['s11_db'] = self.s11_db_hist[-1]

        # Connect the system to TxRx switch
        self.arduino.send(self.states[stateCs] + self.states[stateCt] + self.states[stateCm] + "11")
        print("nanoVNA OFF")

        # Data to sweep sequence
        self.mapVals['sampledPoint'] = s11_db

        return True

    def runManual(self):
        self.arduino.send(self.series + self.tuning + self.matching + "00")
        if self.vna.device is not None:
            s11, impedance = self.vna.getS11(self.frequency)
            self.s11_hist.append(s11)
            self.mapVals['s11'] = s11
            s11dB = 20 * np.log10(np.abs(s11))
            r = impedance.real
            x = impedance.imag
            print("S11 = %0.2f dB" % s11dB)
            print("R = %0.2f Ohms" % r)
            print("X = %0.2f Ohms" % x)
            self.arduino.send(self.series + self.tuning + self.matching + "11")
            print("nanoVNA OFF")
            return True
        else:
            return False

    def addValues(self, s11, cs, ct, cm, ns, nt, nm):
        self.s11_hist.append(s11)
        self.s11_db_hist.append(20 * np.log10(np.abs(s11)))
        self.states_hist[0].append(cs)
        self.states_hist[1].append(ct)
        self.states_hist[2].append(cm)
        self.n_aux[0].append(ns)
        self.n_aux[1].append(nt)
        self.n_aux[2].append(nm)

    def getCsZ(self, n0, stateCt, stateCm):
        print("Series sweep...")
        n = [n0]

        # First measurement
        cs = self.states[n[-1]]
        ct = self.states[stateCt]
        cm = self.states[stateCm]
        self.arduino.send(cs + ct + cm + "00")
        s11, impedance = self.vna.getS11(self.frequency)
        self.addValues(s11, cs, ct, cm, n0, stateCt, stateCm)
        r = np.real(impedance)
        x = np.imag(impedance)
        x0 = [x]
        z = [impedance]

        # if x0[-1] < self.seriesTarget:
        if np.abs(z[-1]) < self.seriesTarget:
            step = 1
        else:
            step = -1

        # Sweep series impedance until reactance goes to 50 Ohms
        # while step * x0[-1] < step * self.seriesTarget and 0 < n[-1] + step < 16 and self.s11_db_hist[-1] > -20:
        while step * np.abs(z[-1]) < step * self.seriesTarget and 0 < n[-1] + step < 16 and self.s11_db_hist[-1] > -20:
            n.append(n[-1] + step)
            cs = self.states[n[-1]]
            self.arduino.send(cs + ct + cm + "00")
            s11, impedance = self.vna.getS11(self.frequency)
            self.addValues(s11, cs, ct, cm, n[-1], stateCt, stateCm)
            r = np.real(impedance)
            x = np.imag(impedance)
            x0.append(x)
            z.append(impedance)

        # Select the value with reactance closest to 50 Ohms
        if self.s11_db_hist[-1] <= -20:
            stateCs = n[-1]
        else:
            try:
                z = np.array(z)
                stateCs = n[np.argmin(np.abs(z) - self.seriesTarget)]
                if np.imag(z[np.argmin(np.abs(z) - self.seriesTarget)]) < 0:
                    stateCs += 1
            except:
                stateCs = n0

        return stateCs

    def getCtZ(self, stateCs, n0, stateCm):
        # Sweep tuning capacitances until resistance goes higher than 50 Ohms
        print("Tuning sweep...")
        n = [n0]

        # First measurement
        cs = self.states[stateCs]
        ct = self.states[n[-1]]
        cm = self.states[stateCm]
        self.arduino.send(cs + ct + cm + "00")
        s11, impedance = self.vna.getS11(self.frequency)
        self.addValues(s11, cs, ct, cm, stateCs, n0, stateCm)
        r = np.real(impedance)
        x = np.imag(impedance)
        r0 = [r]

        if r0[-1] < self.tuningTarget:
            step = 1
        else:
            step = -1

        while step * r0[-1] < step * self.tuningTarget and 0 <= n[-1] + step < 32 and self.s11_db_hist[-1] > -20:
            n.append(n[-1] + step)
            ct = self.states[n[-1]]
            self.arduino.send(cs + ct + cm + "00")
            s11, impedance = self.vna.getS11(self.frequency)
            self.addValues(s11, cs, ct, cm, stateCs, n[-1], stateCm)
            r = np.real(impedance)
            x = np.imag(impedance)
            r0.append(r)

        # Select the value with reactance closest to 50 Ohms
        if self.s11_db_hist[-1] <= -20:
            stateCt = n[-1]
        else:
            try:
                stateCt = n[np.argmin(np.abs(np.array(r0) - self.seriesTarget))]
            except:
                stateCt = n0

        return stateCt

    def getCmZ(self, stateCs, stateCt, n0):
        print("Matching sweep...")
        n = [n0]

        # First measurement
        cs = self.states[stateCs]
        ct = self.states[stateCt]
        cm = self.states[n[-1]]
        self.arduino.send(cs + ct + cm + "00")
        s11, impedance = self.vna.getS11(self.frequency)
        self.addValues(s11, cs, ct, cm, stateCs, stateCt, n0)
        r = np.real(impedance)
        x = np.imag(impedance)
        x0 = [x]

        if x0[-1] < 0.0:
            step = 1
        else:
            step = -1

            # Sweep series impedance until reactance goes to 50 Ohms
            while step * x0[-1] < 0.0 and 0 < n[-1] + step < 16 and self.s11_db_hist[-1] > -20:
                n.append(n[-1] + step)
                cm = self.states[n[-1]]
                self.arduino.send(cs + ct + cm + "00")
                s11, impedance = self.vna.getS11(self.frequency)
                self.addValues(s11, cs, ct, cm, stateCs, stateCt, n[-1])
                r = np.real(impedance)
                x = np.imag(impedance)
                x0.append(impedance.imag)

        # Select the value with reactance closest to 50 Ohms
        if self.s11_db_hist[-1] <= -20:
            stateCm = n[-1]
        else:
            try:
                stateCm = n[np.argmin(np.abs(np.array(x0)))]
            except:
                stateCm = n0

        return stateCm

    def finalOptimization2D(self, stateCs, stateCt, stateCm):
        print("Optimizing...")
        cs = stateCs
        ct_old = copy.copy(stateCt)
        ct_new = copy.copy(stateCt)
        cm_old = copy.copy(stateCm)
        cm_new = copy.copy(stateCm)
        check = True
        iteration = 0
        result = [[], [], [], []]
        while check and iteration < 5:
            ctv = [ct_old - 1, ct_old, ct_old + 1]
            cmv = [cm_old - 1, cm_old, cm_old + 1]
            for ct in ctv:
                for cm in cmv:
                    # Check if current state is inside the boundaries
                    if cm == 0 or cm == 17 or ct == -1 or ct == 32:
                        continue
                    else:
                        state = self.states[ct] + self.states[cm]
                    # Get s11 if current state has not been tested before
                    if state not in result[3]:
                        cs_bin = self.states[cs]
                        ct_bin = self.states[ct]
                        cm_bin = self.states[cm]
                        self.arduino.send(self.states[cs] + self.states[ct] + self.states[cm] + "00")
                        s11, impedance = self.vna.getS11(self.frequency)
                        self.addValues(s11, cs_bin, ct_bin, cm_bin, cs, ct, cm)
                        result[0].append(self.s11_db_hist[-1])
                        result[1].append(ct)
                        result[2].append(cm)
                        result[3].append(state)
            best_state = np.argmin(np.array(result[0]))
            ct_new = result[1][best_state]
            cm_new = result[2][best_state]
            if ct_new == ct_old and cm_new == cm_old:
                check = False
            else:
                ct_old = copy.copy(ct_new)
                cm_old = copy.copy(cm_new)
            iteration += 1

        return ct_new, cm_new

    def finalOptimization3D(self, stateCs, stateCt, stateCm):
        print("Optimizing...")
        cs_old = copy.copy(stateCs)
        cs_new = copy.copy(stateCs)
        ct_old = copy.copy(stateCt)
        ct_new = copy.copy(stateCt)
        cm_old = copy.copy(stateCm)
        cm_new = copy.copy(stateCm)
        result = [[], [], [], [], []]
        check = True
        iteration = 0
        while check and iteration < 10:
            csv = [cs_old - 1, cs_old, cs_old + 1]
            ctv = [ct_old - 1, ct_old, ct_old + 1]
            cmv = [cm_old - 1, cm_old, cm_old + 1]
            for cs in csv:
                for ct in ctv:
                    for cm in cmv:
                        cs_bin = self.states[cs]
                        ct_bin = self.states[ct]
                        cm_bin = self.states[cm]
                        state = cs_bin + ct_bin + cm_bin
                        if state in result[4]:
                            pass
                        else:
                            result[4].append(state)
                            if cm == 0 or cm == 17 or ct == -1 or ct == 32 or cs == 0 or cs == 17:
                                result[0].append(0.0)
                            else:
                                self.arduino.send(state + "00")
                                s11, impedance = self.vna.getS11(self.frequency)
                                self.addValues(s11, cs_bin, ct_bin, cm_bin, cs, ct, cm)
                                result[0].append(self.s11_db_hist[-1])
                            result[1].append(cs)
                            result[2].append(ct)
                            result[3].append(cm)
            best_state = np.argmin(np.array(result[0]))
            cs_new = result[1][best_state]
            ct_new = result[2][best_state]
            cm_new = result[3][best_state]
            if ct_new == ct_old and cm_new == cm_old and cs_new == cs_old:
                check = False
            else:
                cs_old = copy.copy(cs_new)
                ct_old = copy.copy(ct_new)
                cm_old = copy.copy(cm_new)
            iteration += 1

        return cs_new, ct_new, cm_new


if __name__ == '__main__':
    seq = AutoTuning()
    seq.sequenceAtributes()
    seq.sequenceRun()
    seq.sequenceAnalysis(mode='Standalone')
