"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: code to obtain a good combination of tuning/matching
Specific hardware from MRILab @ i3M is required
"""

import os
import sys
#*****************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************
from PyQt5.QtCore import QThreadPool
import experiment as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
from worker import Worker
# import pyfirmata
import time
import configs.hw_config as hw

from plotview.spectrumplot import SpectrumPlot
import pyqtgraph as pg


class AutoTuning(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(AutoTuning, self).__init__()
        # Input the parameters
        self.pyfirmata = None
        self.expt = None
        self.repeat = None
        self.threadpool = None
        self.rfExAmp = None
        self.rfExTime = None
        self.txChannel = None
        self.freqOffset = None
        self.addParameter(key='seqName', string='AutoTuningInfo', val='AutoTuning')
        self.addParameter(key='freqOffset', string='RF frequency offset (kHz)', val=0.0, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (s)', val=10, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.1, field='RF')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        # Output parameters
        self.voltage = None

    def sequenceInfo(self):
        print("\n RF Auto-tuning")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Look for the best combination of tuning/matching.")
        print("Specific hardware from MRILab @ i3M is required. \n")

    def sequenceTime(self):
        return 0  # minutes, scanTime

    def sequenceRun(self, plotSeq=0):
        # Create the inputs automatically as class properties
        for key in self.mapKeys:
            setattr(self, key, self.mapVals[key])

        # Fix units to MHz and us
        hw.larmorFreq = 3.066 # MHz
        self.freqOffset *= 1e-3  # MHz
        self.rfExTime *= 1e6 # us

        # SEQUENCE
        self.expt = ex.Experiment(lo_freq=hw.larmorFreq + self.freqOffset, init_gpa=False)
        t0 = 5
        self.iniSequence(t0, np.array([0, 0, 0]))
        t0 = 10
        self.ttl(t0, self.rfExTime + hw.blkTime, channel=1)
        self.rfRawPulse(t0 + hw.blkTime, self.rfExTime, self.rfExAmp, txChannel=1)
        t0 += hw.blkTime + self.rfExTime + 10
        self.endSequence(t0)

        # Run sequence continuously
        if not plotSeq:
            self.repeat = True
            # Sweep the tuning matching states in parallel thread
            self.threadpool = QThreadPool()
            print("Multithreading with maximum %d threads \n" % self.threadpool.maxThreadCount())
            worker = Worker(self.run())  # Any other args, kwargs are passed to the run function
            self.threadpool.start(worker)
            # Excite
            while self.repeat:
                self.expt.run()
                # pass
            print("Ready!")
        self.expt.__del__()

    def sequenceAnalysis(self, obj=''):
        self.saveRawData()
        x = np.linspace(0, 2**10, 2**10)
        y = np.reshape(self.voltage, -1)

        # Plot signal versus time
        voltageWidget = SpectrumPlot(xData=x,
                                     yData=[y],
                                     legend=['first iteration'],
                                     xLabel='State',
                                     yLabel='Voltage (mV)',
                                     title='Votage VS state')

        self.out = [voltageWidget]

        if obj == 'Standalone':
            voltageWidget.show()
            pg.exec()

        return (self.out)

    # def runTest2(self):
    #
    def runTest(self):
        time.sleep(2)
        print("Soy A")
        time.sleep(2)
        self.repeat = False

    def run(self):
        arduino = self.pyfirmata.Arduino('/dev/ttyACM0')
        print('\n Arduino connected!')

        it = self.pyfirmata.util.Iterator(arduino)
        it.start()

        # Input analog port (to measure power)
        vIn = arduino.analog[5]
        vIn.enable_reporting()

        # Ouput ports (to control capacitance)
        # Tuning
        ctA = arduino.digital[13]
        ctB = arduino.digital[11]
        ctC = arduino.digital[9]
        ctD = arduino.digital[7]
        ctE = arduino.digital[5]
        # Matching
        cmA = arduino.digital[12]
        cmB = arduino.digital[10]
        cmC = arduino.digital[8]
        cmD = arduino.digital[6]
        cmE = arduino.digital[4]
        # Serie
        csA = arduino.digital[3]
        csB = arduino.digital[2]
        csC = arduino.digital[15]
        # csD = arduino.digital[46]
        # csE = arduino.digital[45]
        capacitors = [ctA, ctB, ctC, ctD, ctE, cmA, cmB, cmC, cmD, cmE]
        # capacitorsS = [csA, csB, csC, csD, csE]
        capacitorsS = [csA, csB]

        # Creates all possible states
        nCapacitors = 10
        cState = np.arange(0, 2)
        nStates = 2 ** nCapacitors
        ct2, ct1, ct3, ct4, ct5, cm1, cm2, cm3, cm4, cm5 = np.meshgrid(cState, cState, cState, cState, cState, cState,
                                                                       cState, cState, cState, cState)
        ct1 = np.reshape(ct1, (nStates, 1))
        ct2 = np.reshape(ct2, (nStates, 1))
        ct3 = np.reshape(ct3, (nStates, 1))
        ct4 = np.reshape(ct4, (nStates, 1))
        ct5 = np.reshape(ct5, (nStates, 1))
        cm1 = np.reshape(cm1, (nStates, 1))
        cm2 = np.reshape(cm2, (nStates, 1))
        cm3 = np.reshape(cm3, (nStates, 1))
        cm4 = np.reshape(cm4, (nStates, 1))
        cm5 = np.reshape(cm5, (nStates, 1))
        states01 = np.concatenate((ct1, ct2, ct3, ct4, ct5, cm1, cm2, cm3, cm4, cm5), axis=1)
        states = np.full((np.size(states01, 0), np.size(states01, 1)), True)
        for ii in range(np.size(states01, 0)):
            for jj in range(np.size(states01, 1)):
                if states01[ii, jj]:
                    states[ii, jj] = True
                else:
                    states[ii, jj] = False


        # Sweep states
        t0 = time.time()
        self.voltage = np.ones((nStates, 1))
        state = 0
        lastVoltage = 1
        csA.write(1)
        time.sleep(0.001)
        # while state < nStates:
        while state < 50:
            ii = 0
            for capacitor in capacitors:
                capacitor.write(states[state, ii])
                ii += 1
                time.sleep(0.001)
            time.sleep(0.05)
            self.voltage[state] = vIn.read()
            lastVoltage = self.voltage[state]
            state += 1

        vMin = np.min(self.voltage)
        idx = np.argmin(self.voltage)
        stateMin = states[idx, :]

        print("Best state = ", stateMin)
        print('Elapsed time = ', time.time() - t0)

        # Set the state to the best one
        ii = 0
        for capacitor in capacitors:
            capacitor.write(stateMin[ii])
            ii += 1
            time.sleep(0.001)
        self.repeat = False
        arduino.exit()


if __name__ == '__main__':
    seq = AutoTuning()
    seq.sequenceRun()
    seq.sequenceAnalysis(obj='Standalone')
