"""
@author: J.M. Algarín, May 2nd 2022
"""
import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import configs.hw_config as hw # Import the scanner hardware config
import mrilabMethods.mrilabMethods as mri
import time
import serial

def fasttrStandalone(
    init_gpa= False,
    larmorFreq = 3.07,  # MHz 
    # rfExAmp = 0.2, # RF amplitude in a.u.
    rfExAmp = 0.2, 
    rfExTime = 10000, # us 
    repetitionTime = 0.1, # s
    nRepetitions = 1): # number of samples
    
    # Inputs in fundamental units
    rfExTime = rfExTime*1e-6
    
    rawData = {}
    rawData['seqName'] = 'autoTuning'
    rawData['larmorFreq'] = larmorFreq*1e6
    rawData['rfExAmp'] = rfExAmp
    rawData['rfExTime'] = rfExTime
    rawData['repetitionTime'] = repetitionTime
    rawData['nRepetitions'] = nRepetitions
    
    # Miscellaneous
    shimming = np.array([0, 0, 0])
    plt.ion()
    
    # Bandwidth and sampling rate
    bw = 0.06 # MHz
    bwov = bw*hw.oversamplingFactor
    samplingPeriod = 1/bwov

    #*******************************************************
    def createSequence():
        # Set shimming
        mri.iniSequence(expt, 20, shimming)
        t0 = 20
        mri.rfRecPulse(expt, t0, rfExTime, rfExAmp, 0)
            
        # End sequence
        mri.endSequence(expt, rfExTime+40)
    #*******************************************************
    
    #*******************************************************
    def ardToVolt(x):
        # Convert arduino inputs (in ascii format) to voltage (0 to 5 V)
        y = 0
        for ii in range(len(x)-2):
            y = y+(x[-3-ii]-48)*10**ii
        return np.double(y/1023*5)
    #*******************************************************
    
    # Time variables in us
    rfExTime *= 1e6
    
    # Create all possible states
    states = np.array([0, 1])
    nCapacitors = 10
    nStates = 2**nCapacitors
    c2, c1, c3, c4, c5, c6, c7, c8, c9, c10 = np.meshgrid(states, states, states, states, states, states, states, states, states, states)
    c1 = np.reshape(c1, (nStates, 1))
    c2 = np.reshape(c2, (nStates, 1))
    c3 = np.reshape(c3, (nStates, 1))
    c4 = np.reshape(c4, (nStates, 1))
    c5 = np.reshape(c5, (nStates, 1))
    c6 = np.reshape(c6, (nStates, 1))
    c7 = np.reshape(c7, (nStates, 1))
    c8 = np.reshape(c8, (nStates, 1))
    c9 = np.reshape(c9, (nStates, 1))
    c10 = np.reshape(c10, (nStates, 1))
    states = np.concatenate((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10), axis=1)
    
    # Convert states to strings
    strStates = []
    for ii in range(nStates):
        currentState = ""
        for jj in range(nCapacitors):
            currentState = currentState+str(states[ii, jj])
        strStates.append(currentState)
    
    # Create experiment
    arduino = serial.Serial(port = '/dev/ttyACM1', baudrate = 9600,  timeout=1.5)
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0] # us
    bw = 1/samplingPeriod/hw.oversamplingFactor # MHz
    createSequence()
    voltage = []
    
    # Initialize the experiment with a few runs
    arduino.write(bytes(strStates[0], 'utf-8'))
    expt.run()
    data = arduino.readline()
    arduino.write(bytes(strStates[0], 'utf-8'))
    expt.run()
    data = arduino.readline()
    
    # Initialize figure
    plt.figure(1)
    plt.plot(0, 0, 'b.')
    plt.ylabel('Voltage (V)')
    plt.xlabel('Iteration')
    plt.xlim([0, nStates])
    plt.show(block=False)
    plt.pause(0.01)
    
    # Sweep all states
    count = 0
    t0 = time.time()
    for state in strStates:
        arduino.write(bytes(state, 'utf-8'))    # Send current state through serial port to arduino.
        time.sleep(0.02)
        expt.run()                                              # Run experiment
        data = arduino.readline()                       # After experiment is finished, read data stored in serial port
        voltage.append(ardToVolt(data))                    # Convert data from (0-1023) to (0-5 V)
        print('Progress: %.2f %%, Voltage = %.3f V' %(count/nStates*100, voltage[-1])) 
        # Plots
        plt.plot(count, voltage[-1], 'b.')
        plt.show(block=False)
        plt.pause(0.01)
        # Delay to next repetition (Current version of RFPA needs 10% duty cycle)
        time.sleep(repetitionTime)
        count += 1
    t1 = time.time()
    expt.__del__()
    print('Ready!')
    rawData['voltage'] = voltage
    rawData['states'] = strStates
    print('Elapsed time: %.0f s'  %(t1-t0))
    
    
    
    # Get best state
    idx = np.argmin(voltage)
    arduino.write(bytes(strStates[idx], 'utf-8'))
    print('Best state: ', strStates[idx])
    rawData['optState'] = strStates[idx]
    plt.plot(idx, voltage[idx], 'r.')
    
    
    # Save data
    mri.saveRawData(rawData)
    plt.title(rawData['fileName'])
        
    # Plots
    plt.show(block=True)

if __name__ == "__main__":
    fasttrStandalone()

