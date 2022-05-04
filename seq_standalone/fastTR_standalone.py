"""
@author: Teresa Guallart Naval
@modifield: J.M. algarín, february 25th 2022
"""
import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import configs.hw_config as hw # Import the scanner hardware config
import mrilabMethods.mrilabMethods as mri
import scipy.signal as sig
import time

def fasttrStandalone(
    init_gpa= False,
    larmorFreq = 3.07968,  # MHz 
    rfExAmp = 0.3,
    rfReAmp = 0.6, 
    rfExTime = 30, # us 
    rfReTime = 60, # us
    acqTime = 4,  # ms
    echoTime = 20, # ms
    repetitionTime =0.8, # s
    nRepetitions = 2, # number of samples
    nRD=50, 
    plotSeq =0,
    shimming=[-70, -90, 10]):                
    
    shimming = np.array(shimming)*1e-4
    
    if rfReTime is None:
        rfReTime = 2*rfExTime
    
    rfExTime = rfExTime*1e-6
    rfReTime = rfReTime*1e-6
    acqTime = acqTime*1e-3
    echoTime = echoTime*1e-3
    
    rawData = {}
    rawData['seqName'] = 'fastTR'
    rawData['larmorFreq'] = larmorFreq*1e6
    rawData['rfExAmp'] = rfExAmp
    rawData['rfReAmp'] = rfReAmp
    rawData['rfExTime'] = rfExTime
    rawData['rfRetime'] = rfReTime
    rawData['repetitionTime'] = repetitionTime
    rawData['nRepetitions'] = nRepetitions
    rawData['acqTime'] = acqTime
    rawData['nRD'] = nRD
    rawData['echoTime'] = echoTime
    
    # Miscellaneous
    gradRiseTime = 200 # us
    crusherTime = 1000 # us
    gSteps = int(gradRiseTime/5)
    axes = np.array([0, 1, 2])
    rawData['gradRiseTime'] = gradRiseTime
    rawData['gSteps'] = gSteps
    
    # Bandwidth and sampling rate
    bw = nRD/acqTime*1e-6 # MHz
    bwov = bw*hw.oversamplingFactor
    samplingPeriod = 1/bwov

    def createSequence():
        # Set shimming
        mri.iniSequence(expt, 20, shimming)
        
        for repeIndex in range(nRepetitions):
            # Initialize time
            tEx = 20e3+repetitionTime*repeIndex
            
            # Excitation pulse
            t0 = tEx-hw.blkTime-rfExTime/2
            mri.rfRecPulse(expt, t0, rfExTime, rfExAmp, 0)
            
#            # Rx gating
#            t0 = tEx+rfExTime/2+hw.deadTime
#            mri.rxGate(expt, t0, acqTime)
            
            # Crusher gradient
            t0 = tEx+echoTime/2-crusherTime/2-gradRiseTime-hw.gradDelay-43
            mri.gradTrap(expt, t0, gradRiseTime, crusherTime, 0.005, gSteps, axes[0], shimming)
            mri.gradTrap(expt, t0, gradRiseTime, crusherTime, 0.005, gSteps, axes[1], shimming)
            mri.gradTrap(expt, t0, gradRiseTime, crusherTime, 0.005, gSteps, axes[2], shimming)
            
            # Refocusing pulse
            t0 = tEx+echoTime/2-rfReTime/2-hw.blkTime
            mri.rfRecPulse(expt, t0, rfReTime, rfReAmp, np.pi/2)
            
            # Rx gating
            t0 = tEx+echoTime-acqTime/2
            mri.rxGate(expt, t0, acqTime)
        
        # End sequence
        mri.endSequence(expt, scanTime)
    
    # Time variables in us
    rfExTime *= 1e6
    rfReTime *= 1e6
    repetitionTime *= 1e6
    echoTime *= 1e6
    scanTime = nRepetitions*repetitionTime # us
    
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0] # us
    bw = 1/samplingPeriod/hw.oversamplingFactor # MHz
    acqTime = nRD/bw # us
    rawData['samplingPeriod'] = samplingPeriod*1e-6
    rawData['bw'] = bw*1e6
    createSequence()
    
    # Representar Secuencia o tomar los datos.
    data = []
    nn = 1
    if plotSeq==1:                
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        print('Running...')
        for repe in range(nn):
            rxd, msgs = expt.run()
            data = np.concatenate((data, rxd['rx0']*13.788), axis=0)
            if nn>1: time.sleep(20)
        expt.__del__()
        print('Ready!')
        data = sig.decimate(data, hw.oversamplingFactor, ftype='fir', zero_phase=True)
        data = np.reshape(data, (nn, nRD*2))
        data = np.average(data, axis=0)
        rawData['fullData'] = data
        dataIndiv = np.reshape(np.abs(data),  (nRepetitions, nRD))
        print('End')
        
        # Save data
        mri.saveRawData(rawData)
        
        # Plots
        plt.figure(1)
        plt.plot(dataIndiv[0, :])
        plt.plot(dataIndiv[1, :])
        plt.xlabel('Time (a.u.)')
        plt.ylabel('Signal (mV)')
        plt.legend('First echo', 'Second echo')
        plt.title(rawData['fileName'])
        
        ratio = dataIndiv[0, int(nRD/2)]/dataIndiv[1, int(nRD/2)]
        print('Signal ratio = ', ratio)
        
        plt.show()

if __name__ == "__main__":
    fasttrStandalone()

