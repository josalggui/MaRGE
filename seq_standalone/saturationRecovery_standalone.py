"""
@author: J.M. algarín, february 25th 2022
MRILab, i3M, CSIC, Valencia
@ email: josalggui@i3m.upv.es
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

def saturationrecoveryStandalone(
    init_gpa= False,                 
    larmorFreq = 3.077,  # MHz 
    rfExAmp = 0.4,
    rfReAmp = 0.4, 
    rfExTime = 20, # us 
    rfReTime = 20, # us
    acqTime = 4,  # ms
    echoTime = 20, # ms
    repetitionTime = 5, # s
    tSatIni = 0.01, # s
    tSatFin = 0.9, # s
    nRepetitions = 1, # number of samples
    nRD=100, 
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
    rawData['seqName'] = 'inversionRecovery'
    rawData['larmorFreq'] = larmorFreq*1e6
    rawData['rfExAmp'] = rfExAmp
    rawData['rfReAmp'] = rfReAmp
    rawData['rfExTime'] = rfExTime
    rawData['rfRetime'] = rfReTime
    rawData['repetitionTime'] = repetitionTime
    rawData['tSatIni'] = tSatIni
    rawData['tSatFin'] = tSatFin
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

    tSat = np.geomspace(tSatIni, tSatFin, nRepetitions)
    rawData['tSat'] = tSat
    
    
    def createSequence():
        # Set shimming
        mri.iniSequence(expt, 20, shimming)
        
        for repeIndex in range(nRepetitions):
            # Initialize time
            tEx = 20e3+np.max(tSat)+repetitionTime*repeIndex
            
            # Inversion time for current iteration
            inversionTime = tSat[repeIndex]
            
            # Crusher gradient for inversion rf pulse
#            t0 = tEx-inversionTime-crusherTime/2-gradRiseTime-hw.gradDelay-50
#            mri.gradTrap(expt, t0, gradRiseTime, crusherTime, 0.005, gSteps, axes[0], shimming)
#            mri.gradTrap(expt, t0, gradRiseTime, crusherTime, 0.005, gSteps, axes[1], shimming)
#            mri.gradTrap(expt, t0, gradRiseTime, crusherTime, 0.005, gSteps, axes[2], shimming)

            # Saturation pulse
            t0 = tEx-inversionTime-hw.blkTime-rfReTime/2
            mri.rfRecPulse(expt, t0, rfReTime, rfReAmp, 0)
            
            # Spoiler gradients to destroy residual transversal signal detected for ultrashort inversion times
#            mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[0], shimming)
#            mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[1], shimming)
#            mri.gradTrap(expt, t0+hw.blkTime+rfReTime, gradRiseTime, inversionTime*0.5, 0.005, gSteps, axes[2], shimming)
            
            # Excitation pulse
            t0 = tEx-hw.blkTime-rfExTime/2
            mri.rfRecPulse(expt, t0, rfExTime, rfExAmp, 0)
            
#            # Rx gating
#            t0 = tEx+rfExTime/2+hw.deadTime
#            mri.rxGate(expt, t0, acqTime)
            
            # Crusher gradient
            t0 = tEx+echoTime/2-crusherTime/2-gradRiseTime-hw.gradDelay-50
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
    tSat *= 1e6
    scanTime = nRepetitions*repetitionTime # us
    
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0] # us
    bw = 1/samplingPeriod/hw.oversamplingFactor # MHz
    acqTime = nRD/bw # us
    rawData['samplingPeriod'] = samplingPeriod*1e-6
    rawData['bw'] = bw*1e6
    createSequence()
    
    # Representar Secuencia o tomar los datos.
    tSat1 = np.geomspace(tSatIni, tSatFin, nRepetitions)
    tSat2 = np.geomspace(tSatIni, tSatFin, 10*nRepetitions)
    if plotSeq==1:                
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        print('Running...')
        rxd, msgs = expt.run()
        print('End')
        data = rxd['rx0']*13.788
        expt.__del__()
        data = sig.decimate(data, hw.oversamplingFactor, ftype='fir', zero_phase=True)
        rawData['fullData'] = data
        dataIndiv = np.reshape(data,  (nRepetitions, nRD))
        dataIndiv = np.real(dataIndiv[:, int(nRD/2)]*np.exp(-1j*(np.angle(dataIndiv[0, int(nRD/2)])+np.pi)))
        results = np.transpose(np.array([tSat1, dataIndiv/np.max(dataIndiv)]))
#        results = np.transpose(np.array([tSat1, dataIndiv]))
        rawData['signalVsTime'] = results
        
        plt.figure(1)
        plt.plot(np.abs(data))
        plt.show()
        
        # For 1 component
        fitData1, xxx = curve_fit(func1, results[:, 0],  results[:, 1])
        print('For one component:')
        print('mA', round(fitData1[0], 1))
        print('T1', round(fitData1[1]*1e3), ' ms')
        rawData['T11'] = fitData1[1]*1e3
        rawData['M1'] = fitData1[0]
        
        # For 2 components
        fitData2, xxx = curve_fit(func2, results[:, 0],  results[:, 1], p0=(1, 0.1, 0.5, 0.05), bounds=(0., 5.))
        print('For two components:')
        print('Ma', round(fitData2[0], 1))
        print('Mb', round(fitData2[2], 1))
        print('T1a', round(fitData2[1]*1e3), ' ms')
        print('T1b', round(fitData2[3]*1e3), ' ms')
        rawData['T12'] = [fitData2[1], fitData2[3]]
        rawData['M2'] = [fitData2[0], fitData2[2]]
        
        # For 3 components
        fitData3, xxx = curve_fit(func3, results[:, 0],  results[:, 1], p0=(1, 0.1, 0.5, 0.05, 1, 0.01), bounds=(0., 5.))
        print('For three components:')
        print('Ma', round(fitData3[0], 1), ' ms')
        print('Mb', round(fitData3[2], 1), ' ms')
        print('Mc', round(fitData3[4], 1), ' ms')
        print('T1a', round(fitData3[1]*1e3), ' ms')
        print('T1b', round(fitData3[3]*1e3), ' ms')
        print('T1c', round(fitData3[5]*1e3), ' ms')
        rawData['T13'] = [fitData3[1], fitData3[3], fitData3[5]]
        rawData['M3'] = [fitData3[0], fitData3[2], fitData3[4]]
        
        # Save data
        mri.saveRawData(rawData)
        
        # Plots
        plt.figure(2, figsize=(5, 5))
        plt.plot(results[:, 0], results[:, 1], 'o')
        plt.plot(tSat2, func1(tSat2, *fitData1))
        plt.plot(tSat2, func2(tSat2, *fitData2))
        plt.plot(tSat2, func3(tSat2, *fitData3))
        plt.title(rawData['fileName'])
        plt.xscale('log')
        plt.xlabel('t(s)')
        plt.ylabel('Signal (mV)')
        plt.legend(['Experimental', 'Fitting 1 component', 'Fitting 2 components','Fitting 3 components' ])
        plt.title(rawData['fileName'])
        plt.show()
        

def func1(x, m, t1):
    return m*(1-np.exp(-x/t1))

def func2(x, ma, t1a, mb, t1b):
    return ma*(1-np.exp(-x/t1a))+mb*(1-np.exp(-x/t1b))

def func3(x, ma, t1a, mb, t1b, mc, t1c):
    return ma*(1-np.exp(-x/t1a))+mb*(1-np.exp(-x/t1b))+mc*(1-np.exp(-x/t1c))

if __name__ == "__main__":
    saturationrecoveryStandalone()

