"""
@author: José Miguel Algarín Guisado
@modifield: T. Guallart Naval, february 28th 2022
MRILAB @ I3M
"""

import sys
# marcos_client path for linux
sys.path.append('../marcos_client')
# marcos_client and PhysioMRI_GUI for Windows
sys.path.append('D:\CSIC\REPOSITORIOS\marcos_client')
sys.path.append('D:\CSIC\REPOSITORIOS\PhysioMRI_GUI')
import numpy as np
import experiment as ex
import matplotlib.pyplot as plt
import pdb
st = pdb.set_trace
from scipy.optimize import curve_fit
from datetime import date,  datetime 
import os
from scipy.io import savemat


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def cpmgStandalone(
    init_gpa=False,               # Starts the gpa
    larmorFreq = 3.0605,      # Larmor frequency
    rfExAmp = 0.3,             # rf excitation pulse amplitude
    rfReAmp = None,             # rf refocusing pulse amplitude
    rfExTime =35,          # rf excitation pulse time
    rfReTime = None,          # rf refocusing pulse time
    repetitionTime = 10000, 
    echoSpacing = 10,        # time between echoes
    nPoints = 200,                 # Number of points along readout, phase and slice
    etl = 1000,                   # Echo train length
    acqTime =4,             # Acquisition time
    shimming = [-20, -30, 20], 
    inversionTime = 0
    ):

    plotSeq = 0

    larmorFreq = larmorFreq*1e6
    rfExTime = rfExTime*1e-6
    echoSpacing = echoSpacing*1e-3
    acqTime = acqTime*1e-3
    inversionTime= inversionTime*1e-3
    repetitionTime = repetitionTime*1e-3
    shimming = np.array(shimming)*1e-4
    
    # Miscellaneous
    blkTime = 100             # Deblanking time (us)
    larmorFreq = larmorFreq*1e-6
    rfReAmp = rfExAmp
    rfReTime = 2*rfExTime
    
    # BW
    BW = nPoints/acqTime*1e-6
    samplingPeriod = 1/BW

    # Initialize the experiment
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BW = 1/samplingPeriod
    acqTime = nPoints/BW        # us
    
    rawData = {}
    rawData['larmorFreq'] = larmorFreq
    rawData['rfExAmp'] = rfExAmp
    rawData['rfReAmp'] = rfReAmp
    rawData['rfReTime'] = rfExAmp
    rawData['rfExtime'] = rfExTime
    rawData['echoSpacing'] = echoSpacing
    rawData['nPoints'] = nPoints
    rawData['etl'] = etl
    rawData['acqTime'] = acqTime
    rawData['TR'] = repetitionTime
    rawData['shimming'] = shimming
    rawData['BW'] = BW
    
    # Create functions
    def endSequence(tEnd):
        expt.add_flodict({
                'grad_vx': (np.array([tEnd]),np.array([0]) ), 
                'grad_vy': (np.array([tEnd]),np.array([0]) ), 
                'grad_vz': (np.array([tEnd]),np.array([0]) ),
             })
             
    def iniSequence(tEnd, shimming):
        expt.add_flodict({
                'grad_vx': (np.array([tEnd]),np.array([shimming[0]]) ), 
                'grad_vy': (np.array([tEnd]),np.array([shimming[1]]) ), 
                'grad_vz': (np.array([tEnd]),np.array([shimming[2]]) ),
             })
             
    # Create an rf pulse function
    def rfPulse(tStart,rfTime,rfAmplitude,rfPhase):
        txTime = np.array([tStart+blkTime,tStart+blkTime+rfTime])
        txAmp = np.array([rfAmplitude*np.exp(1j*rfPhase),0.])
        txGateTime = np.array([tStart,tStart+blkTime+rfTime])
        txGateAmp = np.array([1,0])
        expt.add_flodict({
            'tx0': (txTime, txAmp),
            'tx_gate': (txGateTime, txGateAmp)
            })

    # Readout function
    def rxGate(tStart,gateTime):
        rxGateTime = np.array([tStart,tStart+gateTime])
        rxGateAmp = np.array([1,0])
        expt.add_flodict({
            'rx0_en':(rxGateTime, rxGateAmp), 
            'rx_gate': (rxGateTime, rxGateAmp), 
            })

    # Gradients
    def gradSetup(tStart, gAmp):
        expt.add_flodict({
            'grad_vx': (np.array(tStart), np.array(gAmp)),
            'grad_vy': (np.array(tStart), np.array(gAmp)),
            'grad_vz': (np.array(tStart), np.array(gAmp))
            })
                
    def finalizeExperiment(tStart, gAmp):
        expt.add_flodict({
            'grad_vx': (np.array(tStart), np.array(gAmp)),
            'grad_vy': (np.array(tStart), np.array(gAmp)),
            'grad_vz': (np.array(tStart), np.array(gAmp))
            })

    # Changing time parameters to us
    rfExTime = rfExTime*1e6
    rfReTime = rfReTime*1e6
    echoSpacing = echoSpacing*1e6
    repetitionTime = repetitionTime*1e6
    inversionTime = inversionTime*1e6

    # Initialize time
    tIni= 20
    
    # Shimming
    iniSequence(tIni, shimming)
    t0 = 2*tIni
    if inversionTime!=0:
        rfPulse(t0,rfReTime,rfReAmp,0)
    t0 += rfReTime/2+inversionTime+rfExTime/2
    # Excitation pulse
    rfPulse(t0,rfExTime,rfExAmp,0)
    
    # First readout to avoid RP initial readout effect
    rxGate(t0+blkTime+rfExTime+200, acqTime)
    t0 += (rfExTime+echoSpacing-rfReTime)/2

    # Echo train
    for echoIndex in range(etl):
        # Refocusing pulse
        rfPulse(t0,rfReTime,rfReAmp,np.pi/2)

        # Rx gate
        rxGate(t0+blkTime+rfReTime/2+echoSpacing/2-acqTime/2,acqTime)
        
        # Update t0
        t0 = t0+echoSpacing
    
    tEnd = repetitionTime+2*tIni
    endSequence(tEnd)
    if plotSeq==1:                
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        # Run the experiment and get data
        rxd, msgs = expt.run()
        rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
        data = rxd['rx0']
        data = np.reshape(data, (etl+1, nPoints))
        rawData['dataFull'] = data
        data = np.abs(data[1:etl+1, :])
        data = np.amax(data, axis=1)
        t = (np.arange(etl)*echoSpacing+echoSpacing)*1e-3
        results = np.transpose(np.array([t, data]))
        rawData['signalVsTime'] = results
        
        
    # For 1 component
        fitData1, xxx = curve_fit(func1, results[:, 0],  results[:, 1])
        print('For one component:')
        print('mA', round(fitData1[0], 1))
        print('T2', round(fitData1[1]), ' ms')
        rawData['T21'] = fitData1[1]
        rawData['M1'] = fitData1[0]
        
        # For 2 components
#        fitData2, xxx = curve_fit(func2, results[:, 0],  results[:, 1])
#        print('For two components:')
#        print('Ma', round(fitData2[0], 1))
#        print('Mb', round(fitData2[2], 1))
#        print('T2a', round(fitData2[1]), ' ms')
#        print('T2b', round(fitData2[3]), ' ms')
#        rawData['T22'] = [fitData2[1], fitData2[3]]
#        rawData['M2'] = [fitData2[0], fitData2[2]]
        
#        # For 3 components
#        fitData3, xxx = curve_fit(func3, results[:, 0],  results[:, 1])
#        print('For three components:')
#        print('Ma', round(fitData3[0], 1), ' ms')
#        print('Mb', round(fitData3[2], 1), ' ms')
#        print('Mc', round(fitData3[4], 1), ' ms')
#        print('T2a', round(fitData3[1]), ' ms')
#        print('T2b', round(fitData3[3]), ' ms')
#        print('T2c', round(fitData3[5]), ' ms')
#        rawData['T23'] = [fitData3[1], fitData3[3], fitData3[5]]
#        rawData['M3'] = [fitData3[0], fitData3[2], fitData3[4]]
#        
        # Save data
        name = saveMyData(rawData)
        
        # Plots
        plt.figure(2, figsize=(5, 5))
        plt.plot(results[:, 0], results[:, 1], 'o')
        plt.plot(t, func1(t, *fitData1))
#        plt.plot(t, func2(t, *fitData2))
#        plt.plot(t, func3(t, *fitData3))
        plt.title(name)
        plt.xlabel('t(ms)')
        plt.ylabel('Signal (mV)')
#        plt.legend(['Experimental', 'Fitting 1 component', 'Fitting 2 components','Fitting 3 components' ])
        plt.title(name)
        plt.show()
    

def func1(x, m, t2):
    return m*np.exp(-x/t2)

def func2(x, ma, t2a, mb, t2b):
    return ma*np.exp(-x/t2a)+mb*np.exp(-x/t2b)

def func3(x, ma, t2a, mb, t2b, mc, t2c):
    return ma*np.exp(-x/t2a)+mb*np.exp(-x/t2b)+mc*np.exp(-x/t2c)

def saveMyData(rawData):
    # Save data
    dt = datetime.now()
    dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
    dt2 = date.today()
    dt2_string = dt2.strftime("%Y.%m.%d")
    if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
        os.makedirs('experiments/acquisitions/%s' % (dt2_string))
    if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
        os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
    rawData['name'] = "%s.%s.mat" % ("CPMG",dt_string)
    savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, "CPMG",dt_string),  rawData)
    return rawData['name']

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


if __name__ == "__main__":

    cpmgStandalone()
