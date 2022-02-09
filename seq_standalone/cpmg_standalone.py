# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 12:40:05 2021

@author: José Miguel Algarín Guisado
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


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def cpmg_standalone(
    init_gpa=False,               # Starts the gpa
    larmorFreq = 3.075e6,      # Larmor frequency
    rfExAmp = 0.3,             # rf excitation pulse amplitude
    rfReAmp = None,             # rf refocusing pulse amplitude
    rfExTime =22e-6,          # rf excitation pulse time
    rfReTime = None,          # rf refocusing pulse time
    echoSpacing = 10e-3,        # time between echoes
    repetitionTime = 2000e-3,     # TR
    nPoints = 500,                 # Number of points along readout, phase and slice
    etl = 100,                   # Echo train length
    acqTime = 2e-3,             # Acquisition time
    ):

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
#            'tx_gate': (rxGateTime, rxGateAmp)
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

    # Initialize time
    t0 = 20
    
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

    # Run the experiment and get data
    rxd, msgs = expt.run()
    rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
    data = rxd['rx0']
    data = np.reshape(data, (etl+1, nPoints))
    data = np.abs(data[1:etl+1, :])
    data = np.amax(data, axis=1)
    t = (np.arange(etl)*echoSpacing+echoSpacing)*1e-3
    
    # Fitting
    dataLog = np.log(data)
    fitting = np.polyfit(t, dataLog, 1)
    dataFitting = np.poly1d(fitting)
    dataFitLog = dataFitting(t)
    dataFit = np.exp(dataFitLog)
    T2 = -1/fitting[0]
    
    # Plot data
    plt.plot(t, data, 'o', t, dataFit, 'r')
    plt.ylabel('Echo amplitude (mV)')
    plt.xlabel('Echo time (ms)')
    plt.legend(['Experimental', 'Fitting'])
    plt.title('CPMG, T2 = '+str(round(T2, 1))+' ms')
    plt.show()

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


if __name__ == "__main__":

    cpmg_standalone()
