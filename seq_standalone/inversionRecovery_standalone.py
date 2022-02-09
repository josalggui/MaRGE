# -*- coding: utf-8 -*-
"""
Created on Thu Oct  14 20:41:05 2021

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


def inversionRecovery_standalone(
    init_gpa=False,               # Starts the gpa
    larmorFreq = 3.075e6,      # Larmor frequency
    rfExAmp = 0.3,             # rf excitation pulse amplitude
    rfReAmp = None,             # rf refocusing pulse amplitude
    rfExTime =30e-6,          # rf excitation pulse time
    rfReTime = None,          # rf refocusing pulse time
    echoTime = 10e-3,        # time between echoes
    repetitionTime = 4,     # TR
    nPoints = 500,                 # Number of points along readout, phase and slice
    nRepetitions = 40,                   # Echo train length
    minIRTime = 10e-3,           # Minimum inversion recovery time
    maxIRTime = 3000e-3,          # Maximum inversion recovery time
    acqTime = 2e-3,             # Acquisition time
    ): 

    # Miscellaneous
    blkTime = 15             # Deblanking time (us)
    larmorFreq = larmorFreq*1e-6
    if rfReAmp==None:
        rfReAmp = rfExAmp
    if rfReTime==None:
        rfReTime = 2*rfExTime
    deadTime = 250              # Dead time (us)
    irTimeVector = np.geomspace(minIRTime,maxIRTime,num=nRepetitions)*1e6
    
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
            'grad_vx': (np.array([tStart]), np.array([gAmp])),
            'grad_vy': (np.array([tStart]), np.array([gAmp])),
            'grad_vz': (np.array([tStart]), np.array([gAmp]))
            })

    # Changing time parameters to us
    rfExTime = rfExTime*1e6
    rfReTime = rfReTime*1e6
    echoTime = echoTime*1e6
    repetitionTime = repetitionTime*1e6
    
    # Create sequence
    t0 = 0
    for nRepetition in range(nRepetitions):
        # Pi pulse
        t0 += 20
        rfPulse(t0, rfReTime, rfReAmp, 0)
        
        # Pi/2 pulse
        t0 = t0+rfReTime/2+irTimeVector[nRepetition]-rfExTime/2
        rfPulse(t0, rfExTime, rfExAmp, 0)
        
        # FID readout
        t0 = t0+blkTime+rfExTime+deadTime
        rxGate(t0, acqTime)
        
        # Pi pulse
        t0 = t0-deadTime-rfExTime/2+echoTime/2-rfReTime/2-blkTime
        rfPulse(t0, rfReTime, rfReAmp, np.pi/2)
        
        # Echo readout
        t0 = t0+blkTime+rfReTime/2+echoTime/2-acqTime/2
        rxGate(t0, acqTime)
        
        # Finalize repetition
        t0 = repetitionTime*(nRepetition+1)
        finalizeExperiment(t0, 0)

    # Run the experiment and get data
    rxd, msgs = expt.run()
    rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
    data = rxd['rx0']
    data = np.reshape(data, (nRepetitions, 2*nPoints))
    data = np.abs(data[:, int(3*2*nPoints/4)])
    dataMinIndex = np.argmin(data)
    T1 = irTimeVector[dataMinIndex]*1e-3/np.log(2)
    # t = (np.arange(etl)*echoSpacing+echoSpacing)*1e-3
    
    # Plot data
    plt.plot(irTimeVector*1e-6, np.abs(data))
    plt.ylabel('Echo amplitude (mV)')
    plt.xlabel('Inversion time (s)')
    plt.title('IR, T1 = '+str(round(T1, 1))+' ms')
    plt.xscale('log')
    plt.show()

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


if __name__ == "__main__":

    inversionRecovery_standalone()
