# -*- coding: utf-8 -*-
"""
Created on Sat Nov  13 10:46:05 2021

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
import scipy.signal as sig
import os
from scipy.io import savemat
from datetime import date,  datetime 
import pdb
from configs.hw_config import Gx_factor
from configs.hw_config import Gy_factor
from configs.hw_config import Gz_factor
st = pdb.set_trace



#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def rare_standalone(
    init_gpa=False,              # Starts the gpa
    larmorFreq = 3.0785e6,      # Larmor frequency
    rfExAmp = 0.03,             # rf excitation pulse amplitude
    rfExTime = 33e-6,          # rf excitation pulse time
    nFIDs = 2,                        # Number of FIDs to be acquired
    repetitionTime = 5e-3,     # TR
    BW = 30e3,                  # Bandwidth
    ):
    
    # rawData fields
    rawData = {}
    inputs = {}
    outputs = {}
    auxiliar = {}
    
    # Miscellaneous
    blkTime = 10             # Deblanking time (us)
    larmorFreq = larmorFreq*1e-6
    deadTime = 200
    oversamplingFactor = 6
    auxiliar['oversamplingFactor'] = oversamplingFactor
    
    # Inputs for rawData
    inputs['larmorFreq'] = larmorFreq      # Larmor frequency
    inputs['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    inputs['rfExTime'] = rfExTime          # rf excitation pulse time
    inputs['repetitionTime'] = repetitionTime     # TR
    inputs['bandwidth'] = BW
    inputs['nFIDs'] = nFIDs
    
    # BW
    acqTime = repetitionTime*nFIDs
    nPoints = BW*acqTime
    BW = BW*1e-6
    BWov = BW*oversamplingFactor
    samplingPeriod = 1/BWov
    
    # Initialize the experiment
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BW = 1/samplingPeriod/oversamplingFactor
    acqTime = nPoints/BW        # us
    auxiliar['bandwidth'] = BW*1e6
    
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

    
    # Changing time parameters to us
    rfExTime = rfExTime*1e6
    repetitionTime = repetitionTime*1e6
    
    # Rx gate
    t0 = 20
    rxGate(t0, acqTime)
    
    for iFID in range(nFIDs):
        t0 = 20+iFID*repetitionTime
        
        # Excitation pulse
        rfPulse(t0,rfExTime,rfExAmp,0)
            
    # Plot sequence:
    expt.plot_sequence()
    
    # Run the experiment
    rxd, msgs = expt.run()
    rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
    expt.__del__()
    # Get data
    scanData = sig.decimate(rxd['rx0'], oversamplingFactor, ftype='fir', zero_phase=True)
    data = np.reshape(scanData, (nFIDs, int(nPoints/nFIDs)))
    outputs['sampled'] = data

    # Plot signals
    plt.figure(2)
    tVector = np.linspace(0., acqTime/nFIDs, num=int(nPoints/nFIDs),endpoint=False)*1e-3
    for ii in range(nFIDs):
        plt.subplot(1, 2, 1)
        plt.plot(tVector[0:100], np.abs(data[ii, 10:110]))
        plt.xlabel('t (ms)')
        plt.ylabel('Signal magnitude (mV)')
        plt.legend(['First ex', 'Econd ex'])
        
        plt.subplot(1, 2, 2)
        plt.plot(tVector[0:100], np.unwrap(np.angle(data[ii, 10:110])))    
        plt.xlabel('t (ms)')
        plt.ylabel('Signal phase (rad)')
        plt.legend(['First ex', 'Econd ex'])
            
    # Save data
    dt = datetime.now()
    dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
    dt2 = date.today()
    dt2_string = dt2.strftime("%Y.%m.%d")
    if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
        os.makedirs('experiments/acquisitions/%s' % (dt2_string))
            
    if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
        os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
    inputs['name'] = "%s.%s.mat" % ("TSE",dt_string)
    rawData['inputs'] = inputs
    rawData['auxiliar'] = auxiliar
    rawData['outputs'] = outputs
    rawdata = {}
    rawdata['rawData'] = rawData
    savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, "TSE",dt_string),  rawdata)
    
    plt.show()


if __name__ == "__main__":

    rare_standalone()
