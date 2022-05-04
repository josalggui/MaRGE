"""
Created on Thu Jan 20 12:20:45 2022

@author: Teresa
"""
import sys
# marcos_client path for linux
sys.path.append('../marcos_client')
import numpy as np
import pdb
from configs import hw_config as conf
st = pdb.set_trace


##### RF PULSE  #tRef es el tiempo medio del pulso
def rfPulse(tRef, rfAmp, rfDuration, txTimePrevious,txAmpPrevious,  txGateTimePrevious, txGateAmpPrevious):
    txTime = np.array([tRef-rfDuration/2,tRef+rfDuration/2])
    txAmp = np.array([rfAmp,0])
    txGateTime = np.array([txTime[0]-conf.blkTime,txTime[1]])
    txGateAmp = np.array([1,0])
    txTime = np.concatenate((txTimePrevious,txTime),  axis=0)
    txAmp = np.concatenate((txAmpPrevious,txAmp ),  axis=0)
    txGateTime = np.concatenate((txGateTimePrevious,txGateTime),  axis=0)
    txGateAmp = np.concatenate((txGateAmpPrevious,txGateAmp),  axis=0)
    return txTime,  txAmp,  txGateTime,  txGateAmp
    
   
##### RX PULSE  #tRef es el tiempo medio del pulso
def readoutGate(tRef,tRd,rxTimePrevious,  rxAmpPrevious):
    rxTime = np.array([tRef-tRd/2, tRef+tRd/2])
    rxAmp = np.array([1,0])
    rxTime=np.concatenate((rxTimePrevious, rxTime),  axis=0)
    rxAmp=np.concatenate((rxAmpPrevious, rxAmp),  axis=0)
    return rxTime,  rxAmp

##### GRADIENT PULSE 
# tIni: initial pulse time; gAmplitude: pulse amplitude in RP units; 
# gDuration: effective gradient time (gDuration = fltTopTime+ riseTime); 
# shim: shimming value; gTimePrevious and gAmpPrevious: previous pulses. 
def gradPulse(tIni, gAmplitude, gDuration, shim, gTimePrevious, gAmpPrevious): 
    # Trapezoid characteristics
    gRise = gAmplitude*conf.slewRate
    nSteps = gAmplitude*conf.stepsRate
    
    # Creating trapezoid
    tRise = np.linspace(tIni, tIni+gRise, nSteps, endpoint=True)
    aRise = np.linspace(0, gAmplitude, nSteps+1, endpoint=True)
    aRise = np.delete(aRise,0)
    tDown = np.linspace(tIni+gDuration,tIni+gDuration+gRise,nSteps,endpoint=True)
    aDown = np.linspace(gAmplitude,0,nSteps+1,endpoint=True)
    aDown = np.delete(aDown,0)
    gTime = np.concatenate((tRise,tDown),  axis=0)
    gAmp = np.concatenate((aRise,aDown),  axis=0)
    
    # Concatenating all pulses
    gTime=np.concatenate((gTimePrevious, gTime),  axis=0)
    gAmp=np.concatenate((gAmpPrevious, gAmp),  axis=0)
    return gTime, gAmp

