"""
@author: Teresa Guallart Naval
MRILAB @ I3M
"""

import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt

def RPInstructionsFalse(
    init_gpa= False,                 
    larmorFreq=3.07419, 
    rfExAmp=0.3, 
    rfReAmp=None, 
    rfExPhase = 0,
    rfExTime=38, 
    rfReTime=None,
    nReadout = 500,
    tAdq =5*1e3,
    gAxis =0,
    gNsteps =4, 
    gRiseTime = 100,
    gDuration = 500, #Flattop gDuration-2*gRiseTime
    gAmplitude = 1,
    tIni = 20, 
    plotSeq =1):
    
    txGatePre = 15
    txGatePost = 1
    
    if rfReAmp is None:
        rfReAmp = rfExAmp
    if rfReTime is None:
        rfReTime = 2*rfExTime
    
    rfExPhase = rfExPhase*np.pi/180
    rfExAmp = rfExAmp*np.exp(1j*rfExPhase)
    rfRePhase = np.pi/2
    rfReAmp = rfReAmp *np.exp(1j*rfRePhase)
    
    #DEFINICIÓN DE LOS PULSOS
    def endSequence(tEnd):
        gTime = [tEnd]
        gAmp = [0]
        return gTime,  gAmp


    def gradPulse(tIni):
        tRise = np.linspace(tIni, tIni+gRiseTime, gNsteps, endpoint=True)
        aRise = np.linspace(0, gAmplitude, gNsteps+1, endpoint=True)
        aRise = np.delete(aRise,0)
        tDown = np.linspace(tIni+gDuration-gRiseTime,tIni+gDuration,gNsteps,endpoint=True)
        aDown = np.linspace(gAmplitude,0,gNsteps+1,endpoint=True)
        aDown = np.delete(aDown,0)
        gTime = np.concatenate((tRise,tDown),  axis=0)
        gAmp = np.concatenate((aRise,aDown),  axis=0)
        tEnd = tIni+gDuration
        return gTime,  gAmp,  tEnd
    
    def rfPulse(tIni, rfAmp, rfDuration):
        txTime = np.array([tIni,tIni+rfDuration])
        txAmp = np.array([rfAmp,0])
        
        txGateTime = np.array([tIni-txGatePre,tIni+rfDuration+txGatePost])
        txGateAmp = np.array([1,0])
        tRef = tIni+rfDuration/2
        return txTime,  txAmp, txGateTime,  txGateAmp,  tRef
    
    
    #Ini Experiment
    BW = nReadout/tAdq
    rx_period = 1/BW
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    
    #Gradient Pulse
    if gNsteps == 1:
        gRiseTime = 0
   
#FIRST run()
    tSequence = tIni
    txTime,  txAmp, txGateTime,  txGateAmp,  tRef = rfPulse(tIni, rfExAmp, rfExTime)
    tSequence = tRef+200
    gTime,  gAmp,  tEnd = gradPulse(tSequence)
    tSequence = tEnd+500
    gTimeAux, gAmpAux,  tEnd = gradPulse(tSequence)
    gTime = np.concatenate((gTime,gTimeAux),  axis=0)
    gAmp = np.concatenate((gAmp,gAmpAux),  axis=0)
    tSequence = tEnd+500
    expt.add_flodict({'grad_vz': (gTime,gAmp)})
    expt.add_flodict({
                        'tx0': (txTime, txAmp),
                        'tx_gate': (txGateTime, txGateAmp)})
    
    if plotSeq == 0:
        rxd, msgs = expt.run()
    elif plotSeq==1:                
        expt.plot_sequence()
        plt.show()

#SECOND run()
    tSequence = tIni #Para tiempos de inicio
    txTime,  txAmp, txGateTime,  txGateAmp,  tRef = rfPulse(tSequence, rfReAmp, rfReTime)
    tSequence = tRef+200
    gTime,  gAmp,  tEnd = gradPulse(tSequence)
    tSequence = tEnd+500
    expt.add_flodict({'grad_vx': (gTime,gAmp)}, False)
    expt.add_flodict({
                        'tx0': (txTime, txAmp),
                        'tx_gate': (txGateTime, txGateAmp)}, False)
    
    txTime,  txAmp, txGateTime,  txGateAmp,  tRef = rfPulse(tSequence, rfExAmp, rfExTime)
    tSequence = tRef+200
    gTime, gAmp,  tEnd = gradPulse(tSequence)
    tSequence = tEnd+500
    gTimeAux,  gAmpAux = endSequence(tSequence)
    gTime = np.concatenate((gTime,gTimeAux),  axis=0)
    gAmp = np.concatenate((gAmp,gAmpAux),  axis=0)
    expt.add_flodict({'grad_vx': (gTime,gAmp)})
    expt.add_flodict({
                        'tx0': (txTime, txAmp),
                        'tx_gate': (txGateTime, txGateAmp)})
    
    if plotSeq == 0:
        rxd, msgs = expt.run()
    elif plotSeq==1:                
        expt.plot_sequence()
        plt.show()
    
    
# THIRD run()
    tSequence = tIni 
    txTime,  txAmp, txGateTime,  txGateAmp,  tRef = rfPulse(tSequence, rfReAmp, rfReTime)
    tSequence = tRef+200
    gTime,  gAmp,  tEnd = gradPulse(tSequence)
    tSequence = tEnd+500
    txTimeAux,  txAmpAux, txGateTimeAux,  txGateAmpAux,  tRef = rfPulse(tSequence, rfExAmp, rfExTime)
    tSequence = tRef+200
    gTimeAux, gAmpAux,  tEnd = gradPulse(tSequence)
    tSequence = tEnd+500
    gTime = np.concatenate((gTime,gTimeAux),  axis=0)
    gAmp = np.concatenate((gAmp,gAmpAux),  axis=0)
    txTime = np.concatenate((txTime,txTimeAux),  axis=0)
    txAmp = np.concatenate((txAmp,txAmpAux),  axis=0)
    txGateTime = np.concatenate((txGateTime,txGateTimeAux),  axis=0)
    txGateAmp = np.concatenate((txGateAmp,txGateAmpAux),  axis=0)
    expt.add_flodict({'grad_vx': (gTime,gAmp)}, False)
    expt.add_flodict({
                        'tx0': (txTime, txAmp),
                        'tx_gate': (txGateTime, txGateAmp)}, False)

    if plotSeq == 0:
        rxd, msgs = expt.run()
    elif plotSeq==1:                
        expt.plot_sequence()
        plt.show()

# FOURTH run()
    expt.add_flodict({'grad_vx': (gTime,gAmp*2)}, False)
    
    if plotSeq == 0:
        rxd, msgs = expt.run()
    elif plotSeq==1:                
        expt.plot_sequence()
        plt.show()

# FIFTH run()
    tSequence = tIni+10 #Para tiempos de inicio
    txTime = np.array([tSequence])
    txAmp = np.array([0])
    txGateTime = np.array([tSequence])
    txGateAmp = np.array([0])
    tSequence = tSequence + 100
    gTime,  gAmp,  tEnd = gradPulse(tSequence)
    tSequence = tEnd+500
    expt.add_flodict({'grad_vx': (gTime,gAmp)}, False)
    expt.add_flodict({
                        'tx0': (txTime, txAmp),
                        'tx_gate': (txGateTime, txGateAmp)}, False)
    
    if plotSeq == 0:
        rxd, msgs = expt.run()
    elif plotSeq==1:                
        expt.plot_sequence()
        plt.show()
#    
    #Las concatenaciones mejor meterlas en las definiciones de los pulsos
    #RUN
    if plotSeq==1:                
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        expt.__del__()
        
if __name__ == "__main__":
    RPInstructionsFalse()
