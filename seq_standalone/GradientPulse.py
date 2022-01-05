"""
Created on Tue Nov  9 10:37:29 2021

@author: Teresa
"""

import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import time 

def GradientPulse(
    init_gpa= False,                 
    larmorFreq=3.077, 
    nReadout = 500,
    tAdq =5*1e3,
    tPlus = 500, 
    gAxis =0,
    gNsteps = 8, 
    gRiseTime = 40,
    gDuration = 16000, #Flattop gDuration-2*gRiseTime
    gAmplitude =  0.04,
#    gNsteps = 160, #50
#    gRiseTime =800,
#    gDuration = 1700, #Flattop gDuration-2*gRiseTime
#    gAmplitude =  0.2,
    tIni = 20, 
    plotSeq =0, 
    nPulses =1):
    
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
    
    
    #Ini Experiment
    BW = nReadout/tAdq
    rx_period = 1/BW
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    
    #Gradient Pulse
    if gNsteps == 1:
        gRiseTime = 0

    tSequence = tIni
    gTime,  gAmp,  tEnd = gradPulse(tSequence)
    tSequence = tEnd
    tSequence = tSequence + tPlus
    gTime=np.concatenate((gTime, [tSequence]))
    gAmp=np.concatenate((gAmp, [0]))
    expt.add_flodict({'grad_vx': (gTime,gAmp)})
    
    #RUN
    if plotSeq==1:                
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        for i in range(nPulses):
            print(i)
            print('Running...')
            rxd, msgs = expt.run()
            time.sleep(0.1)
            print('End')
        expt.__del__()
    
if __name__ == "__main__":
    GradientPulse()
