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


def fovCal_standalone(
    init_gpa=False,              # Starts the gpa
    nScans = 1,                 # NEX
    larmorFreq = 3.072e6,      # Larmor frequency
    rfExAmp = 0.3,             # rf excitation pulse amplitude
    rfReAmp = 0.3,             # rf refocusing pulse amplitude
    rfExTime = 35e-6,          # rf excitation pulse time
    rfReTime = 70e-6,            # rf refocusing pulse time
    echoSpacing = 20e-3,        # time between echoes
    inversionTime = 0,       # Inversion recovery time
    repetitionTime = 150e-3,     # TR
    fov = np.array([20e-2, 20e-2, 20e-2]),           # FOV along readout, phase and slice
    dfov = np.array([0e-3, 0e-3, 0e-3]),            # Displacement of fov center
    nPoints = np.array([60, 60, 60]),                 # Number of points along readout, phase and slice
    etl = 5,                    # Echo train length
    acqTime = 2e-3,             # Acquisition time
    axes = np.array([0, 1, 2]),       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    axesEnable = np.array([1, 1, 1]), # 1-> Enable, 0-> Disable
    sweepMode = 1,               # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    phaseGradTime = 1000e-6,       # Phase and slice dephasing time
    rdPreemphasis = 1.008,
    drfPhase = 0,                           # phase of the excitation pulse (in degrees)
    dummyPulses = 1,                     # Dummy pulses for T1 stabilization
    shimming = np.array([-70, -90, 10]),       # Shimming along the X,Y and Z axes (a.u. *1e4)
    parAcqLines = 0,                        # Number of additional lines, Full sweep if 0
    ima = 0
    ):
    
    # Miscellaneous
    killingPhaseGradient = 1   # multiply phase gradient by this number
    blkTime = 10             # Deblanking time (us)
    larmorFreq = larmorFreq*1e-6
    gradRiseTime = 400e-6       # Estimated gradient rise time
    gSteps = int(gradRiseTime*1e6/5)    # Gradient ramp steps
    gradDelay = 9            # Gradient amplifier delay
    addRdPoints = 10             # Initial rd points to avoid artifact at the begining of rd
    addRdGradTime = 1000     # Additional readout gradient time to avoid turn on/off effects on the Rx channel
    gammaB = 42.56e6            # Gyromagnetic ratio in Hz/T
    rfReAmp = rfExAmp
    rfReTime = 2*rfExTime
    if axesEnable[0]==0:
        oversamplingFactor = 60
    else:
        oversamplingFactor = 6
    shimming = shimming*1e-4
    resolution = fov/nPoints
    
    # Matrix size
    nRD = nPoints[0]+2*addRdPoints
    nPH = nPoints[1]*axesEnable[1]+(1-axesEnable[1])
    nSL = nPoints[2]*axesEnable[2]+(1-axesEnable[2])
    nPoints[1] = nPH
    nPoints[2] = nSL
    
    # ETL if nPH = 1
    if etl>nPH:
        etl = nPH
    
    # parAcqLines in case parAcqLines = 0
    if parAcqLines==0:
        parAcqLines = int(nSL/2)
    
    # BW
    BW = nPoints[0]/acqTime*1e-6
    BWov = BW*oversamplingFactor
    samplingPeriod = 1/BWov
    
    # Readout dephasing time
    rdDephTime = (acqTime-gradRiseTime)/2
    
    # Phase and slice de- and re-phasing time
    if phaseGradTime==0:
        phaseGradTime = echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime
    elif phaseGradTime>echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime:
        phaseGradTime = echoSpacing/2-rfExTime/2-rfReTime/2-2*gradRiseTime
        
    # Max gradient amplitude
    rdGradAmplitude = nPoints[0]/(gammaB*fov[0]*acqTime)*axesEnable[0]
    phGradAmplitude = nPH/(2*gammaB*fov[1]*(phaseGradTime+gradRiseTime))*axesEnable[1]*killingPhaseGradient
    slGradAmplitude = nSL/(2*gammaB*fov[2]*(phaseGradTime+gradRiseTime))*axesEnable[2]

    # Change gradient values to OCRA units
    gFactor = reorganizeGfactor(axes)
    rdGradAmplitude = rdGradAmplitude/gFactor[0]*1000/5
    phGradAmplitude = phGradAmplitude
    slGradAmplitude = slGradAmplitude
    
    # Phase and slice gradient vector
    phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
    slGradients = np.linspace(-slGradAmplitude,slGradAmplitude,num=nSL,endpoint=False)
    phGradients = phGradients/gFactor[1]*1000/5
    slGradients = slGradients/gFactor[2]*1000/5
    
    # Set phase vector to given sweep mode
    ind = getIndex(phGradients, etl, nPH, sweepMode)
    phGradients = phGradients[::-1]
    phGradients = phGradients[ind]

    # Create functions
    def rfPulse(tStart,rfTime,rfAmplitude,rfPhase):
        txTime = np.array([tStart+blkTime,tStart+blkTime+rfTime])
        txAmp = np.array([rfAmplitude*np.exp(1j*rfPhase),0.])
        txGateTime = np.array([tStart,tStart+blkTime+rfTime])
        txGateAmp = np.array([1,0])
        expt.add_flodict({
            'tx0': (txTime, txAmp),
            'tx_gate': (txGateTime, txGateAmp)
            })

    def rxGate(tStart,gateTime):
        rxGateTime = np.array([tStart,tStart+gateTime])
        rxGateAmp = np.array([1,0])
        expt.add_flodict({
            'rx0_en':(rxGateTime, rxGateAmp), 
            'rx_gate': (rxGateTime, rxGateAmp), 
            })

    def gradTrap(tStart, gTime, gAmp, gAxis):
        tUp = np.linspace(tStart, tStart+gradRiseTime, num=gSteps, endpoint=False)
        tDown = tUp+gradRiseTime+gTime
        t = np.concatenate((tUp, tDown), axis=0)
        dAmp = gAmp/gSteps
        aUp = np.linspace(dAmp, gAmp, num=gSteps)
        aDown = np.linspace(gAmp-dAmp, 0, num=gSteps)
        a = np.concatenate((aUp, aDown), axis=0)
        if gAxis==0:
            expt.add_flodict({'grad_vx': (t, a+shimming[0])})
        elif gAxis==1:
            expt.add_flodict({'grad_vy': (t, a+shimming[1])})
        elif gAxis==2:
            expt.add_flodict({'grad_vz': (t, a+shimming[2])})
    
    def gradPulse(tStart, gTime, gAmp,  gAxes):
        t = np.array([tStart, tStart+gradRiseTime+gTime])
        for gIndex in range(np.size(gAxes)):
            a = np.array([gAmp[gIndex], 0])
            if gAxes[gIndex]==0:
                expt.add_flodict({'grad_vx': (t, a+shimming[0])})
            elif gAxes[gIndex]==1:
                expt.add_flodict({'grad_vy': (t, a+shimming[1])})
            elif gAxes[gIndex]==2:
                expt.add_flodict({'grad_vz': (t, a+shimming[2])})
    
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

    def createSequence():
        phIndex = 0
        slIndex = 0
        scanTime = (nPH*nSL/etl+dummyPulses)*repetitionTime
        # Set shimming
        iniSequence(20, shimming)
        for repeIndex in range(int(nPH*nSL/etl)+dummyPulses):
            # Initialize time
            t0 = 20+repetitionTime*repeIndex
            
            # Inversion pulse
            if inversionTime!=0:
                rfPulse(t0,rfReTime,rfReAmp,0)
                t0 += (rfReTime-rfExTime)/2+inversionTime
            
            # Excitation pulse
            rfPulse(t0,rfExTime,rfExAmp,drfPhase*np.pi/180)
        
            # Dephasing readout
            t0 += blkTime+rfExTime-gradDelay
            if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                gradTrap(t0, acqTime+2*addRdGradTime, rdGradAmplitude/2*rdPreemphasis, axes[0])
            
            # Echo train
            for echoIndex in range(etl):
                # Refocusing pulse
                if echoIndex == 0:
                    t0 += gradDelay+(-rfExTime+echoSpacing-rfReTime)/2-blkTime
                else:
                    t0 += gradDelay-addRdGradTime-acqTime/2+echoSpacing/2-rfReTime/2-blkTime
                rfPulse(t0, rfReTime, rfReAmp, np.pi/2)
    
                # Dephasing phase and slice gradients
                t0 += blkTime+rfReTime-gradDelay
                if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                    gradTrap(t0, phaseGradTime, phGradients[phIndex], axes[1])
                    gradTrap(t0, phaseGradTime, slGradients[slIndex], axes[2])
                
                # Readout gradient
                t0 += -rfReTime/2+echoSpacing/2-acqTime/2-gradRiseTime-addRdGradTime
                if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                    gradTrap(t0, acqTime+2*addRdGradTime, rdGradAmplitude, axes[0])
    
                # Rx gate
                t0 += gradDelay+gradRiseTime+addRdGradTime-addRdPoints/BW
                if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                    rxGate(t0, acqTime+2*addRdPoints/BW)
    
                # Rephasing phase and slice gradients
                t0 += addRdPoints/BW+acqTime-gradDelay+addRdGradTime
                if (echoIndex<etl-1 and repeIndex>=dummyPulses):
                    gradTrap(t0, phaseGradTime, -phGradients[phIndex], axes[1])
                    gradTrap(t0, phaseGradTime, -slGradients[slIndex], axes[2])
    
                # Update the phase and slice gradient
                if repeIndex>=dummyPulses:
                    if phIndex == nPH-1:
                        phIndex = 0
                        slIndex += 1
                    else:
                        phIndex += 1
                
                if phIndex == nPH-1 and slIndex == nSL-1:
                    endSequence(scanTime)
    
    
    def createFreqCalSequence():
        t0 = 20
        
        # Shimming
        iniSequence(t0, shimming)
            
        # Excitation pulse
        rfPulse(t0,rfExTime,rfExAmp,drfPhase*np.pi/180)
        
        # Refocusing pulse
        t0 += rfExTime/2+echoSpacing/2-rfReTime/2
        rfPulse(t0, rfReTime, rfReAmp, np.pi/2)
        
        # Rx
        t0 += blkTime+rfReTime/2+echoSpacing/2-acqTime/2-addRdPoints/BW
        rxGate(t0, acqTime+2*addRdPoints/BW)
        
        # Finalize sequence
        endSequence(repetitionTime)
        
    
    # Changing time parameters to us
    rfExTime = rfExTime*1e6
    rfReTime = rfReTime*1e6
    echoSpacing = echoSpacing*1e6
    repetitionTime = repetitionTime*1e6
    gradRiseTime = gradRiseTime*1e6
    phaseGradTime = phaseGradTime*1e6
    rdDephTime = rdDephTime*1e6
    inversionTime = inversionTime*1e6
    
    # Calibrate frequency
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BW = 1/samplingPeriod/oversamplingFactor
    acqTime = nPoints[0]/BW        # us
    createFreqCalSequence()
    rxd, msgs = expt.run()
    dataFreqCal = sig.decimate(rxd['rx0']*13.788, oversamplingFactor, ftype='fir', zero_phase=True)
    dataFreqCal = dataFreqCal[addRdPoints:nPoints[0]+addRdPoints]
    # Plot fid
    tVector = np.linspace(-acqTime/2, acqTime/2, num=nPoints[0],endpoint=True)*1e-3
    for ii in range(nPH):
        angle = np.unwrap(np.angle(dataFreqCal))
    # Get larmor frequency
    dPhi = angle[-1]-angle[0]
    df = dPhi/(2*np.pi*acqTime)
    larmorFreq += df
    print("f0 = %s MHz" % (round(larmorFreq, 5)))
    expt.__del__()
    
    # Create full sequence
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BW = 1/samplingPeriod/oversamplingFactor
    acqTime = nPoints[0]/BW        # us
    createSequence()
    
    # Run the experiment
    dataFull = []
    for ii in range(nScans):
        rxd, msgs = expt.run()
        rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
        # Get data
        scanData = sig.decimate(rxd['rx0'], oversamplingFactor, ftype='fir', zero_phase=True)
        dataFull = np.concatenate((dataFull, scanData), axis = 0)
    expt.__del__()
    
    # Get index for krd = 0
    # Average data
    if axesEnable[0]==0:
        indkrd0 = int(nRD/2)
    else:
        dataProv = np.reshape(dataFull, (nScans, nRD*nPH*nSL))
        dataProv = np.average(dataProv, axis=0)
        dataProv = np.reshape(dataProv, (nSL, nPH, nRD))
        # Reorganize the data acording to sweep mode
        dataTemp = dataProv*0
        for ii in range(nPH):
            dataTemp[:, ind[ii], :] = dataProv[:,  ii, :]
        dataProv = dataTemp
        # Check where is krd = 0
        dataProv = dataProv[int(nSL/2), int(nPH/2), :]
        indkrd0 = np.argmax(np.abs(dataProv))
        indkrd0 = int(nRD/2)

    # Get required readout points
    dataFull = np.reshape(dataFull, (nPH*nSL*nScans, nRD))
    dataFull = dataFull[:, indkrd0-int(nPoints[0]/2):indkrd0+int(nPoints[0]/2)]
    dataFull = np.reshape(dataFull, (1, nPoints[0]*nPH*nSL*nScans))
    
    # Average data
    data = np.reshape(dataFull, (nScans, nPoints[0]*nPH*nSL))
    data = np.average(data, axis=0)
    data = np.reshape(data, (nSL, nPH, nPoints[0]))
    
    # Reorganize the data acording to sweep mode
    dataTemp = data*0
    for ii in range(nPH):
        dataTemp[:, ind[ii], :] = data[:,  ii, :]
    
    # Do zero padding
    data = np.zeros((nPoints[2], nPoints[1], nPoints[0]))
    data = data+1j*data
    if nSL==1:
        data = dataTemp
    else:
        data[0:nSL-1, :, :] = dataTemp[0:nSL-1, :, :]
    data = np.reshape(data, (1, nPoints[0]*nPoints[1]*nPoints[2]))
        
    # Fix the position of the sample according t dfov
    kMax = nPoints/(2*fov)*axesEnable
    kRD = np.linspace(-kMax[0],kMax[0],num=nPoints[0],endpoint=False)
    kPH = np.linspace(-kMax[1],kMax[1],num=nPH,endpoint=False)
    kSL = np.linspace(-kMax[2],kMax[2],num=nSL,endpoint=False)
    kPH = kPH[::-1]
    kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
    kRD = np.reshape(kRD, (1, nPoints[0]*nPH*nSL))
    kPH = np.reshape(kPH, (1, nPoints[0]*nPH*nSL))
    kSL = np.reshape(kSL, (1, nPoints[0]*nPH*nSL))
    dPhase = np.exp(-2*np.pi*1j*(dfov[0]*kRD+dfov[1]*kPH+dfov[2]*kSL))
    data = data*dPhase
    
    # Create sampled data
    kRD = np.reshape(kRD, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    kPH = np.reshape(kPH, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    kSL = np.reshape(kSL, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    data = np.reshape(data, (nPoints[0]*nPoints[1]*nPoints[2], 1))
    
    # Get image with FFT
    data = np.reshape(data, (nSL, nPH, nPoints[0]))
    img=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
    
    # Calculate image mask
    imgM = np.abs(img)
    imgM = imgM/np.max(imgM)
    imgM[imgM<0.3] = 0
    imgM[imgM>=0.3] = 1
    imgX = np.sum(np.sum(imgM, 1), 0)
    imgY = np.sum(np.sum(imgM, 2), 0)
    imgX[imgX>0]=1
    imgY[imgY>0]=1
    xSample = np.squeeze(np.where(imgX==1))
    ySample = np.squeeze(np.where(imgY==1))
    xSpan = (xSample[-1]-xSample[0]+1)*resolution[0]*1e3
    ySpan = (ySample[-1]-ySample[0]+1)*resolution[1]*1e3
    x0 = ((xSample[-1]+xSample[0])/2*resolution[0]-fov[0]/2)*1e3
    y0 = ((ySample[-1]+ySample[0])/2*resolution[1]-fov[1]/2)*1e3
    fovOut = np.array([xSpan, ySpan])
    dfovOut = np.array([x0, y0])
    
    # Plot images
    plt.figure(3)
    imgPlot = img[round(nSL/2), :, :]
    plt.subplot(1, 2, ima)
    plt.imshow(np.abs(imgPlot), cmap='gray')
    plt.axis('off')
    if ima==1:
        plt.title('XY proyection')
    else:
        plt.title('XZ proyection')
        
    return(fovOut, dfovOut)
    

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def getIndex(g_amps, echos_per_tr, n_ph, sweep_mode):
    n2ETL=np.int32(n_ph/2/echos_per_tr)
    ind:np.int32 = [];
    if n_ph==1:
         ind = np.linspace(np.int32(n_ph)-1, 0, n_ph)
    
    else: 
        if sweep_mode==0:   # Sequential for T2 contrast
            for ii in range(np.int32(n_ph/echos_per_tr)):
               ind = np.concatenate((ind, np.arange(1, n_ph+1, n_ph/echos_per_tr)+ii))
            ind = ind-1

        elif sweep_mode==1: # Center-out for T1 contrast
            if echos_per_tr==n_ph:
                for ii in range(np.int32(n_ph/2)):
                    cont = 2*ii
                    ind = np.concatenate((ind, np.array([n_ph/2-cont/2])), axis=0);
                    ind = np.concatenate((ind, np.array([n_ph/2+1+cont/2])), axis=0);
            else:
                for ii in range(n2ETL):
                    ind = np.concatenate((ind,np.arange(n_ph/2, 0, -n2ETL)-(ii)), axis=0);
                    ind = np.concatenate((ind,np.arange(n_ph/2+1, n_ph+1, n2ETL)+(ii)), axis=0);
            ind = ind-1
        elif sweep_mode==2: # Out-to-center for T2 contrast
            if echos_per_tr==n_ph:
                ind=np.arange(1, n_ph+1, 1)
            else:
                for ii in range(n2ETL):
                    ind = np.concatenate((ind,np.arange(1, n_ph/2+1, n2ETL)+(ii)), axis=0);
                    ind = np.concatenate((ind,np.arange(n_ph, n_ph/2, -n2ETL)-(ii)), axis=0);
            ind = ind-1
        elif sweep_mode==3:
            if echos_per_tr==n_ph:
                ind = np.arange(0, n_ph, 1)
            else:
                for ii in range(int(n2ETL)):
                    ind = np.concatenate((ind, np.arange(0, n_ph, 2*n2ETL)+2*ii), axis=0)
                    ind = np.concatenate((ind, np.arange(n_ph-1, 0, -2*n2ETL)-2*ii), axis=0)

    return np.int32(ind)


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def reorganizeGfactor(axes):
    gFactor = np.array([0., 0., 0.])
    
    # Set the normalization factor for readout, phase and slice gradient
    for ii in range(3):
        if axes[ii]==0:
            gFactor[ii] = Gx_factor
        elif axes[ii]==1:
            gFactor[ii] = Gy_factor
        elif axes[ii]==2:
            gFactor[ii] = Gz_factor
    
    return(gFactor)

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


if __name__ == "__main__":
    
    print("Calibrating X and Y directions...")
    fovxy, dfovxy = fovCal_standalone(axes = (0, 1, 2), axesEnable = (1, 1, 0), ima = 1)
    print("Calibrating X and Z directions...")
    fovxz, dfovxz = fovCal_standalone(axes = (0, 2, 1), axesEnable = (1, 1, 0),  ima = 2)
    print("FOV(x,y,z) = (%s %s %s) mm" %(np.round(fovxy[0], 0), np.round(fovxy[1], 0), np.round(fovxz[1], 0)))
    print("dFOV(x,y,z) = (%s %s %s) mm" % (-np.round(dfovxy[0], 0), -np.round(dfovxy[1], 0), -np.round(dfovxz[1], 0)))
    plt.show()
