# -*- coding: utf-8 -*-
"""
Created on Sat Nov  13 13:45:05 2021

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


def gre_standalone(
    init_gpa=False,              # Starts the gpa
    nScans = 30,                 # NEX
    larmorFreq = 3.076e6,      # Larmor frequency
    rfExAmp = 0.05,             # rf excitation pulse amplitude
    rfExTime = 30e-6,          # rf excitation pulse time
    echoTime = 2e-3,              # TE
    repetitionTime = 10e-3,     # TR
    fov = np.array([13e-2, 13e-2, 13e-2]),           # FOV along readout, phase and slice
    dfov = np.array([0e-3, 2e-3, -5e-3]),            # Displacement of fov center
    nPoints = np.array([60, 60, 30]),                 # Number of points along readout, phase and slice
    acqTime = 1e-3,             # Acquisition time
    axes = np.array([1, 2, 0]),       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    axesEnable = np.array([1, 1, 1]), # 1-> Enable, 0-> Disable
    dephaseGradTime = 1000e-6,       # Phase and slice dephasing time
    rdPreemphasis = 1.0,                               # Preemphasis factor for readout dephasing
    drfPhase = 0,                           # phase of the excitation pulse (in degrees)
    dummyPulses = 20,                     # Dummy pulses for T1 stabilization
    shimming = np.array([-80, -100, 10]),       # Shimming along the X,Y and Z axes (a.u. *1e4)
    parAcqLines = 0,                         # Number of additional lines, Full sweep if 0
    plotSeq = 1):
    
    # rawData fields
    rawData = {}
    
    # Miscellaneous
    blkTime = 1             # Deblanking time (us)
    larmorFreq = larmorFreq*1e-6
    gradRiseTime = 100e-6       # Estimated gradient rise time
    gradDelay = 9            # Gradient amplifier delay
    addRdPoints = 5             # Initial rd points to avoid artifact at the begining of rd
    gammaB = 42.56e6            # Gyromagnetic ratio in Hz/T
    oversamplingFactor = 6
    addRdGradTime = 200e-6     # Additional readout gradient time to avoid turn on/off effects on the Rx channel
    shimming = shimming*1e-4
    rawData['gradDelay'] = gradDelay*1e-6
    rawData['gradRiseTime'] = gradRiseTime
    rawData['oversamplingFactor'] = oversamplingFactor
    rawData['addRdGradTime'] = addRdGradTime*1e-6
    
    # Matrix size
    nRD = nPoints[0]+2*addRdPoints
    nPH = nPoints[1]*axesEnable[1]+(1-axesEnable[1])
    nSL = nPoints[2]*axesEnable[2]+(1-axesEnable[2])
    nPoints[1] = nPH
    nPoints[2] = nSL
    
    # rawData for rawData
    rawData['nScans'] = nScans
    rawData['larmorFreq'] = larmorFreq      # Larmor frequency
    rawData['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    rawData['rfExTime'] = rfExTime          # rf excitation pulse time
    rawData['echoTime'] = echoTime        # TE
    rawData['repetitionTime'] = repetitionTime     # TR
    rawData['fov'] = fov           # FOV along readout, phase and slice
    rawData['dfov'] = dfov            # Displacement of fov center
    rawData['nPoints'] = nPoints                 # Number of points along readout, phase and slice
    rawData['acqTime'] = acqTime             # Acquisition time
    rawData['axes'] = axes       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    rawData['axesEnable'] = axesEnable # 1-> Enable, 0-> Disable
    rawData['phaseGradTime'] = dephaseGradTime       # Phase and slice dephasing time
    rawData['rdPreemphasis'] = rdPreemphasis
    rawData['drfPhase'] = drfPhase 
    rawData['dummyPulses'] = dummyPulses                    # Dummy pulses for T1 stabilization
    rawData['shimming'] = shimming
    rawData['parAcqLines'] = parAcqLines
    
    # parAcqLines in case parAcqLines = 0
    if parAcqLines==0:
        parAcqLines = int(nSL/2)
    
    # BW
    BW = nPoints[0]/acqTime*1e-6
    BWov = BW*oversamplingFactor
    samplingPeriod = 1/BWov
    
    # Readout rephasing time
    rdRephTime = acqTime+2*addRdGradTime
    
    # Check if dephasing grad time is ok
    maxDephaseGradTime = echoTime-(rfExTime+rdRephTime)-3*gradRiseTime
    if dephaseGradTime==0 or dephaseGradTime>maxDephaseGradTime:
        dephaseGradTime = maxDephaseGradTime
    
    # Max gradient amplitude
    rdGradAmplitude = nPoints[0]/(gammaB*fov[0]*acqTime)*axesEnable[0]
    rdDephGradAmplitude = -rdGradAmplitude*(rdRephTime+gradRiseTime)/(2*(dephaseGradTime+gradRiseTime))
    phGradAmplitude = nPH/(2*gammaB*fov[1]*(dephaseGradTime+gradRiseTime))*axesEnable[1]
    slGradAmplitude = nSL/(2*gammaB*fov[2]*(dephaseGradTime+gradRiseTime))*axesEnable[2]

    # Phase and slice gradient vector
    phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
    slGradients = np.linspace(-slGradAmplitude,slGradAmplitude,num=nSL,endpoint=False)
    rawData['phGradients'] = phGradients
    rawData['slGradients'] = slGradients
    
    # Change gradient values to OCRA units
    gFactor = reorganizeGfactor(axes)
    rdGradAmplitude = rdGradAmplitude/gFactor[0]*1000/5
    rdDephGradAmplitude = rdDephGradAmplitude/gFactor[0]*1000/5
    phGradAmplitude = phGradAmplitude/gFactor[1]*1000/5
    slGradAmplitude = slGradAmplitude/gFactor[2]*1000/5
    phGradients = phGradients/gFactor[1]*1000/5
    slGradients = slGradients/gFactor[2]*1000/5
    if np.abs(rdGradAmplitude)>1:
        return(0)
    if np.abs(rdDephGradAmplitude)>1:
        return(0)
    if np.abs(phGradAmplitude)>1 or np.abs(slGradAmplitude)>1:
        return(0)
    
    # Initialize the experiment
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BW = 1/samplingPeriod/oversamplingFactor
    acqTime = nPoints[0]/BW        # us
    rawData['bandwidth'] = BW*1e6
    
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
    def gradPulse(tStart, gTime, gAmp,  gAxes):
        t = np.array([tStart, tStart+gradRiseTime+gTime])
        a = np.array([gAmp, 0])
        if gAxes==0:
            expt.add_flodict({'grad_vx': (t, a+shimming[0])})
        elif gAxes==1:
            expt.add_flodict({'grad_vy': (t, a+shimming[1])})
        elif gAxes==2:
            expt.add_flodict({'grad_vz': (t, a+shimming[2])})
    
    def endSequence(tEnd):
        expt.add_flodict({
                'grad_vx': (np.array([tEnd]),np.array([0]) ), 
                'grad_vy': (np.array([tEnd]),np.array([0]) ), 
                'grad_vz': (np.array([tEnd]),np.array([0]) ),
             })
             
    def iniSequence(tEnd):
        expt.add_flodict({
                'grad_vx': (np.array([tEnd]),np.array([shimming[0]]) ), 
                'grad_vy': (np.array([tEnd]),np.array([shimming[1]]) ), 
                'grad_vz': (np.array([tEnd]),np.array([shimming[2]]) ),
             })

    # Changing time parameters to us
    rfExTime = rfExTime*1e6
    echoTime = echoTime*1e6
    repetitionTime = repetitionTime*1e6
    gradRiseTime = gradRiseTime*1e6
    dephaseGradTime = dephaseGradTime*1e6
    rdRephTime = rdRephTime*1e6
    addRdGradTime = addRdGradTime*1e6
    
    # Create sequence instructions
    phIndex = 0
    slIndex = 0
    scanTime = (nPH*nSL+dummyPulses)*repetitionTime
    nRepetitions = nPH*nSL+dummyPulses
    # Set shimming
    iniSequence(20)
    for repeIndex in range(nRepetitions):
        # Initialize time
        t0 = 20+repetitionTime*repeIndex
        
        # Excitation pulse
        rfPulse(t0,rfExTime,rfExAmp,drfPhase*np.pi/180)
    
        # Dephasing gradients
        t0 += blkTime+rfExTime-gradDelay
        gradPulse(t0, dephaseGradTime, rdDephGradAmplitude*rdPreemphasis, axes[0])
        gradPulse(t0, dephaseGradTime, phGradients[phIndex], axes[1])
        gradPulse(t0, dephaseGradTime, slGradients[slIndex], axes[2])
        
        # Rephasing readout gradient
        t0 = 20+repetitionTime*repeIndex+blkTime+rfExTime/2+echoTime-rdRephTime-gradRiseTime-gradDelay
        gradPulse(t0, rdRephTime, rdGradAmplitude, axes[0])
        
        # Rx gate
        t0 += gradDelay+gradRiseTime+addRdGradTime-addRdPoints/BW
        if repeIndex>=dummyPulses:         # This is to account for dummy pulses
            rxGate(t0, acqTime+2*addRdPoints/BW)
        
        # Spoiler
        t0 += acqTime+2*addRdPoints/BW+addRdGradTime+gradRiseTime
        gradPulse(t0, dephaseGradTime, -rdDephGradAmplitude*rdPreemphasis, axes[0])
        gradPulse(t0, dephaseGradTime, phGradients[phIndex], axes[1])
        gradPulse(t0, dephaseGradTime, slGradients[slIndex], axes[2])
        
        # Update the phase and slice gradient
        if repeIndex>=dummyPulses:
            if phIndex == nPH-1:
                phIndex = 0
                slIndex += 1
            else:
                phIndex += 1
        
        if repeIndex == nRepetitions:
            endSequence(scanTime)
    
    # Plot sequence:
    if plotSeq==1:
        expt.plot_sequence()
        plt.show()
    
    # Run the experiment
    dataFull = []
    for ii in range(nScans):
        rxd, msgs = expt.run()
        rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
        # Get data
        scanData = sig.decimate(rxd['rx0'], oversamplingFactor, ftype='fir', zero_phase=True)
        dataFull = np.concatenate((dataFull, scanData), axis = 0)
    # Delete experiment:
    expt.__del__()
    
    # Delete the addRdPoints
#    nPoints[0] = nRD
    dataFull = np.reshape(dataFull, (nPH*nSL*nScans, nRD))
    dataFull = dataFull[:, addRdPoints:addRdPoints+nPoints[0]]
    dataFull = np.reshape(dataFull, (1, nPoints[0]*nPH*nSL*nScans))
    
    # Average data
    data = np.reshape(dataFull, (nScans, nPoints[0]*nPH*nSL))
    data = np.average(data, axis=0)
    data = np.reshape(data, (nSL, nPH, nPoints[0]))
    
    # Do zero padding
    dataTemp = data
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
    rawData['kMax'] = kMax
    rawData['sampled'] = np.concatenate((kRD, kPH, kSL, data), axis=1)

    # Get image with FFT
    data = np.reshape(data, (nSL, nPH, nPoints[0]))
    img=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
    rawData['image'] = img
    
    # Plot provisional image
#    plt.figure(1, figsize=(12, 6), dpi=80)
#    tVector = np.linspace(-acqTime/2, acqTime/2, num=nPoints[0],endpoint=False)*1e-3
#    for ii in range(nPH):
#        plt.subplot(1, 2, 1)
#        plt.plot(tVector, np.abs(data[0, ii, :]))
#        plt.xlabel('Time (ms)')
#        plt.ylabel('Signal (mV)')
#        
#        plt.subplot(1, 2, 2)
#        plt.plot(tVector, np.angle(data[0, ii, :]))
#        plt.ylim([-np.pi, np.pi])
#        plt.xlabel('Time (ms)')
#        plt.ylabel('Phase (rad)')
    
    
    
    # Plot data for 1D case
    if (nPH==1 and nSL==1):
        # Plot k-space
        plt.figure(2)
        dataPlot = data[0, 0, :]
        plt.subplot(1, 2, 1)
        if axesEnable[0]==0:
            tVector = np.linspace(-acqTime/2, acqTime/2, num=nPoints[0],endpoint=False)*1e-3
            sMax = np.max(np.abs(dataPlot))
            indMax = np.argmax(np.abs(dataPlot))
            timeMax = tVector[indMax]
            sMax3 = sMax/3
            dataPlot3 = np.abs(np.abs(dataPlot)-sMax3)
            indMin = np.argmin(dataPlot3)
            timeMin = tVector[indMin]
            T2 = np.abs(timeMax-timeMin)
            plt.plot(tVector, np.abs(dataPlot))
            plt.plot(tVector, np.real(dataPlot))
            plt.plot(tVector, np.imag(dataPlot))
            plt.xlabel('t (ms)')
            plt.ylabel('Signal (mV)')
            print(T2)
        else:
            plt.plot(kRD[0, :], np.abs(dataPlot))
            plt.yscale('log')
            plt.xlabel('krd (mm^-1)')
            plt.ylabel('Signal (mV)')
        # Plot image
        plt.subplot(122)
        img = img[0, 0, :]
        if axesEnable[0]==0:
            xAxis = np.linspace(-BW/2, BW/2, num=nPoints[0], endpoint=False)*1e3
            plt.plot(xAxis, np.abs(img), '.')
            plt.xlabel('Frequency (kHz)')
            plt.ylabel('Density (a.u.)')
            print(np.max(np.abs(img)))
        else:
            xAxis = np.linspace(-fov[0]/2*1e2, fov[0]/2*1e2, num=nPoints[0], endpoint=False)
            plt.plot(xAxis, np.abs(img))
            plt.xlabel('Position RD (cm)')
            plt.ylabel('Density (a.u.)')
    else:
        # Plot k-space
        plt.figure(2)
        dataPlot = data[round(nSL/2), :, :]
        plt.subplot(131)
        plt.imshow(np.log(np.abs(dataPlot)),cmap='gray')
        plt.title('k-Space')
        plt.axis('off')
        # Plot image
        imgPlot = img[round(nSL/2), :, :]
        plt.subplot(132)
        plt.imshow(np.abs(imgPlot), cmap='gray')
        plt.axis('off')
        plt.title('Image magnitude')
        
        plt.subplot(133)
        plt.imshow(np.angle(imgPlot), cmap='gray')
        plt.axis('off')
        plt.title('Image phase')
        
#        fig = plt.figure(3)
#        img2Darg = np.angle(imgPlot)
#        X = np.arange(-1, 1, 2/(np.size(imgPlot, 1)))
#        Y = np.arange(-1, 1, 2/(np.size(imgPlot, 0)))
#        X,  Y = np.meshgrid(X, Y)
#        ax = fig.gca(projection='3d')
#        surf = ax.plot_surface(X, Y, img2Darg)
    
    # Save data
    dt = datetime.now()
    dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
    dt2 = date.today()
    dt2_string = dt2.strftime("%Y.%m.%d")
    if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
        os.makedirs('experiments/acquisitions/%s' % (dt2_string))
            
    if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
        os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
    rawData['name'] = "%s.%s.mat" % ("GRE",dt_string)
    savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, "GRE",dt_string),  rawData)
    print(rawData['name'])
    plt.show()
    

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

    gre_standalone()
