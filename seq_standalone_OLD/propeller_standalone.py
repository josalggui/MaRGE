# -*- coding: utf-8 -*-
"""
Created on Tue Feb  01 16:48:05 2022

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
from scipy.interpolate import griddata as gd
import pdb
from configs.hw_config import Gx_factor
from configs.hw_config import Gy_factor
from configs.hw_config import Gz_factor
import time
st = pdb.set_trace



#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def propellerStack_standalone(
    init_gpa=False,              # Starts the gpa
    nScans = 1,                 # NEX
    larmorFreq = 3.0746e6,      # Larmor frequency
    rfExAmp = 0.3,             # rf excitation pulse amplitude
    rfReAmp = 0.3,             # rf refocusing pulse amplitude
    rfExTime = 40e-6,          # rf excitation pulse time
    rfReTime = 80e-6,            # rf refocusing pulse time
    echoSpacing = 20e-3,        # time between echoes
    repetitionTime = 200e-3,     # TR
    inversionTime = 0e-3,           # Inversion time for Inversino Recovery experiment
    fov = np.array([6.5e-2, 6.5e-2, 13e-2]),           # FOV along readout, phase and slice
    dfov = np.array([0e-2, 0e-2, 0e-2]),            # Displacement of fov center
    nPoints = np.array([60, 60, 30]),                 # Number of points along readout, phase and slice
    etl = 5,                    # Echo train length
    undersampling = 1,               # Angular undersampling
    acqTimeMax = 2e-3,             # Maximum acquisition time (s)
    axes = np.array([2, 1, 0]),       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    sweepMode = 1,               # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    phaseGradTime = 1000e-6,       # Phase and slice dephasing time
    rdPreemphasis = 1.012,            # Readout preemphasis factor (dephasing gradient is multiplied by this number)
    drfPhase = 0,                           # phase of the excitation pulse (in degrees)
    dummyPulses = 3,                     # Dummy pulses for T1 stabilization
    shimming = np.array([-80, -100, 20]),       # Shimming along the X,Y and Z axes (a.u. *1e4)
    parAcqLines = 0,                         # Number of additional lines, Full sweep if 0
    phi0 = 45                               # Rotates the rd/ph 
    ):
    
    # Test for linearly dependent preemphasis
    rdP0 = 1.00         # premphasis for G = 0
    rdPm = 10.5e-2         # Additional preemphasis per each unit voltage
    
    # rawData fields
    rawData = {}
    inputs = {}
    kSpace = {}
    auxiliar = {}
    
    # Miscellaneous
    blkTime = 10             # Deblanking time (us)
    larmorFreq = larmorFreq*1e-6
    gradRiseTime = 400e-6       # Estimated gradient rise time
    gSteps = int(gradRiseTime*1e6/10)*0+1    # Gradient ramp steps
    gradDelay = 9            # Gradient amplifier delay
    addRdPoints = 10             # Initial rd points to avoid artifact at the begining of rd
    addRdGradTime = 1000     # Additional readout gradient time to avoid turn on/off effects on the Rx channel
    gammaB = 42.56e6            # Gyromagnetic ratio in Hz/T
    rfReAmp = rfExAmp
    rfReTime = 2*rfExTime
    shimming = shimming*1e-4
    resolution = fov/nPoints
    kMax = 1/(2*resolution)
    oversamplingFactor = 6
    auxiliar['resolution'] = resolution
    auxiliar['kMax'] = kMax
    auxiliar['gradDelay'] = gradDelay*1e-6
    auxiliar['gradRiseTime'] = gradRiseTime
    auxiliar['oversamplingFactor'] = oversamplingFactor
    auxiliar['addRdGradTime'] = addRdGradTime*1e-6
    
    # Inputs for rawData
    inputs['nScans'] = nScans
    inputs['larmorFreq'] = larmorFreq      # Larmor frequency
    inputs['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    inputs['rfReAmp'] = rfReAmp             # rf refocusing pulse amplitude
    inputs['rfExTime'] = rfExTime          # rf excitation pulse time
    inputs['rfReTime'] = rfReTime            # rf refocusing pulse time
    inputs['echoSpacing'] = echoSpacing        # time between echoes
    inputs['repetitionTime'] = repetitionTime     # TR
    inputs['inversionTime'] = inversionTime         # Inversion time for Inversion Recovery
    inputs['fov'] = fov           # FOV along readout, phase and slice
    inputs['dfov'] = dfov            # Displacement of fov center
    inputs['nPoints'] = nPoints                 # Number of points along readout, phase and slice
    inputs['etl'] = etl                    # Echo train length
    inputs['undersampling'] = undersampling   # Angular undersampling
    inputs['acqTimeMax'] = acqTimeMax             # Acquisition bandwidth
    inputs['axes'] = axes       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    inputs['sweepMode'] = sweepMode               # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    inputs['phaseGradTime'] = phaseGradTime       # Phase and slice dephasing time
    inputs['rdPreemphasis'] = rdPreemphasis
    inputs['drfPhase'] = drfPhase 
    inputs['dummyPulses'] = dummyPulses                    # Dummy pulses for T1 stabilization
    inputs['shimming'] = shimming
    
    # Calculate the acquisition bandwidth
    if nPoints[0]>nPoints[1]:
        bandwidth = nPoints[0]/acqTimeMax
    else:
        bandwidth = nPoints[1]/acqTimeMax
    auxiliar['bandwidth'] = bandwidth
    
    # Oversampled BW
    BWov = bandwidth*oversamplingFactor
    samplingPeriod = 1/BWov*1e6
    
    # Calculate the angles of each normalized k-space line
    phi = np.array([0])
    while phi[-1]<np.pi:
        dPhi = np.array([(nPoints[1]+nPoints[0])/(nPoints[0]*nPoints[1])+np.abs(nPoints[1]-nPoints[0])/(nPoints[0]*nPoints[1])*np.cos(2*phi[-1])])
        phi = np.concatenate((phi,phi[-1]+dPhi),axis=0)
    nBlocks = np.size(phi)-1
    phi = phi[0:nBlocks]
    phiEtlUnder = phi[0::etl*undersampling]
    alpha = np.pi/(phiEtlUnder[1]+phiEtlUnder[-1])
    phi = phiEtlUnder*alpha
    nBlocks = np.size(phi)
    del alpha, dPhi, phiEtlUnder
    
    # Calculate readout gradients (directions 0 and 1)
    rdGradAmp = bandwidth/(gammaB*fov)
    rdGradAmpArray = np.array([rdGradAmp[0]*np.cos(phi),rdGradAmp[1]*np.sin(phi)])
    
    # Calculate the phase gradients (directions 0 and 1)
    phGradAmp = rdGradAmp/(bandwidth*(phaseGradTime+gradRiseTime))
    phGradAmpArray = np.array([-phGradAmp[0]*np.sin(phi),phGradAmp[1]*np.cos(phi)])
    
    # Calculate the slice gradients and k-points (direction 2)
    if nPoints[2]>1:
        slGradAmp = 1/(resolution[2]*gammaB*(phaseGradTime+gradRiseTime))
    else:
        slGradAmp = 0
    if nPoints[2]%2==0:
        slGradAmpArray = np.linspace(-slGradAmp, slGradAmp, num = nPoints[2], endpoint = False)
        kSl = np.linspace(-kMax[2], kMax[2], num = nPoints[2], endpoint = False)
    else:
        slGradAmpArray = np.linspace(-slGradAmp, slGradAmp, num = nPoints[2], endpoint = True)
        kSl = np.linspace(-kMax[2], kMax[2], num = nPoints[2], endpoint = True)
    
    # Calculate the number of readouts and acqTime as a function of phi
    nrdx = nPoints[0]*np.abs(np.cos(phi))
    nrdy = nPoints[1]*np.abs(np.sin(phi))
    nrd = np.int32(np.round(np.sqrt(np.power(nrdx,2)+np.power(nrdy,2))))
    acqTime = nrd/bandwidth
    del nrdx, nrdy
    
    # Normalized kmax (time space)
    kMaxN = 1/(2*gammaB*resolution*rdGradAmp)
    
    # Create k-space and get phase gradients
    kPropellerX = np.array([])
    kPropellerY = np.array([])
    kPropellerZ = np.array([])
    phGradAmpBlock = {}
    ind = getIndex(etl, sweepMode)
    for blockIndex in range(nBlocks):
        if nrd[blockIndex]%2==0:
            nprov = np.linspace(-1.0, 1.0, num = nrd[blockIndex], endpoint = False)
        else:
            nprov = np.linspace(-1.0, 1.0, num = nrd[blockIndex], endpoint = True)
        # Main lines in normalized k-space
        kLineN = np.array([nprov*np.cos(phi[blockIndex])*kMaxN[0],
                           nprov*np.sin(phi[blockIndex])*kMaxN[1]])
        # Add lines for propeller in normalized k-space
        dkN = np.array([kLineN[1,1]-kLineN[1,0],
                        kLineN[0,0]-kLineN[0,1]])*undersampling
        for slIndex in range(nPoints[2]):
            for echoIndex in range(etl):
                # True k-space
                if etl%2==0:
                    kLine = np.array([(kLineN[0,:]-dkN[0]*(echoIndex-etl/2))*gammaB*np.abs(rdGradAmp[0]),
                                      (kLineN[1,:]-dkN[1]*(echoIndex-etl/2))*gammaB*np.abs(rdGradAmp[1]),
                                      np.ones(nrd[blockIndex])*kSl[slIndex]])
                else:
                    kLine = np.array([(kLineN[0,:]-dkN[0]*(echoIndex-(etl-1)/2))*gammaB*np.abs(rdGradAmp[0]),
                                      (kLineN[1,:]-dkN[1]*(echoIndex-(etl-1)/2))*gammaB*np.abs(rdGradAmp[1]),
                                      np.ones(nrd[blockIndex])*kSl[slIndex]])
                # Save points into propeller k-points
                kPropellerX = np.concatenate((kPropellerX, kLine[0, :]), axis = 0)
                kPropellerY = np.concatenate((kPropellerY, kLine[1, :]), axis = 0)
                kPropellerZ = np.concatenate((kPropellerZ, kLine[2, :]), axis = 0)
        # Define here all the phase gradients for current block
        if etl%2==1:
            phGradAmpBlock[blockIndex] = np.array([np.linspace(-(etl-1)/2,(etl-1)/2,num=etl,endpoint=True)*phGradAmpArray[0,blockIndex],
                                                   np.linspace(-(etl-1)/2,(etl-1)/2,num=etl,endpoint=True)*phGradAmpArray[1,blockIndex]])
        else:
            phGradAmpBlock[blockIndex] = np.array([np.linspace(-etl/2,etl/2,num=etl,endpoint=False)*phGradAmpArray[0,blockIndex],
                                                   np.linspace(-etl/2,etl/2,num=etl,endpoint=False)*phGradAmpArray[1,blockIndex]])
        # Reorganize gradients according to the sweep mode
        phGradAmpBlock[blockIndex] = phGradAmpBlock[blockIndex][:,ind]
    del kLineN, kLine, dkN, nprov    
    
    # Generate cartesian k-points
    if nPoints[0]%2==1:
        kCartesianX = np.linspace(-kMax[0], kMax[0], num = nPoints[0], endpoint = True)
    else:
        kCartesianX = np.linspace(-kMax[0], kMax[0], num = nPoints[0], endpoint = False)
    if nPoints[1]%2==1:
        kCartesianY = np.linspace(-kMax[1], kMax[1], num = nPoints[1], endpoint = True)
    else:
        kCartesianY = np.linspace(-kMax[1], kMax[1], num = nPoints[1], endpoint = False)
    if nPoints[2]%2==1:
        kCartesianZ = np.linspace(-kMax[2], kMax[2], num = nPoints[2], endpoint = True)
    else:
        kCartesianZ = np.linspace(-kMax[2], kMax[2], num = nPoints[2], endpoint = False)
    kCartesianY, kCartesianZ, kCartesianX = np.meshgrid(kCartesianY, kCartesianZ, kCartesianX)
    kCartesianX = np.squeeze(np.reshape(kCartesianX, (1, nPoints[0]*nPoints[1]*nPoints[2])))
    kCartesianY = np.squeeze(np.reshape(kCartesianY, (1, nPoints[0]*nPoints[1]*nPoints[2])))
    kCartesianZ = np.squeeze(np.reshape(kCartesianZ, (1, nPoints[0]*nPoints[1]*nPoints[2])))
    if nPoints[2]==1:
        kCartesianZ *=0
    
    # Change gradient values to OCRA units
    gFactor = reorganizeGfactor(axes)
    rdGradAmpArray[0, :] = rdGradAmpArray[0, :]/gFactor[0]*1000/10
    rdGradAmpArray[1, :] = rdGradAmpArray[1, :]/gFactor[1]*1000/10
    slGradAmpArray = slGradAmpArray/gFactor[2]*1000/10
    for blockIndex in range(nBlocks):
        phGradAmpBlock[blockIndex][0, :] = phGradAmpBlock[blockIndex][0, :]/gFactor[0]*1000/10
        phGradAmpBlock[blockIndex][1, :] = phGradAmpBlock[blockIndex][1, :]/gFactor[1]*1000/10
    
    # Create sequence
    def createSequence():
        nRepetitions = nBlocks*nPoints[2]+dummyPulses
        scanTime = nRepetitions*repetitionTime
        # Set shimming
        iniSequence(20, shimming)
        slIndex = 0
        blockIndex = 0
        for repeIndex in range(nRepetitions):
            # Initialize time
            tRep = 20e3+inversionTime+repetitionTime*repeIndex
            
            # Inversion pulse
            if inversionTime!=0:
                t0 = tRep-inversionTime-rfReTime/2-blkTime
                rfPulse(t0,rfReTime,rfReAmp,0)
                
            # Excitation pulse
            t0 = tRep-rfExTime/2-blkTime
            rfPulse(t0,rfExTime,rfExAmp,drfPhase*np.pi/180)
        
            # Dephasing readout
            t0 = tRep+rfExTime/2-gradDelay
            rdPreX = rdP0+rdPm*np.abs(rdGradAmpArray[0,blockIndex])
            rdPreY = rdP0+rdPm*np.abs(rdGradAmpArray[1,blockIndex])
            if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                gradTrap(t0, nrd[blockIndex]/bandwidth+2*addRdGradTime, rdGradAmpArray[0,blockIndex]/2*rdPreX, axes[0])
                gradTrap(t0, nrd[blockIndex]/bandwidth+2*addRdGradTime, rdGradAmpArray[1,blockIndex]/2*rdPreY, axes[1])
#            if repeIndex>=dummyPulses:         # This is to account for dummy pulses
#                gradTrap(t0, nrd[blockIndex]/bandwidth+2*addRdGradTime, rdGradAmpArray[0,blockIndex]/2*rdPreemphasis, axes[0])
#                gradTrap(t0, nrd[blockIndex]/bandwidth+2*addRdGradTime, rdGradAmpArray[1,blockIndex]/2*rdPreemphasis, axes[1])
            
            # Echo train
            for echoIndex in range(etl):
                tEcho = tRep+echoSpacing*(echoIndex+1)
                
                # Refocusing pulse
                t0 = tEcho-echoSpacing/2-rfReTime/2-blkTime
                rfPulse(t0, rfReTime, rfReAmp, np.pi/2)
    
                # Dephasing phase and slice gradients
                t0 = tEcho-echoSpacing/2+rfReTime/2-gradDelay
                if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                    gradTrap(t0, phaseGradTime, phGradAmpBlock[blockIndex][0,echoIndex], axes[0])
                    gradTrap(t0, phaseGradTime, phGradAmpBlock[blockIndex][1,echoIndex], axes[1])
                    gradTrap(t0, phaseGradTime, slGradAmpArray[slIndex], axes[2])
                    
                # Readout gradient
                if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                    if nrd[blockIndex]%2==0:
                        t0 = tEcho-(nrd[blockIndex]+1)/(2*bandwidth)-addRdGradTime-gradRiseTime-gradDelay
                        gradTrap(t0, (nrd[blockIndex]+1)/bandwidth+2*addRdGradTime, rdGradAmpArray[0,blockIndex], axes[0])
                        gradTrap(t0, (nrd[blockIndex]+1)/bandwidth+2*addRdGradTime, rdGradAmpArray[1,blockIndex], axes[1])
                    else:
                        t0 = tEcho-nrd[blockIndex]/(2*bandwidth)-addRdGradTime-gradRiseTime-gradDelay
                        gradTrap(t0, nrd[blockIndex]/bandwidth+2*addRdGradTime, rdGradAmpArray[0,blockIndex], axes[0])
                        gradTrap(t0, nrd[blockIndex]/bandwidth+2*addRdGradTime, rdGradAmpArray[1,blockIndex], axes[1])
                    
                # Rx gate
                if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                    if nrd[blockIndex]%2==0:
                        t0 = tEcho-(nrd[blockIndex]+1)/(2*bandwidth)-addRdPoints/bandwidth-1/(2*bandwidth)
                    else:
                        t0 = tEcho-(nrd[blockIndex])/(2*bandwidth)-addRdPoints/bandwidth-1/(2*bandwidth)
                    rxGate(t0, (nrd[blockIndex]+2*addRdPoints)/bandwidth)
                
                # Rephasing phase and slice gradients
                t0 = tEcho+nrd[blockIndex]/(2*bandwidth)+addRdGradTime+gradRiseTime
                if (echoIndex<etl-1 and repeIndex>=dummyPulses):
                    gradTrap(t0, phaseGradTime, -phGradAmpBlock[blockIndex][0,echoIndex], axes[0])
                    gradTrap(t0, phaseGradTime, -phGradAmpBlock[blockIndex][1,echoIndex], axes[1])
                    gradTrap(t0, phaseGradTime, -slGradAmpArray[slIndex], axes[2])
    
            # Update the block and slice index
            if repeIndex>=dummyPulses:
                if slIndex == nPoints[2]-1:
                    blockIndex += 1
                    slIndex = 0
                else:
                    slIndex += 1
            
            if repeIndex == nRepetitions:
                endSequence(scanTime)
    
    # Frequency calibration function
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
        t0 += blkTime+rfReTime/2+echoSpacing/2-acqTimeFreqCal/2-addRdPoints/bandwidth
        rxGate(t0, acqTimeFreqCal+2*addRdPoints/bandwidth)
        
        # Finalize sequence
        endSequence(repetitionTime)
    
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
    
    # Changing time parameters to us
    rfExTime = rfExTime*1e6
    rfReTime = rfReTime*1e6
    echoSpacing = echoSpacing*1e6
    repetitionTime = repetitionTime*1e6
    gradRiseTime = gradRiseTime*1e6
    phaseGradTime = phaseGradTime*1e6
    inversionTime = inversionTime*1e6
    bandwidth = bandwidth*1e-6
    
    # Calibrate frequency
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    bandwidth = 1/samplingPeriod/oversamplingFactor
    acqTimeFreqCal = nPoints[0]/bandwidth        # us
    auxiliar['bandwidth'] = bandwidth*1e6
    createFreqCalSequence()
    rxd, msgs = expt.run()
    dataFreqCal = sig.decimate(rxd['rx0']*13.788, oversamplingFactor, ftype='fir', zero_phase=True)
    dataFreqCal = dataFreqCal[addRdPoints:nPoints[0]+addRdPoints]
    # Plot fid
    plt.figure(1)
    tVector = np.linspace(-acqTimeFreqCal/2, acqTimeFreqCal/2, num=nPoints[0],endpoint=True)*1e-3
    plt.subplot(1, 2, 1)
    plt.plot(tVector, np.abs(dataFreqCal))
    plt.title("Signal amplitude")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (mV)")
    plt.subplot(1, 2, 2)
    angle = np.unwrap(np.angle(dataFreqCal))
    plt.title("Signal phase")
    plt.xlabel("Time (ms)")
    plt.ylabel("Phase (rad)")
    plt.plot(tVector, angle)
    # Get larmor frequency
    dPhi = angle[-1]-angle[0]
    df = dPhi/(2*np.pi*acqTimeFreqCal)
    larmorFreq += df
    auxiliar['larmorFreq'] = larmorFreq*1e6
    print("f0 = %s MHz" % (round(larmorFreq, 5)))
    # Plot sequence:
#    expt.plot_sequence()
#    plt.show()
    # Delete experiment:
    expt.__del__()
    
    # Create full sequence
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    bandwidth = 1/samplingPeriod/oversamplingFactor
    auxiliar['bandwidth'] = bandwidth*1e6
    createSequence()

    # Plot sequence:
#    expt.plot_sequence()
#    plt.show()
    
    # Run the experiment
    seqTime = nPoints[2]*nBlocks*repetitionTime*1e-6*nScans
    print("Expected scan time = %s s" %(round(seqTime, 0)))
    print("Runing sequence...")
    tStart = time.time()
    dataFull = []
    for ii in range(nScans):
        rxd, msgs = expt.run()
        rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
        # Get data
        scanData = sig.decimate(rxd['rx0'], oversamplingFactor, ftype='fir', zero_phase=True)
        dataFull = np.concatenate((dataFull, scanData), axis = 0)
    expt.__del__()
    tEnd = time.time()
    print("True scan time = %s s" %(round(tEnd-tStart, 0)))
    
    # Average data
    print("Averaging data...")
    nrdTotal = np.sum(nrd+2*addRdPoints)*etl
    dataProvA = np.reshape(dataFull, (nScans, nrdTotal*nPoints[2]))
    dataProvA = np.average(dataProvA, axis=0)
    
    # Reorganize data according to sweep mode
    print("Reorganizing data according to sweep mode...")
    dataProvB = np.zeros(np.size(dataProvA), dtype=complex)
    n0 = 0
    n1 = 0
    for blockIndex in range(nBlocks):
        n1 += nPoints[2]*etl*(nrd[blockIndex]+2*addRdPoints)
        dataProvC = np.reshape(dataProvA[n0:n1], (nPoints[2], etl, nrd[blockIndex]+2*addRdPoints))
        dataProvD = dataProvC*0
        for ii in range(etl):
            dataProvD[:, ind[ii], :] = dataProvC[:,  ii, :]
        dataProvD = np.reshape(dataProvD, (1, nPoints[2]*etl*(nrd[blockIndex]+2*addRdPoints)))
        dataProvB[n0:n1] = dataProvD
        n0 = n1
    del dataProvC,  dataProvD,  n0,  n1
    
    # Delete the additional rd points and fix echo delay!!
    print("Deleting additional readout points...")
    dataPropeller = np.zeros(nrdTotal*nPoints[2]-2*addRdPoints*nPoints[2]*nBlocks*etl, dtype=complex)
    n0 = 0
    n1 = 0
    n2 = 0
    n3 = 0
    for blockIndex in range(nBlocks):
        n1 += nPoints[2]*etl*(nrd[blockIndex]+2*addRdPoints)
        n3 += nPoints[2]*etl*(nrd[blockIndex])
        dataProvC = np.reshape(dataProvB[n0:n1], (nPoints[2], etl, nrd[blockIndex]+2*addRdPoints))
        if etl%2==0: # Asumo que nPoints[2] es par
            k0Line = dataProvC[int((nPoints[2]-1)/2), int(etl/2), :]
        else:
            k0Line = dataProvC[int((nPoints[2]-1)/2), int((etl-1)/2), :]
        k0 = np.argmax(np.abs(k0Line))
#        k0 = int((nrd[blockIndex]+2*addRdPoints)/2-2)
        if nrd[blockIndex]%2==0:
            dataProvC = dataProvC[:, :, k0-int(nrd[blockIndex]/2):k0+int(nrd[blockIndex]/2)]
        else:
            dataProvC = dataProvC[:, :, k0-int((nrd[blockIndex]-1)/2):k0+int((nrd[blockIndex]+1)/2)]
        dataProvC = np.reshape(dataProvC, (1,nPoints[2]*etl*nrd[blockIndex]))
        dataPropeller[n2:n3] = dataProvC
        n0 = n1
        n2 = n3
    del dataProvA, dataProvB, dataProvC,  n0,  n1,  n2,  n3
    
    # Regridding to cartesian k-space
#    print("Regridding to cartesian grid...")
#    tStart = time.time()
#    if nPoints[2]==1:
#        dataCartesian = gd((kPropellerX, kPropellerY), dataPropeller, (kCartesianX, kCartesianY), method='linear', fill_value=0.)
#    else:
#        dataCartesian = gd((kPropellerX, kPropellerY, kPropellerZ), dataPropeller, (kCartesianX, kCartesianY, kCartesianZ), method='linear', fill_value=0.)
#    tEnd = time.time()
#    print("Regridding done in = %s s" %(round(tEnd-tStart, 0)))
    kPropeller = np.array([kPropellerX, kPropellerY, kPropellerZ, dataPropeller])
#    kCartesian = np.array([kCartesianX, kCartesianY, kCartesianZ, dataCartesian])
    kCartesian = np.array([kCartesianX, kCartesianY, kCartesianZ])
    del kCartesianX, kCartesianY, kCartesianZ,  kPropellerX,  kPropellerY,  kPropellerZ,  dataPropeller
    auxiliar['kMax'] = kMax
    kSpace['sampled'] = np.transpose(kCartesian)
    kSpace['sampledNonCart'] = np.transpose(kPropeller)
    print("Done!")
    
#    # Get image with FFT
#    dataCartesian = np.reshape(dataCartesian, (nPoints[2], nPoints[1], nPoints[0]))
#    img = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(dataCartesian)))
#    kSpace['image'] = img
    
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
    auxiliar['fileName'] = "%s.%s.mat" % ("PROPELLER STACK",dt_string)
    rawData['inputs'] = inputs
    rawData['auxiliar'] = auxiliar
    rawData['kSpace'] = kSpace
    rawdata = {}
    rawdata['rawData'] = rawData
    savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, "PROPELLER STACK",dt_string),  rawdata)
    
#    # Plot k-space
#    plt.figure(3)
#    dataPlot = dataCartesian[round(nPoints[2]/2), :, :]
#    plt.subplot(121)
#    plt.imshow(np.log(np.abs(dataPlot)),cmap='gray')
#    plt.axis('off')
#    # Plot image
#    imgPlot = img[round(nPoints[2]/2), :, :]
#    plt.subplot(122)
#    plt.imshow(np.abs(imgPlot), cmap='gray')
#    plt.axis('off')
#    plt.title("PROPELLER_STACK.%s.mat" % (dt_string))
#    plt.show()
    

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def getIndex(etl, sweepMode):
    ind = np.zeros(etl)
    if sweepMode==0:   # Sequential for T2 contrast
        ind = np.arange(etl)
    elif sweepMode==1: # Center-out for T1 contrast
        if etl%2==0:
            indA = np.arange(etl/2, etl)
            indB = np.arange(etl/2-1, -1, -1)
            ind[0::2] = indA
            ind[1::2] = indB
        else:
            indA = np.arange((etl+1)/2, etl)
            indB = np.arange((etl-1)/2-1, -1, -1)
            ind[0] = (etl-1)/2
            ind[1::2] = indA
            ind[2::2] = indB
    elif sweepMode==2: # Out-to-center for T2 contrast
        if etl%2==0:
            indA = np.arange(etl/2, etl)
            indB = np.arange(etl/2-1, -1, -1)
            ind[0::2] = indA
            ind[1::2] = indB
        else:
            indA = np.arange((etl+1)/2, etl)
            indB = np.arange((etl-1)/2-1, -1, -1)
            ind[0] = (etl-1)/2
            ind[1::2] = indA
            ind[2::2] = indB
        ind = ind[::-1]

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

    propellerStack_standalone()
