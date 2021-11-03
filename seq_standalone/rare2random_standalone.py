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


def rare_standalone(
    init_gpa=False,              # Starts the gpa
    nScans = 1,                 # NEX
    larmorFreq = 3.08e6,      # Larmor frequency
    rfExAmp = 0.3,             # rf excitation pulse amplitude
    rfReAmp = 0.3,             # rf refocusing pulse amplitude
    rfExTime = 35e-6,          # rf excitation pulse time
    rfReTime = 70e-6,            # rf refocusing pulse time
    echoSpacing = 20e-3,        # time between echoes
    inversionTime = 0,       # Inversion recovery time
    repetitionTime = 600e-3,     # TR
    fov = np.array([12e-2, 12e-2, 12e-2]),           # FOV along readout, phase and slice
    dfov = np.array([0e-2, 0e-2, 0e-2]),            # Displacement of fov center
    nPoints = np.array([120, 120, 40]),                 # Number of points along readout, phase and slice
    etl = 15,                    # Echo train length
    acqTime = 2e-3,             # Acquisition time
    axes = np.array([0, 1, 2]),       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    axesEnable = np.array([1, 1, 0]), # 1-> Enable, 0-> Disable
    sweepMode = 1,               # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    phaseGradTime = 500e-6,       # Phase and slice dephasing time
    rdPreemphasis = 1.000,
    drfPhase = 0, 
    dummyPulses = 1                    # Dummy pulses for T1 stabilization
    ):
    
    # rawData fields
    rawData = {}
    inputs = {}
    outputs = {}
    auxiliar = {}
    
    # Inputs for rawData
    inputs['nScans'] = nScans
    inputs['larmorFreq'] = larmorFreq      # Larmor frequency
    inputs['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    inputs['rfReAmp'] = rfReAmp             # rf refocusing pulse amplitude
    inputs['rfExTime'] = rfExTime          # rf excitation pulse time
    inputs['rfReTime'] = rfReTime            # rf refocusing pulse time
    inputs['echoSpacing'] = echoSpacing        # time between echoes
    inputs['inversionTime'] = inversionTime       # Inversion recovery time
    inputs['repetitionTime'] = repetitionTime     # TR
    inputs['fov'] = fov           # FOV along readout, phase and slice
    inputs['dfov'] = dfov            # Displacement of fov center
    inputs['nPoints'] = nPoints                 # Number of points along readout, phase and slice
    inputs['etl'] = etl                    # Echo train length
    inputs['acqTime'] = acqTime             # Acquisition time
    inputs['axes'] = axes       # 0->x, 1->y and 2->z defined as [rd,ph,sl]
    inputs['axesEnable'] = axesEnable # 1-> Enable, 0-> Disable
    inputs['sweepMode'] = sweepMode               # 0->k2k (T2),  1->02k (T1),  2->k20 (T2), 3->Niquist modulated (T2)
    inputs['phaseGradTime'] = phaseGradTime       # Phase and slice dephasing time
    inputs['rdPreemphasis'] = rdPreemphasis
    inputs['drfPhase'] = drfPhase 
    inputs['dummyPulses'] = dummyPulses                    # Dummy pulses for T1 stabilization
    
    # Miscellaneous
    blkTime = 10             # Deblanking time (us)
    larmorFreq = larmorFreq*1e-6
    gradRiseTime = 100e-6       # Estimated gradient rise time
    gradDelay = 9            # Gradient amplifier delay
    addRdPoints = 10             # Initial rd points to avoid artifact at the begining of rd
    gammaB = 42.56e6            # Gyromagnetic ratio in Hz/T
    deadTime = 200
    oversamplingFactor = 6
    addRdGradTime = 400     # Additional readout gradient time to avoid turn on/off effects on the Rx channel
    randFactor = 2e-3           # Amplitude of the random displacement of phase lines
    
    auxiliar['gradDelay'] = gradDelay*1e-6
    auxiliar['gradRiseTime'] = gradRiseTime
    auxiliar['oversamplingFactor'] = oversamplingFactor
    auxiliar['addRdGradTime'] = addRdGradTime*1e-6
    auxiliar['randFactor'] = randFactor
    
    # Matrix size
    nRD = nPoints[0]+2*addRdPoints
    nPH = nPoints[1]*axesEnable[1]+(1-axesEnable[1])
    nSL = nPoints[2]*axesEnable[2]+(1-axesEnable[2])
    
    # ETL if nPH = 1
    if etl>nPH:
        etl = nPH
    
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
    phGradAmplitude = nPH/(2*gammaB*fov[1]*(phaseGradTime+gradRiseTime))*axesEnable[1]
    slGradAmplitude = nSL/(2*gammaB*fov[2]*(phaseGradTime+gradRiseTime))*axesEnable[2]
    auxiliar['rdGradAmplitude'] = rdGradAmplitude
    auxiliar['phGradAmplitude'] = phGradAmplitude
    auxiliar['slGradAmplitude'] = slGradAmplitude
    
    # Change gradient values to OCRA units
    gFactor = reorganizeGfactor(axes)
    auxiliar['gFactor'] = gFactor
    rdGradAmplitude = rdGradAmplitude/gFactor[0]*1000/10
    phGradAmplitude = phGradAmplitude
    slGradAmplitude = slGradAmplitude
    
    # Phase and slice gradient vector
    phGradients = np.linspace(-phGradAmplitude,phGradAmplitude,num=nPH,endpoint=False)
    slGradients = np.linspace(-slGradAmplitude,slGradAmplitude,num=nSL,endpoint=False)
    
    # Add random displacemnt to phase encoding lines
    for ii in range(nPH):
        if ii<np.ceil(nPH/2-nPH/20) or ii>np.ceil(nPH/2+nPH/20):
            phGradients[ii] = phGradients[ii]+randFactor*np.random.randn()
    auxiliar['phGradients'] = phGradients
    auxiliar['slGradients'] = slGradients
    kPH = gammaB*phGradients*(gradRiseTime+phaseGradTime)
    phGradients = phGradients/gFactor[1]*1000/10
    slGradients = slGradients/gFactor[2]*1000/10
    
    
    # Set phase vector to given sweep mode
    ind = getIndex(phGradients, etl, nPH, sweepMode)
    phGradients = phGradients[::-1]
    phGradients = phGradients[ind]

    # Initialize the experiment
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    samplingPeriod = expt.get_rx_ts()[0]
    BW = 1/samplingPeriod/oversamplingFactor
    acqTime = nPoints[0]/BW        # us
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

    # Gradients
    def gradPulse(tStart, gTime,gAmp, gAxes):
        t = np.array([tStart, tStart+gradRiseTime+gTime])
        for gIndex in range(np.size(gAxes)):
            a = np.array([gAmp[gIndex], 0])
            if gAxes[gIndex]==0:
                expt.add_flodict({'grad_vx': (t, a)})
            elif gAxes[gIndex]==1:
                expt.add_flodict({'grad_vy': (t, a)})
            elif gAxes[gIndex]==2:
                expt.add_flodict({'grad_vz': (t, a)})
    
    def endSequence(tEnd):
        expt.add_flodict({
                'grad_vx': (np.array([tEnd]),np.array([0]) ), 
                'grad_vy': (np.array([tEnd]),np.array([0]) ), 
                'grad_vz': (np.array([tEnd]),np.array([0]) ),
             })

    # Changing time parameters to us
    rfExTime = rfExTime*1e6
    rfReTime = rfReTime*1e6
    echoSpacing = echoSpacing*1e6
    repetitionTime = repetitionTime*1e6
    gradRiseTime = gradRiseTime*1e6
    phaseGradTime = phaseGradTime*1e6
    rdDephTime = rdDephTime*1e6
    inversionTime = inversionTime*1e6
    
    # Create sequence instructions
    phIndex = 0
    slIndex = 0
    scanTime = (nPH*nSL/etl+dummyPulses)*repetitionTime
    for repeIndex in range(int(nPH*nSL/etl)+dummyPulses):
        # Initialize time
        t0 = 20+repetitionTime*repeIndex
        
        # Inversion pulse
        if inversionTime!=0:
            rfPulse(t0,rfReTime,rfReAmp,0)
        
        # Excitation pulse
        t0 += rfReTime/2+inversionTime-rfExTime/2
        rfPulse(t0,rfExTime,rfExAmp,drfPhase*np.pi/180)
    
        # Dephasing readout
        t0 += blkTime+rfExTime-gradDelay
        if repeIndex>=dummyPulses:         # This is to account for dummy pulses
            gradPulse(t0, acqTime+2*addRdGradTime, [rdGradAmplitude/2*rdPreemphasis], [axes[0]])
    
        # First readout to avoid RP initial readout effect
        if repeIndex>=dummyPulses:         # This is to account for dummy pulses
            rxGate(t0+gradDelay+deadTime, acqTime+2*addRdPoints/BW)
        
        # Echo train
        for echoIndex in range(etl):
            # Refocusing pulse
            if echoIndex == 0:
                t0 += (-rfExTime+echoSpacing-rfReTime)/2-blkTime
            else:
                t0 += gradDelay-acqTime/2+echoSpacing/2-rfReTime/2-blkTime-addRdGradTime
            rfPulse(t0, rfReTime, rfReAmp, np.pi/2)

            # Dephasing phase and slice gradients
            t0 += blkTime+rfReTime
            if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                gradPulse(t0, phaseGradTime, [phGradients[phIndex]], [axes[1]])
                gradPulse(t0, phaseGradTime, [slGradients[slIndex]], [axes[2]])
            
            # Readout gradient
            t0 += -rfReTime/2+echoSpacing/2-acqTime/2-gradRiseTime-gradDelay-addRdGradTime
            if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                gradPulse(t0, acqTime+2*addRdGradTime, [rdGradAmplitude], [axes[0]])

            # Rx gate
            t0 += gradDelay+gradRiseTime+addRdGradTime-addRdPoints/BW
            if repeIndex>=dummyPulses:         # This is to account for dummy pulses
                rxGate(t0, acqTime+2*addRdPoints/BW)

            # Rephasing phase and slice gradients
            t0 += addRdPoints/BW+acqTime-gradDelay+addRdGradTime
            if (echoIndex<etl-1 and repeIndex>=dummyPulses):
                gradPulse(t0, phaseGradTime, [-phGradients[phIndex]], [axes[1]])
                gradPulse(t0, phaseGradTime, [-slGradients[slIndex]], [axes[2]])

            # Update the phase and slice gradient
            if repeIndex>=dummyPulses:
                if phIndex == nPH-1:
                    phIndex = 0
                    slIndex += 1
                else:
                    phIndex += 1
            
            if phIndex == nPH-1 and slIndex == nSL-1:
                endSequence(scanTime)
    
    # Plot sequence:
#    expt.plot_sequence()
#    plt.show()
    
    # Run the experiment
    dataFull = []
    for ii in range(nScans):
        rxd, msgs = expt.run()
        rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
        # Get data
        scanData = sig.decimate(rxd['rx0'], oversamplingFactor, ftype='fir', zero_phase=True)
        dataFull = np.concatenate((dataFull, scanData), axis = 0)
    
    # Average data
    data = np.reshape(dataFull, (nScans, int(nRD*(etl+1)*nPH*nSL/etl)))
    data = np.average(data, axis=0)
    
    # Delete the FID measurement
    # Reshape to numberOfRepetitions X nRD*etl
    data = np.reshape(data, (int(nPH*nSL/etl), nRD*(etl+1)))
    # Delete the FID measurement
    data = data[:, nRD:]
    # Reshape to 1 X nRD*nPH*nSL
    data = np.reshape(data, (1, nRD*nPH*nSL))

    # Delete the addRdPoints
    # Reshape to nSL X nPH X nSL
    data = np.reshape(data, (nSL, nPH, nRD))
    # Delete the additional readout points
    data = data[:, :, addRdPoints:addRdPoints+nPoints[0]]
    
    # Reorganize the data acording to sweep mode
    dataTemp = data*0
    for ii in range(nPH):
        dataTemp[:, ind[ii], :] = data[:, ii, :]
    data = np.reshape(dataTemp, (1, nPoints[0]*nPH*nSL))
    
    # Fix the position of the sample according t dfov
    kMax = nPoints/(2*fov)*axesEnable
    kRD = np.linspace(-kMax[0],kMax[0],num=nPoints[0],endpoint=False)
#    kPH = np.linspace(-kMax[1],kMax[1],num=nPH,endpoint=False)
    kSL = np.linspace(-kMax[2],kMax[2],num=nSL,endpoint=False)
    kPH = kPH[::-1]
    kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
    kRD = np.reshape(kRD, (1, nPoints[0]*nPH*nSL))
    kPH = np.reshape(kPH, (1, nPoints[0]*nPH*nSL))
    kSL = np.reshape(kSL, (1, nPoints[0]*nPH*nSL))
    dPhase = np.exp(-2*np.pi*1j*(dfov[0]*kRD+dfov[1]*kPH+dfov[2]*kSL))
    data = data*dPhase
    
    # Save data in rawData
    auxiliar['kMax'] = kMax
    outputs['sampled'] = np.concatenate((kRD, kPH, kSL, data), axis=0)
    
    # Get image with FFT
    data = np.reshape(data, (nSL, nPH, nPoints[0]))
    img=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
    outputs['image'] = img
    
    # Plot data for 1D case
    if (nPH==1 and nSL==1):
        # Plot k-space
        plt.figure(2)
        dataPlot = data[0, 0, :]
        plt.subplot(1, 2, 1)
        plt.plot(kRD[0, :], np.abs(dataPlot))
        plt.yscale('log')
        plt.xlabel('krd (mm^-1)')
        plt.ylabel('Signal (mV)')
        # Plot image
        xAxis = np.linspace(-fov[0]/2*1e2, fov[0]/2*1e2, num=nPoints[0])
        img = img[0, 0, :]
        plt.subplot(122)
        plt.plot(xAxis, np.abs(img))
        plt.xlabel('Position RD (cm)')
        plt.ylabel('Density (a.u.)')
    else:
        # Plot k-space
        plt.figure(2)
        dataPlot = data[round(nSL/2), :, :]
        plt.subplot(121)
        plt.imshow(np.log(np.abs(dataPlot)),cmap='gray')
        plt.axis('off')
        # Plot image
        if sweepMode==3:
            imgPlot = img[round(nSL/2), round(nPH/4):round(3*nPH/4), :]
        else:
            imgPlot = img[round(nSL/2), :, :]
        plt.subplot(122)
        plt.imshow(np.abs(imgPlot), cmap='gray')
        plt.axis('off')
        # Plot image in log scale
#        plt.subplot(133)
#        plt.imshow(np.log(np.abs(img)), cmap='gray')
#        plt.axis('off')
    
    # Delete experiment:
    expt.__del__()
    
    # Plot central line in k-space
#    plt.subplot(133)
#    data = np.abs(data[0, :])
#    plt.plot(data)
#    plt.show()
    
    # Save data
    dt = datetime.now()
    dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
    dt2 = date.today()
    dt2_string = dt2.strftime("%Y.%m.%d")
    if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
        os.makedirs('experiments/acquisitions/%s' % (dt2_string))
            
    if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
        os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
#    rawdata = dict
#    rawdata = {"dataFull":dataFull, "rawdata":data,  "img":img}
    inputs['name'] = "%s.%s.mat" % ("TSE",dt_string)
    rawData['inputs'] = inputs
    rawData['auxiliar'] = auxiliar
    rawData['outputs'] = outputs
    rawdata = {}
    rawdata['rawData'] = rawData
    savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, "TSE",dt_string),  rawdata) 
    
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

    rare_standalone()
