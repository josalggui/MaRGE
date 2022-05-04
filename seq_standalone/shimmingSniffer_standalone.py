"""
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia, Spain
@date: 19 tue Apr 2022
@email: josalggui@i3m.upv.es
"""

import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import configs.hw_config as hw # Import the scanner hardware config
import mrilabMethods.mrilabMethods as mri


def rabiflopStandalone(
    init_gpa= False,                 
    larmorFreq = 3.0806, # MHz
    rfExAmp = 0.4, # a.u.
    rfReAmp = 0.8, # a.u.
    rfExTime = 22, # us
    rfReTime = 22, # us 
    nPoints = 100,
    acqTime = 18, # ms
    echoTime = 20, # ms
    repetitionTime = 1000, # ms
    plotSeq = 0, # 0 to run sequence, 1 to plot sequence
    pulseShape = 'Rec',  # 'Rec' for square pulse shape, 'Sinc' for sinc pulse shape
    shimming0 = [-0, -0, 0], 
    nShimming = 10, # number of samples to sweep in each gradient direction
    dShimming = [10, 10, 10] # shimming step in each direction
    ):
    
    freqCal = 1
    plt.ion()

    # Miscellaneous
    deadTime = 400 # us, time between excitation and first acquisition
    shimming0 = np.array(shimming0)*1e-4
    dShimming = np.array(dShimming)*1e-4
    
    # Varibales to fundamental units
    larmorFreq *= 1e6
    rfExTime *= 1e-6
    rfReTime *= 1e-6
    acqTime *= 1e-3
    echoTime *= 1e-3
    repetitionTime *= 1e-3
    
    # Inputs for rawData
    rawData={}
    rawData['seqName'] = 'ShimmingCal'
    rawData['larmorFreq'] = larmorFreq      # Larmor frequency
    rawData['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    rawData['rfReAmp'] = rfReAmp
    rawData['rfExTime'] = rfExTime
    rawData['rfReTime'] = rfReTime
    rawData['nPoints'] = nPoints
    rawData['acqTime'] = acqTime
    rawData['echoTime'] = echoTime
    rawData['repetitionTime'] = repetitionTime
    rawData['pulseShape'] = pulseShape
    rawData['deadTime'] = deadTime*1e-6
    rawData['shimming0'] = shimming0
    rawData['nShimming'] = nShimming
    rawData['dShimming'] = dShimming
    rawData['shimming'] = shimming0
    rawData['addRdPoints'] = 10
    
    # Bandwidth 
    bw = nPoints/acqTime*1e-6 # MHz
    bwov = bw*hw.oversamplingFactor # MHz
    samplingPeriod = 1/bwov # us
    rawData['bw'] = bw
    rawData['samplingPeriod'] = samplingPeriod
    fVector = np.linspace(-bw/2, bw/2, num=nPoints, endpoint=False)
    
    # Time variables in us and MHz
    larmorFreq *=1e-6
    rfExTime *=1e6
    rfReTime *=1e6
    echoTime *=1e6
    repetitionTime *=1e6
    acqTime *=1e6
    
    #  SEQUENCE  ############################################################################################
    def createSequence(expt, shimming):
        # Set shimming
        mri.iniSequence(expt, 20, shimming)
        
        # Initialize time
        tEx = 20e3
            
        # Excitation pulse
        t0 = tEx-hw.blkTime-rfExTime/2
        if pulseShape=='Rec':
            mri.rfRecPulse(expt, t0, rfExTime, rfExAmp, 0)
        elif pulseShape=='Sinc':
            mri.rfSincPulse(expt, t0, rfExTime, 7, rfExAmp, 0)
        
        # Refocusing pulse
        t0 = tEx+echoTime/2-rfReTime/2-hw.blkTime
        if pulseShape=='Rec':
            mri.rfRecPulse(expt, t0, rfReTime, rfReAmp, np.pi/2)
        elif pulseShape=='Sinc':
            mri.rfSincPulse(expt, t0, rfReTime, 7, rfReAmp, np.pi/2)
        
        # Acquisition window
        t0 = tEx+echoTime-acqTime/2
        mri.rxGate(expt, t0, acqTime)
        
        # End sequence
        mri.endSequence(expt, repetitionTime)
        
        
    
    
    def runExperiment(shimmingExp, samplingRate):
        expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingRate, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = expt.get_rx_ts()[0]
        bw = 1/samplingPeriod/hw.oversamplingFactor
        acqTime = nPoints/bw
        rawData['acqTime'] = acqTime
        rawData['bw'] = bw
        createSequence(expt, shimmingExp)
        print(shimmingExp*1e4,  '.- Running...')
        rxd, msgs = expt.run()
        expt.__del__()
        data = sig.decimate(rxd['rx0']*13.788, hw.oversamplingFactor, ftype='fir', zero_phase=True)
        return(data)
    
    def sweepAxis(axis):
        # First two samples to determine the sweep direction
        shimmingA = shimming0*1
        shimmingB = shimming0*1
        shimmingB[axis] = shimming0[axis]+dShimming[axis]
        dataA = runExperiment(shimmingA, samplingPeriod)
        fwhmA, maxA = getPeakProperties(fVector, dataA)
        print('Peak amplitude = ', maxA)
        print('Homogeneity = ', fwhmA/larmorFreq*1e6, ' ppm \n')
        dataB = runExperiment(shimmingB, samplingPeriod)
        fwhmB, maxB = getPeakProperties(fVector, dataB)
        print('Peak amplitude = ', maxB)
        print('Homogeneity = ', fwhmB/larmorFreq*1e6, ' ppm \n')
        if maxB<maxA:
            dShimming[axis] = -dShimming[axis]
            maxP = maxB
            maxB = maxA
            maxA = maxP
            shimmingP = shimmingB
            shimmingB = shimmingA
            shimmingA = shimmingP
            del maxP,  shimmingP
        
        # Sweep axis until the max start to goes down
        while maxB>maxA:
            fwhmA = fwhmB
            maxA = maxB
            shimmingA = shimmingB*1
            shimmingB[axis] += dShimming[axis]
            dataB = runExperiment(shimmingB, samplingPeriod)
            fwhmB, maxB = getPeakProperties(fVector, dataB)
            print('Peak amplitude = ', maxB)
            print('Homogeneity = ', fwhmB/larmorFreq*1e6, ' ppm \n˘')
        
        print('Axis ', axis, ' optimized!')
        
        # Return fwhm and max values
        return(fwhmA, maxA, shimmingA)
        
    
    # Calibrate frequency
    if freqCal==1: 
        mri.freqCalibration(rawData, bw=0.05)
        mri.freqCalibration(rawData, bw=0.005)
        larmorFreq = rawData['larmorFreq']*1e-6
    
    for nSweep in range(3):
        for axis in range(3):
            fwhmOpt,  maxOpt, shimmingOpt = sweepAxis(axis)
            shimming0 = shimmingOpt
        dShimming /= 2
    
    # Save data
    mri.saveRawData(rawData)
#    plt.figure(3)
#    plt.plot(sxVector*1e4, gx, 'b.', 
#        syVector*1e4, gy, 'g.', 
#        szVector*1e4, gz, 'r.')
#    plt.title(rawData['fileName'])
#    plt.xlabel('Shimming (a.u.)')
#    plt.ylabel('FFT peak amplitude (a.u.)')
#    plt.legend(['x Axis', 'y Axis', 'z Axis'])
    
#    plt.figure(4)
#    plt.plot(sxVector*1e4, ppmx, 'b.', 
#        syVector*1e4, ppmy, 'g.', 
#        szVector*1e4, ppmz, 'r.')
#    plt.title(rawData['fileName'])
#    plt.xlabel('Shimming (a.u.)')
#    plt.ylabel('Homogeneity (ppm)')
#    plt.legend(['x Axis', 'y Axis', 'z Axis'])
    
    print('Best shimming = ', shimming0*1e4)
    
    plt.show(block=True)

def getPeakProperties(fVector, data):
    fft = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data))))
    nPoints = np.size(fft)
    max = np.max(fft)
    idxMax = np.argmax(fft)
    fft2 = np.abs(fft-max/2)
    idxA = np.argmin(fft2[0:idxMax])
    idxB = np.argmin(fft2[idxMax:nPoints])
    fA = fVector[idxA]
    fB = fVector[idxMax+idxB]
    fwhm = np.abs(fB-fA)
        
    return(fwhm, max)

#  MAIN  ######################################################################################################
if __name__ == "__main__":
    rabiflopStandalone()
