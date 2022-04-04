"""
@author: Teresa Guallart Naval
MRILAB @ I3M

@modified: J.M. Algarín, MRILab, i3M, CSIC, Valencia, Spain
@date: 02 Sat Apr 2 2022
"""

import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as sig
from datetime import date,  datetime 
import os
from scipy.io import savemat
from mrilabMethods.mrilabMethods import *


def rabiflopStandalone(
    init_gpa= False,                 
    larmorFreq=3.07436,
    rfExAmp=0.058, 
    rfExPhase = 0,
    rfExTimeIni=250, 
    rfExTimeEnd = 2500, 
    nExTime =20, 
    nReadout =100,
    tAdq = 4*1e3,
    tEcho = 40*1e3,
    tRepetition = 3000*1e3,
    plotSeq = 0, 
    pulseShape = 'Sinc',  # 'Rec' for square pulse shape, 'Sinc' for sinc pulse shape
    method='Amp', # 'Amp' -> rfReAmp=2*rfExAmp, 'Time' -> rfReTime=2*rfReTime
    shimming=[-80, -100, 10]):

    # Miscellaneous
    deadTime = 400 # us, time between excitation and first acquisition
    oversamplingFactor=6
    shimming=np.array(shimming)*1e-4
    
    # Inputs for rawData
    rawData={}
    rawData['larmorFreq'] = larmorFreq      # Larmor frequency
    rawData['rfExAmp'] = rfExAmp             # rf excitation pulse amplitude
    rawData['rfExTimeIni'] = rfExTimeIni          # rf excitation pulse time
    rawData['rfExTimeEnd'] = rfExTimeEnd           # rf refocusing pulse time
    rawData['nExTime'] = nExTime
    rawData['echoSpacing'] = tEcho        # time between echoes
    rawData['repetitionTime'] = tRepetition     # TR
    rawData['nReadout'] = nReadout
    rawData['tAdq'] = tAdq
    rawData['pulseShape'] = pulseShape
    rawData['shimming'] = shimming
    rawData['deadTime'] = deadTime*1e-6
    
    # Excitation times
    rfExTime= np.linspace(rfExTimeIni, rfExTimeEnd, nExTime,  endpoint=True)
    
    # Refocusing amplitude and time
    if method=='Time':
        rfReAmp = rfExAmp
        rfReTime = 2*rfExTime
    elif method=='Amp':
        rfReAmp = 2*rfExAmp
        rfReTime = rfExTime
    rawData['rfReAmp'] = rfReAmp
    rawData['rfReTime'] = rfReTime

    #  SPECIFIC FUNCTIONS   ####################################################################################
    def  plotData(data, rfExTime, tAdqReal):
       plt.figure(1)
       colors = cm.rainbow(np.linspace(0, 0.8, len(rfExTime)))
       for indexExTime in range(nExTime):
            tPlot = np.linspace(-tAdqReal/2, tAdqReal/2, nReadout,  endpoint ='True')*1e-3
            leg = 'Time = '+ str(np.round(rfExTime[indexExTime]))+ 'us'
            plt.plot(tPlot[5:], np.abs(data[indexExTime, 5:]),  label = leg, color=colors[indexExTime])
#            plt.plot(tPlot[5:], np.real(data[indexExTime, 5:]))
#            plt.plot(tPlot[5:], np.imag(data[indexExTime, 5:]))
       plt.xlabel('t(ms)')
       plt.ylabel('A(mV)')
       plt.legend()
    
    def  getRabiFlopData(data, rfExTime, tAdqReal):
       for indexExTime in range(nExTime):
            if indexExTime == 0:
                maxEchoes = np.max(np.abs(data[indexExTime,5:]))
            else:
                maxEchoes=np.append(maxEchoes,np.max(np.abs(data[indexExTime, 5:])))
       rabiFlopData= np.transpose(np.array([rfExTime, maxEchoes]))
       return rabiFlopData 
    
    def  plotRabiFlop(rabiFlopData,  name):
       plt.figure(2)
       plt.plot(rabiFlopData[:, 0], rabiFlopData[:, 1])
       plt.xlabel('t(us)')
       plt.ylabel('A(mV)')
       titleRF= 'RF Amp = '+ str(np.real(rfExAmp)) + ';  ' + name
       plt.title(titleRF)
       return rabiFlopData 
 
    def saveMyData(rawData):
        # Save data
        dt = datetime.now()
        dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
        dt2 = date.today()
        dt2_string = dt2.strftime("%Y.%m.%d")
        if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('experiments/acquisitions/%s' % (dt2_string))
        if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
        rawData['name'] = "%s.%s.mat" % ("RABIFLOP",dt_string)
        savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, "RABIFLOP",dt_string),  rawData)
        return rawData['name']


    #  SEQUENCE  ############################################################################################
    def createSequence():
        # Set shimming
        iniSequence(expt, 20, shimming)
        
        # Initialize time
        tEx = 20e3
            
        # Excitation pulse
        t0 = tEx-blkTime-rfExTime[indexExTime]/2
        if pulseShape=='Rec':
            rfRecPulse(expt, t0, rfExTime[indexExTime], rfExAmp, rfExPhase*np.pi/180)
        elif pulseShape=='Sinc':
            rfSincPulse(expt, t0, rfExTime[indexExTime], 7, rfExAmp, rfExPhase*np.pi/180)
        
        # First acquisition
        t0 = tEx+rfExTime[indexExTime]/2+deadTime
        rxGate(expt, t0, tAcq)
        
        # Refocusing pulse
        t0 = tEx+tEcho/2-rfReTime[indexExTime]/2-blkTime
        if pulseShape=='Rec':
            rfRecPulse(expt, t0, rfReTime[indexExTime], rfReAmp, np.pi/2)
        elif pulseShape=='Sinc':
            rfSincPulse(expt, t0, rfReTime[indexExTime], 7, rfReAmp, np.pi/2)
        
        # Second acquisition
        t0 = tEx+tEcho-tAcq/2
        rxGate(expt, t0, tAcq)
        
        # End sequence
        endSequence(expt, tRepetition)
        
        
    # INIT EXPERIMENT
    dataAll = []
    for indexExTime in range(nExTime):
        BW = nReadout/tAdq
        BWov = BW*oversamplingFactor
        samplingPeriod = 1/BWov
        expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = expt.get_rx_ts()[0]
        BW = 1/samplingPeriod/oversamplingFactor
        tAcq = nReadout/BW 
        rawData['bw'] = BW
        createSequence()
        if plotSeq == 0:
            print(indexExTime,  '.- Running...')
            rxd, msgs = expt.run()
            expt.__del__()
            print('   End')
            data = sig.decimate(rxd['rx0']*13.788, oversamplingFactor, ftype='fir', zero_phase=True)
            dataAll = np.concatenate((dataAll, data), axis=0)
            rawData['dataFull'] = dataAll
        elif plotSeq == 1:
            expt.plot_sequence()
            plt.show()
            expt.__del__()
   
    if plotSeq == 1:
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq == 0:
        data = np.reshape(dataAll,  (nExTime, 2, nReadout))
        dataFID = np.squeeze(data[:, 0, :])
        dataEcho = np.squeeze(data[:, 1, :])
        plotData(dataEcho, rfExTime, tAcq)
        rabiFlopData = getRabiFlopData(dataEcho, rfExTime, tAcq)
        rawData['rabiFlopData'] = rabiFlopData
        name = saveMyData(rawData)
        plotRabiFlop(rabiFlopData,  name)
        plt.show()

#  MAIN  ######################################################################################################
if __name__ == "__main__":
    rabiflopStandalone()
