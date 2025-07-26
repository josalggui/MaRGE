"""
@author: José Miguel Algarín Guisado
MRILAB @ I3M
"""

import os
import sys
#*****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory = os.path.dirname(parent_directory)

# Define the subdirectories you want to add to sys.path
subdirs = ['MaRGE', 'marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
#******************************************************************************
import marge.controller.experiment_gui as ex
import numpy as np
import marge.seq.mriBlankSeq as blankSeq
import scipy.signal as sig
import marge.configs.hw_config as hw
import marge.configs.units as units

class FID(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(FID, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='FIDinfo', val='FID')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
        self.addParameter(key='deadTime', string='RF dead time (us)', val=400.0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='SEQ')
        self.addParameter(key='nPoints', string='Number of points', val=100, field='IM')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='shimmingTime', string='Shimming time (ms)', val=1, field='OTH')
        self.addParameter(key='readRFpulse', string='Read RF Pulse', val=0, field='OTH')

    def sequenceInfo(self):
        
        print("FID")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs a single FID\n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        return(repetitionTime*nScans/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False  # Starts the gpa

        # Create input parameters
        nScans = self.mapVals['nScans']
        larmorFreq = self.mapVals['larmorFreq'] # MHz
        rfExAmp = self.mapVals['rfExAmp']
        rfExTime = self.mapVals['rfExTime'] # us
        deadTime = self.mapVals['deadTime'] # us
        repetitionTime = self.mapVals['repetitionTime']*1e3 # us
        acqTime = self.mapVals['acqTime']*1e3 # us
        nPoints = self.mapVals['nPoints']
        shimming = np.array(self.mapVals['shimming'])*units.sh
        txChannel = self.mapVals['txChannel']
        rxChannel = self.mapVals['rxChannel']
        shimmingTime = self.mapVals['shimmingTime']*1e3 # us

        # Miscellaneus
        bw = nPoints/acqTime # MHz

        def createSequence():
            # Shimming
            self.iniSequence(20, shimming)  # shimming is turned on 20 us after experiment beginning

            for scan in range(nScans):
                tEx = shimmingTime + repetitionTime*scan + hw.blkTime + rfExTime / 2

                # Excitation pulse
                t0 = tEx - hw.blkTime - rfExTime / 2
                self.rfRecPulse(t0, rfExTime, rfExAmp, 0, channel=txChannel)

                # Rx gate
                if self.readRFpulse == 0:
                    t0 = tEx + rfExTime / 2 + deadTime
                elif self.readRFpulse == 1:
                    t0 = tEx-rfExTime/2
                self.rxGateSync(t0, acqTime, channel=rxChannel)
                # self.ttl(t0, acqTime, channel=1, rewrite=True)

            self.endSequence(shimmingTime + repetitionTime*nScans)


        # Initialize the experiment
        samplingPeriod = 1 / bw  # us
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
        samplingPeriod = self.expt.getSamplingRate()
        bw = 1 / samplingPeriod
        acqTime = nPoints / bw  # us
        self.mapVals['acqTime'] = acqTime*1e-3 # ms
        self.mapVals['bw'] = bw # MHz
        createSequence()
        if self.floDict2Exp():
            print("Sequence waveforms loaded successfully")
            pass
        else:
            print("ERROR: sequence waveforms out of hardware bounds")
            return False

        if not plotSeq:
            # Run the experiment and get data
            rxd, msgs = self.expt.run()

            # Decimate the signal
            dataFull = self.decimate(rxd['rx%i' % rxChannel], nScans)

            # Average data
            data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
            self.mapVals['data'] = data

            # Save data to sweep plot (single point)
            self.mapVals['sampledPoint'] = data[0]

        self.expt.__del__()

        return True

    def sequenceAnalysis(self, obj=''):
        def getFHWM(s,f_vector,bw):
            target = np.max(s) / 2
            p0 = np.argmax(s)
            f0 = f_vector[p0]
            s1 = np.abs(s[0:p0]-target)
            f1 = f_vector[np.argmin(s1)]
            s2 = np.abs(s[p0::]-target)
            f2 = f_vector[np.argmin(s2)+p0]
            return f2-f1


        # Signal and spectrum from 'fir' and decimation
        signal = self.mapVals['data']
        bw = self.mapVals['bw']*1e3 # kHz
        nPoints = self.mapVals['nPoints']
        deadTime = self.mapVals['deadTime']*1e-3 # ms
        rfExTime = self.mapVals['rfExTime']*1e-3 # ms
        tVector = np.linspace(rfExTime/2 + deadTime + 0.5/bw, rfExTime/2 + deadTime + (nPoints-0.5)/bw, nPoints)
        fVector = np.linspace(-bw/2, bw/2, nPoints)
        spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal))))
        fitedLarmor=self.mapVals['larmorFreq'] + fVector[np.argmax(np.abs(spectrum))] * 1e-3  #MHz
        hw.larmorFreq=fitedLarmor
        fwhm=getFHWM(spectrum, fVector, bw)
        dB0=fwhm*1e6/hw.larmorFreq

        for sequence in self.sequence_list.values():
            if 'larmorFreq' in sequence.mapVals:
                sequence.mapVals['larmorFreq'] = hw.larmorFreq

        # Get the central frequency
        print('Larmor frequency: %1.5f MHz' % fitedLarmor)
        print('FHWM: %1.5f kHz' % fwhm)
        print('dB0/B0: %1.5f ppm' % dB0)

        self.mapVals['signalVStime'] = [tVector, signal]
        self.mapVals['spectrum'] = [fVector, spectrum]

        # Add time signal to the layout
        result1 = {'widget': 'curve',
                   'xData': tVector,
                   'yData': [np.abs(signal), np.real(signal), np.imag(signal)],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Signal vs time',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 0,
                   'col': 0}

        # Add frequency spectrum to the layout
        result2 = {'widget': 'curve',
                   'xData': fVector,
                   'yData': [spectrum],
                   'xLabel': 'Frequency (kHz)',
                   'yLabel': 'Spectrum amplitude (a.u.)',
                   'title': 'Spectrum',
                   'legend': [''],
                   'row': 1,
                   'col': 0}

        # create self.out to run in iterative mode
        self.output = [result1, result2]
        self.saveRawData()

        return self.output


if __name__=='__main__':
    seq = FID()
    seq.sequenceRun()
    seq.sequenceAnalysis(obj='Standalone')
