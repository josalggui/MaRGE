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
import controller.experiment_gui as ex
import numpy as np
import seq.mriBlankSeq as blankSeq
import scipy.signal as sig
import configs.hw_config as hw
import configs.units as units

class SteadyState(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(SteadyState, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='SteadyStateinfo', val='SteadyState')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='RF')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.08, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF')
        self.addParameter(key='deadTime', string='RF dead time (us)', val=400.0, field='RF')
        self.addParameter(key='Npulses', string='Number of pulses', val=10, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='SEQ')
        self.addParameter(key='nPoints', string='Number of points', val=100, field='IM')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='shimmingTime', string='Shimming time (ms)', val=1, field='OTH')

    def sequenceInfo(self):
        print("SteadyState")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs a single SteadyState\n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        nPulses = self.mapVals['Npulses']
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        return(repetitionTime*nScans*nPulses/60)  # minutes, scanTime

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
        Npulses = self.mapVals['Npulses']

        # Miscellaneus
        bw = nPoints/acqTime # MHz

        def createSequence():
            # Shimming
            self.iniSequence(20, shimming)  # shimming is turned on 20 us after experiment beginning

            for scan in range(nScans*Npulses):
                tEx = shimmingTime + repetitionTime*scan + hw.blkTime + rfExTime / 2

                # Excitation pulse
                t0 = tEx - hw.blkTime - rfExTime / 2
                self.rfRecPulse(t0, rfExTime, rfExAmp, 0, channel=txChannel)

                # Rx gate
                t0 = tEx + rfExTime / 2 + deadTime
                self.rxGateSync(t0, acqTime, channel=rxChannel)

            self.endSequence(shimmingTime + repetitionTime*nScans*Npulses)


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
            dataFull = self.decimate(rxd['rx%i' % rxChannel], nScans*Npulses)

            # Average data
            data = np.average(np.reshape(dataFull, (nScans, -1)), axis=0)
            self.mapVals['data'] = data

        self.expt.__del__()

        return True

    def sequenceAnalysis(self, obj=''):


        # Signal and spectrum from 'fir' and decimation
        signal = self.mapVals['data']
        nPoints = self.mapVals['nPoints']
        Npulses = self.mapVals['Npulses']
        Tacq = self.mapVals['acqTime']
        tVector = np.linspace(0, Tacq*Npulses, signal.shape[0])
        Mss = np.abs(signal[::nPoints])/np.abs(signal[0])
        Nindex = np.linspace(1, Npulses, Mss.shape[0])

        self.mapVals['signalVStime'] = [tVector, signal]
        self.mapVals['MssVStime'] = [Nindex, signal]

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
                   'xData': Nindex,
                   'yData': [np.abs(Mss)],
                   'xLabel': 'Repetition index',
                   'yLabel': 'Stationary magnetization',
                   'title': 'Transition to steady state',
                   'legend': [''],
                   'row': 1,
                   'col': 0}

        # create self.out to run in iterative mode
        self.output = [result1, result2]
        self.saveRawData()

        return self.output


if __name__=='__main__':
    seq = SteadyState()
    seq.sequenceRun()
    seq.sequenceAnalysis(obj='Standalone')
