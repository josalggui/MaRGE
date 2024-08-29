"""
@author: José Miguel Algarín Guisado
@date: 2024 May 31
mrilab @ i3M, CSIC, Spain
"""

import os
import sys

# To work with pypulseq
import pypulseq as pp
from flocra_pulseq.interpreter import PSInterpreter

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
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import configs.units as units
import scipy.signal as sig
import experiment as ex
import configs.hw_config as hw



class PulseqReader(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(PulseqReader, self).__init__()
        # Input the parameters
        self.output = None
        self.nScans = None
        self.shimming = None
        self.expt = None
        self.larmorFreq = None
        self.file_path = None
        self.addParameter(key='seqName', string='PulseqReader', val='PulseqReader')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.066, units=units.MHz, field='IM')
        self.addParameter(key='shimming', string='Shimming', val=[0, 0, 0], field='IM', units=units.sh)
        self.addParameter(key='file_path', string='File Path', val="sequence.seq", field='IM', tip='Write the path to your .seq file')

    def sequenceInfo(self):
        
        print("Pulseq Reader")
        print("Author: PhD. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Read a .seq file and run the sequence\n")
        

    def sequenceTime(self):
        return 0  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        init_gpa = False
        self.demo = demo

        # Function to get the dwell time
        def get_seq_info(file_path):
            dwell_time = None
            with open(file_path, 'r') as file:
                lines = file.readlines()
                adc_section = False

                for line in lines:
                    if line.strip() == '[ADC]':
                        adc_section = True
                        continue

                    if adc_section:
                        # If we reach another section, stop processing ADC section
                        if line.startswith('\n'):
                            break

                        # Split the line into components
                        components = line.split()

                        # Check if the line contains the ADC event data
                        if len(components) >= 4:
                            n_readouts = int(components[1])  # Extract the number of acquired points per Rx window
                            dwell_time = int(components[2])  # Extract the dwell time (3rd component)

            return n_readouts, dwell_time

        # Get the dwell time
        n_readouts, dwell = get_seq_info(self.file_path)  # dwell is in ns

        # Create experiment
        if not self.demo:
            self.expt = ex.Experiment(lo_freq=self.larmorFreq * 1e-6,  # MHz
                                      rx_t=dwell * 1e-3,  # us
                                      init_gpa=init_gpa,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                      )
            dwell = self.expt.get_rx_ts()[0]
        bw = 1/dwell * 1e9  # Hz
        self.mapVals['samplingPeriod'] = dwell * 1e-9  # s
        self.mapVals['bw'] = bw  # Hz

        # Run the interpreter to get the waveforms
        waveforms, param_dict = self.flo_interpreter.interpret('sequence.seq')

        # Get number of Rx windows
        n_rx_windows = int(np.sum(waveforms['rx0_en'][1][:]))

        # Convert waveform to mriBlankSeq tools (just do it)
        self.pypulseq2mriblankseq(waveforms=waveforms, shimming=self.shimming)

        if not self.demo:
            if self.floDict2Exp():
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("ERROR: sequence waveforms out of hardware bounds")
                return False

        # Run the experiment
        data_over = []  # To save oversampled data
        if not plotSeq:
            for scan in range(self.nScans):
                print("Scan %i running..." % (scan + 1))
                if not self.demo:
                    rxd, msgs = self.expt.run()
                    rxd['rx0'] = hw.adcFactor * (np.real(rxd['rx0']) - 1j * np.imag(rxd['rx0']))
                else:
                    rxd = {'rx0': np.random.randn(n_readouts * n_rx_windows) +
                                  1j * np.random.randn(n_readouts * n_rx_windows)}
                data_over = np.concatenate((data_over, rxd['rx0']), axis=0)
                print("Acquired points = %i" % np.size([rxd['rx0']]))
                print("Expected points = %i" % n_readouts * n_rx_windows)
                print("Scan %i ready!" % (scan + 1))
                self.mapVals['data_over'] = data_over
        elif plotSeq and standalone:
            self.sequencePlot(standalone=standalone)
            return True

        # Close the experiment
        if not self.demo:
            self.expt.__del__()

        # Add waveforms to the experiment
        if not self.demo:
            if self.floDict2Exp():
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("ERROR: sequence waveforms out of hardware bounds")
                return False

        return True

    def sequenceAnalysis(self, mode=None):
        # Decimate the data
        data_over = self.mapVals['data_over']
        data = sig.decimate(data_over, hw.oversamplingFactor, ftype='fir', zero_phase=True)
        self.mapVals['data'] = data

        # create self.out to run in iterative mode
        self.output = []

        # save data once self.output is created
        self.saveRawData()

        return self.output

if __name__ == '__main__':
    seq = PulseqReader()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=True, demo=True, standalone=True)
    # seq.sequenceAnalysis(mode='Standalone')




