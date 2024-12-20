"""
@author: José Miguel Algarín Guisado
@modifield: T. Guallart Naval, february 28th 2022
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
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import scipy.signal as sig
import configs.hw_config as hw
import configs.units as units
import experiment as ex
from marga_pulseq.interpreter import PSInterpreter
import pypulseq as pp


class LarmorPyPulseq(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(LarmorPyPulseq, self).__init__()
        # Input the parameters
        self.dF = None
        self.bw = None
        self.demo = None
        self.expt = None
        self.larmorFreq = None
        self.repetitionTime = None
        self.rfReTime = None
        self.rfReFA = None
        self.rfExFA = None
        self.rfExTime = None
        self.nScans = None
        self.addParameter(key='seqName', string='LarmorInfo', val='Larmor PyPulseq')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.066, units=units.MHz, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90.0, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180.0, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, field='RF', units=units.us)
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=60.0, field='RF', units=units.us)
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., field='SEQ', units=units.ms)
        self.addParameter(key='bw', string='Bandwidth (kHz)', val=50, field='RF', units=units.kHz)
        self.addParameter(key='dF', string='Frequency resolution (Hz)', val=100, field='RF')
        self.addParameter(key='shimming', string='Shimming', val=[-12.5, -12.5, 7.5], field='OTH', units=units.sh)

    def sequenceInfo(self):
        
        print("Larmor")
        print("Author: PhD. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs a single spin echo to find larmor\n")
        

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        return (repetitionTime * nScans / 60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        init_gpa = False  # Starts the gpa
        self.demo = demo

        # Define the interpreter. It should be updated on calibration
        self.flo_interpreter = PSInterpreter(tx_warmup=hw.blkTime,  # us
                                             rf_center=hw.larmorFreq * 1e6,  # Hz
                                             rf_amp_max=hw.b1Efficiency / (2 * np.pi) * 1e6,  # Hz
                                             gx_max=hw.gFactor[0] * hw.gammaB,  # Hz/m
                                             gy_max=hw.gFactor[1] * hw.gammaB,  # Hz/m
                                             gz_max=hw.gFactor[2] * hw.gammaB,  # Hz/m
                                             grad_max=np.max(hw.gFactor) * hw.gammaB,  # Hz/m
                                             )

        # Define system properties according to hw_config file
        self.system = pp.Opts(
            rf_dead_time=hw.blkTime * 1e-6,  # s
            max_grad=hw.max_grad,  # mT/m
            grad_unit='mT/m',
            max_slew=hw.max_slew_rate,  # mT/m/ms
            slew_unit='mT/m/ms',
            grad_raster_time=hw.grad_raster_time,  # s
            rise_time=hw.grad_rise_time,  # s
        )

        # Set the refocusing time in to twice the excitation time
        if self.rfReTime == 0:
            self.rfReTime = 2 * self.rfExTime

        # Calculate acq_time and echo_time
        n_points = int(self.bw / self.dF)
        acq_time = 1 / self.dF  # s
        echo_time = 2 * acq_time  # s
        self.mapVals['nPoints'] = n_points
        self.mapVals['acqTime'] = acq_time
        self.mapVals['echoTime'] = echo_time

        # Initialize the experiment
        self.bw = n_points / acq_time * hw.oversamplingFactor  # Hz
        sampling_period = 1 / self.bw  # s
        self.mapVals['samplingPeriod'] = sampling_period
        if not self.demo:
            self.expt = ex.Experiment(lo_freq=self.larmorFreq * 1e-6,  # MHz
                                      rx_t=sampling_period * 1e6,  # us
                                      init_gpa=init_gpa,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                      auto_leds=True
                                      )
            sampling_period = self.expt.get_rx_ts()[0]  # us
            self.bw = 1 / sampling_period / hw.oversamplingFactor  # MHz
            acq_time = n_points / self.bw * 1e-6  # s
            self.mapVals['bw_true'] = self.bw * 1e3  # kHz
        else:
            sampling_period = sampling_period * 1e6  # us
            self.bw = 1 / sampling_period / hw.oversamplingFactor  # MHz
            acq_time = n_points / self.bw * 1e-6  # s

        # Create excitation rf event
        delay_rf_ex = self.repetitionTime-acq_time/2-echo_time-self.rfExTime/2-hw.blkTime*1e-6
        event_rf_ex = pp.make_block_pulse(flip_angle=self.rfExFA * np.pi / 180,
                                          duration=self.rfExTime,
                                          delay=delay_rf_ex,
                                          system=self.system,
                                          use="excitation",)

        # Create refocusing rf event
        delay_rf_re = echo_time/2-self.rfExTime/2-self.rfReTime/2-hw.blkTime*1e-6
        event_rf_re = pp.make_block_pulse(flip_angle=self.rfReFA * np.pi / 180,
                                          duration=self.rfReTime,
                                          delay=delay_rf_re,
                                          system=self.system,
                                          use="refocusing")

        # Create ADC event
        delay_adc = echo_time/2-self.rfReTime/2-acq_time/2
        event_adc = pp.make_adc(num_samples=n_points*hw.oversamplingFactor,
                                duration=acq_time,  # s
                                delay=delay_adc,  # s
                                system=self.system)

        # Create the sequence here
        def createSequence():  # Here I will test pypulseq
            rd_points = 0

            # # Shimming
            # self.iniSequence(t0, self.shimming)

            # Add excitatoin
            self.seq.add_block(event_rf_ex)

            # Add refocusing
            self.seq.add_block(event_rf_re)

            # Add ADC
            self.seq.add_block(event_adc)
            rd_points += n_points

            # Create the sequence file
            self.seq.write('sequence.seq')

            # Run the interpreter to get the waveforms
            waveforms, param_dict = self.flo_interpreter.interpret('sequence.seq')

            # Convert waveform to mriBlankSeq tools (just do it)
            self.pypulseq2mriblankseq(waveforms=waveforms, shimming=self.shimming)

            return rd_points

        # Create the sequence and add instructions to the experiment
        acq_points = createSequence()
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
                else:
                    rxd = {'rx0': np.random.randn(acq_points * hw.oversamplingFactor) +
                                  1j * np.random.randn(acq_points * hw.oversamplingFactor)}
                data_over = np.concatenate((data_over, rxd['rx0']), axis=0)
                print("Acquired points = %i" % np.size([rxd['rx0']]))
                print("Expected points = %i" % ((acq_points + 2 * hw.addRdPoints) * hw.oversamplingFactor))
                print("Scan %i ready!" % (scan + 1))
        elif plotSeq and standalone:
            self.sequencePlot(standalone=standalone)
            return True

        # Close the experiment
        if not self.demo:
            self.expt.__del__()

        # Process data to be plotted
        if not plotSeq:
            data_full = sig.decimate(data_over, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['data_full'] = data_full
            data = np.average(np.reshape(data_full, (self.nScans, -1)), axis=0)
            self.mapVals['data'] = data

            # Data to sweep sequence
            self.mapVals['sampledPoint'] = data[int(n_points / 2)]

        return True

    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        # Load data
        signal = self.mapVals['data']
        acq_time = self.mapVals['acqTime'] * 1e3  # ms
        n_points = self.mapVals['nPoints']

        # Generate time and frequency vectors and calcualte the signal spectrum
        tVector = np.linspace(-acq_time / 2, acq_time / 2, n_points)
        fVector = np.linspace(-self.bw / 2, self.bw / 2, n_points) * 1e3  # kHz
        spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal)))

        # Get the central frequency
        idf = np.argmax(np.abs(spectrum))
        fCentral = fVector[idf] * 1e-3  # MHz
        hw.larmorFreq = self.mapVals['larmorFreq'] + fCentral
        print('Larmor frequency: %1.5f MHz' % hw.larmorFreq)
        self.mapVals['larmorFreq'] = hw.larmorFreq
        self.mapVals['signalVStime'] = [tVector, signal]
        self.mapVals['spectrum'] = [fVector, spectrum]

        if mode != 'Standalone':
            for sequence in self.sequence_list.values():
                if 'larmorFreq' in sequence.mapVals:
                    sequence.mapVals['larmorFreq'] = hw.larmorFreq

        # Add time signal to the layout
        result1 = {'widget': 'curve',
                   'xData': tVector,
                   'yData': [np.abs(signal), np.real(signal), np.imag(signal)],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Echo',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 0,
                   'col': 0}

        # Add frequency spectrum to the layout
        result2 = {'widget': 'curve',
                   'xData': fVector,
                   'yData': [np.abs(spectrum)],
                   'xLabel': 'Frequency (kHz)',
                   'yLabel': 'Spectrum amplitude (a.u.)',
                   'title': 'Spectrum',
                   'legend': [''],
                   'row': 1,
                   'col': 0}

        # create self.out to run in iterative mode
        self.output = [result1, result2]

        # save data once self.output is created
        self.saveRawData()

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        return self.output


if __name__ == '__main__':
    seq = LarmorPyPulseq()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    # seq.sequencePlot(standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
