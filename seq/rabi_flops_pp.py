"""
@author: T. Guallart
@author: J.M. Algarín

@summary: increase the pulse width and plot the peak value of the signal received 
@status: under development
@todo:

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
#*****************************************************************************
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import configs.hw_config as hw
import configs.units as units
import controller.controller_device as device
from marga_pulseq.interpreter import PSInterpreter  # Import the marga-pulseq interpreter
import pypulseq as pp  # Import PyPulseq


class RabiFlops(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RabiFlops, self).__init__()
        # Input the parameters
        self.rfFA = None
        self.nScans = None
        self.echoTime = None
        self.deadTime = None
        self.repetitionTime = None
        self.freqOffset = None
        self.acqTime = None
        self.nPoints = None
        self.nSteps = None
        self.rfExTime1 = None
        self.rfExTime0 = None
        self.cal_method = None
        self.addParameter(key='seqName', string='RabiFlopsInfo', val='Rabi Flops')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='pypulseq', string='PyPulseq', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, field='RF', units=units.kHz)
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=0.0, field='RF', units=units.us)
        self.addParameter(key='echoTime', string='Echo time (ms)', val=2.0, field='SEQ', units=units.ms)
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=5., units=units.ms, field='SEQ')
        self.addParameter(key='nPoints', string='nPoints', val=60, field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=0.5, field='SEQ', units=units.ms)
        self.addParameter(key='shimming', string='Shimming', val=[0.0, 0.0, 0.0], field='OTH', units=units.sh)
        self.addParameter(key='rfFA', string='Max Flip Angle (º)', val=270, field='RF')
        self.addParameter(key='rfExTime0', string='RF pulse time, Start (us)', val=50.0, units=units.us, field='RF')
        self.addParameter(key='rfExTime1', string='RF pulse time, End (us)', val=100.0, units=units.us, field='RF')
        self.addParameter(key='nSteps', string='Number of steps', val=2, field='RF')
        self.addParameter(key='deadTime', string='Dead time (us)', val=100, field='SEQ', units=units.us)
        self.addParameter(key='rfRefPhase', string='Refocusing phase (degrees)', val=0.0, field='RF')
        self.addParameter(key='method', string='Rephasing method: 0->Amp, 1->Time', val=0, field='RF')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=0, field='SEQ')
        self.addParameter(key='cal_method', string='Calibration method', val='FID', tip='FID or ECHO', field='OTH')

    def sequenceInfo(self):
        
        print("Rabi Flops")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Rabi Flops with different methods")
        print("Notes:")
        print("Set RF refocusing amplitude to 0.0 to get single excitation behavior")
        print("Set RF refocusing time to 0.0 to auto set the RF refocusing time:")
        print("-If Rephasing method = 0, refocusing amplitude is twice the excitation amplitude")
        print("-If Rephasing method = 1, refocusing time is twice the excitation time\n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        nSteps = self.mapVals['nSteps']
        dummyPulses = self.mapVals['dummyPulses']
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        return (repetitionTime * nScans * nSteps * (dummyPulses + 1) / 60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        self.demo = demo  # Set demo mode
        self.plotSeq = plotSeq
        self.standalone = standalone

        '''
        Step 1: Define the interpreter for FloSeq/PSInterpreter.
        The interpreter is responsible for converting the high-level pulse sequence description into low-level
        instructions for the scanner hardware.
        '''

        flo_interpreter = PSInterpreter(
            tx_warmup=hw.blkTime,  # Transmit chain warm-up time (us)
            rf_center=hw.larmorFreq * 1e6,  # Larmor frequency (Hz)
            rf_amp_max=hw.b1Efficiency / (2 * np.pi) * 1e6,  # Maximum RF amplitude (Hz)
            gx_max=hw.gFactor[0] * hw.gammaB,  # Maximum gradient amplitude for X (Hz/m)
            gy_max=hw.gFactor[1] * hw.gammaB,  # Maximum gradient amplitude for Y (Hz/m)
            gz_max=hw.gFactor[2] * hw.gammaB,  # Maximum gradient amplitude for Z (Hz/m)
            grad_max=np.max(hw.gFactor) * hw.gammaB,  # Maximum gradient amplitude (Hz/m)
            grad_t=hw.grad_raster_time * 1e6,  # Gradient raster time (us)
        )

        '''
        Step 2: Define system properties using PyPulseq (pp.Opts).
        These properties define the hardware capabilities of the MRI scanner, such as maximum gradient strengths,
        slew rates, and dead times. They are typically set based on the hardware configuration file (`hw_config`).
        '''

        system = pp.Opts(
            rf_dead_time=hw.blkTime * 1e-6,  # Dead time between RF pulses (s)
            max_grad=np.max(hw.gFactor) * 1e3,  # Maximum gradient strength (mT/m)
            grad_unit='mT/m',  # Units of gradient strength
            max_slew=hw.max_slew_rate,  # Maximum gradient slew rate (mT/m/ms)
            slew_unit='mT/m/ms',  # Units of gradient slew rate
            grad_raster_time=hw.grad_raster_time,  # Gradient raster time (s)
            rise_time=hw.grad_rise_time,  # Gradient rise time (s)
            rf_raster_time=1e-6,
            block_duration_raster=1e-9
        )

        '''
        Step 3: Perform any calculations required for the sequence.
        In this step, students can implement the necessary calculations, such as timing calculations, RF amplitudes, and
        gradient strengths, before defining the sequence blocks.
        '''

        # RF excitation time vector
        rf_time = np.linspace(self.rfExTime0, self.rfExTime1, num=self.nSteps, endpoint=True)  # s
        self.mapVals['rf_time'] = rf_time  # s

        # Estimated flip angles
        fa_1 = self.rfFA
        fa_0 = fa_1 * self.rfExTime0 / self.rfExTime1
        rf_fa = np.linspace(fa_0, fa_1, num=self.nSteps, endpoint=True) * np.pi / 180
        self.mapVals['rf_fa'] = rf_fa * 180 / np.pi

        # Bandwidth and sampling period
        bw = self.nPoints / self.acqTime * 1e-6  # MHz
        sampling_period = 1 / bw  # us

        '''
        Step 4: Define the experiment to get the true bandwidth
        In this step, student needs to get the real bandwidth used in the experiment. To get this bandwidth, an
        experiment must be defined and the sampling period should be obtained using get_
        '''

        if not self.demo:
            # Define device arguments
            dev_kwargs = {
                "lo_freq": hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                "rx_t": sampling_period,  # us
                "print_infos": True,
                "assert_errors": True,
                "halt_and_reset": False,
                "fix_cic_scale": True,
                "set_cic_shift": False,  # needs to be true for open-source cores
                "flush_old_rx": False,
                "init_gpa": False,
                "gpa_fhdo_offset_time": 1 / 0.2 / 3.1,
                "auto_leds": True
            }

            # Define master arguments
            master_kwargs = {
                'mimo_master': True,
                'trig_output_time': 1e5,
                'slave_trig_latency': 6.079
            }

            dev = device.Device(ip_address=hw.rp_ip_list[0], port=hw.rp_port, **(master_kwargs | dev_kwargs))
            sampling_period = dev.get_sampling_period()  # us
            bw = 1 / sampling_period  # MHz
            print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))
            dev.__del__()

        self.mapVals['bw_MHz'] = bw
        self.mapVals['sampling_period_us'] = sampling_period

        '''
        Step 5: Define sequence blocks.
        In this step, you will define the building blocks of the MRI sequence, including the RF pulses, gradient pulses,
        and ADC blocks.
        '''

        # Create ADC event
        event_adc = pp.make_adc(num_samples=self.nPoints * hw.oversamplingFactor,
                                duration=self.acqTime,  # s
                                delay=0,  # s
                                system=system)

        '''
        Step 6: Define your initializeBatch according to your sequence.
        In this step, you will create the initializeBatch method to create dummy pulses that will be initialized for
        each new batch.
        '''

        def initialize_batch():
            # Instantiate pypulseq sequence object
            batch = pp.Sequence(system)
            n_rd_points = 0
            n_adc = 0

            batch.add_block(pp.make_delay(self.repetitionTime - rf_time[0] / 2 - system.rf_dead_time))

            return batch, n_rd_points, n_adc

        '''
        Step 7: Define your createBatches method.
        In this step you will populate the batches adding the blocks previously defined in step 4, and accounting for
        number of acquired points to check if a new batch is required.
        '''

        def create_batches():
            batches = {}  # Dictionary to save batches PyPulseq sequences
            waveforms = {}  # Dictionary to store generated waveforms per each batch
            n_rd_points_dict = {}  # Dictionary to track readout points for each batch
            n_adc = 0  # To account for number of adc windows
            batch_name = 'batch_1'

            # Initialize new batch
            batches[batch_name], n_rd_points, n_adc_0 = initialize_batch()
            n_adc += n_adc_0

            ###########################################################

            for step in range(self.nSteps):
                # Create excitation rf event
                event_rf_ex = pp.make_block_pulse(flip_angle=rf_fa[step],
                                                  duration=rf_time[step],
                                                  delay=0,
                                                  system=system,
                                                  use="excitation")

                # Create first adc
                event_adc_a = pp.make_adc(
                    num_samples=self.nPoints * hw.oversamplingFactor,
                    duration=self.acqTime,
                    delay=system.rf_dead_time + rf_time[step] + self.deadTime,
                    system=system,
                )

                # Create second adc
                event_adc_b = pp.make_adc(
                    num_samples=self.nPoints * hw.oversamplingFactor,
                    duration=self.acqTime,
                    delay=system.rf_dead_time + rf_time[step] + self.echoTime / 2 - self.acqTime,
                    system=system,
                )

                # Create refocusing rf event
                event_rf_re = pp.make_block_pulse(flip_angle=rf_fa[step] * 2,
                                                  duration=rf_time[step] * 2,
                                                  delay=0,
                                                  system=system,
                                                  use="refocusing")

                if step < self.nSteps - 1:
                    batches[batch_name].add_block(
                        event_rf_ex,
                        event_adc_a,
                        pp.make_delay(self.echoTime / 2 + rf_time[step] / 2 - rf_time[step]),
                    )
                    batches[batch_name].add_block(
                        event_rf_re,
                        event_adc_b,
                        pp.make_delay(self.repetitionTime - self.echoTime / 2 + rf_time[step] - rf_time[step + 1] / 2)
                    )

                else:
                    batches[batch_name].add_block(
                        event_rf_ex,
                        event_adc_a,
                        pp.make_delay(self.echoTime / 2 + rf_time[step] / 2 - rf_time[step]),
                    )
                    batches[batch_name].add_block(
                        event_rf_re,
                        event_adc_b,
                    )

                n_rd_points += 2 * self.nPoints  # Accounts for additional acquired points in each adc block
                n_adc += 2

            ###########################################################

            # Write sequence file and interpret the sequence to get the waveform
            batches[batch_name].write(batch_name + ".seq")
            waveforms[batch_name], param_dict = flo_interpreter.interpret(batch_name + ".seq")
            print(f"{batch_name}.seq ready!")
            n_rd_points_dict[batch_name] = n_rd_points

            return waveforms, n_rd_points_dict, n_adc

        '''
        Step 8: Run the batches
        This step will handle the different batches, run it and get the resulting data. This should not be modified.
        Oversampled data will be available in self.mapVals['data_over']
        Decimated data will be available in self.mapVals['data_decimated']
        The decimated data is shifted to account for CIC delay, so data is synchronized with real-time signal
        '''

        waveforms, n_readouts, n_adc = create_batches()
        return self.runBatches(waveforms=waveforms,
                               n_readouts=n_readouts,
                               n_adc=n_adc,
                               frequency=hw.larmorFreq,  # MHz
                               bandwidth=bw,  # MHz
                               decimate='Normal',
                               hardware=False,
                               )

    def sequenceAnalysis(self, mode=None):
        self.mode = mode

        # Get time vector
        timeVector = self.mapVals['rf_time']
        fa_vector = self.mapVals['rf_fa']

        # Get FID and Echo
        dataFull = self.mapVals['data_decimated'][0]
        dataFull = np.reshape(dataFull, (self.nScans, self.nSteps, 2, -1))
        dataFID = dataFull[:, :, 0, :]
        self.mapVals['dataFID'] = dataFID
        dataEcho = dataFull[:, :, 1, :]
        self.mapVals['dataEcho'] = dataEcho
        dataFIDAvg = np.mean(dataFID, axis=0)
        self.mapVals['dataFIDAvg'] = dataFIDAvg
        dataEchoAvg = np.mean(dataEcho, axis=0)
        self.mapVals['dataEchoAvg'] = dataEchoAvg

        rabiFID = dataFIDAvg[:, 10]
        self.mapVals['rabiFID'] = rabiFID
        rabiEcho = dataEchoAvg[:, int(self.nPoints / 2)]
        self.mapVals['rabiEcho'] = rabiEcho

        # Get values for pi/2 and pi pulses
        test = True
        n = 1
        while test:
            if n >= self.nSteps:
                break
            if self.cal_method == 'FID':
                d = np.abs(rabiFID[n]) - np.abs(rabiFID[n - 1])
            elif self.cal_method == 'ECHO':
                d = np.abs(rabiEcho[n]) - np.abs(rabiEcho[n - 1])
            else:
                break
            n += 1
            if d < 0:
                test = False
        piHalfTime = timeVector[n - 2] * 1e6  # us
        pi_half_fa = fa_vector[n - 2]
        self.mapVals['piHalfTime'] = piHalfTime
        self.mapVals['pi_half_fa'] = pi_half_fa
        hw.b1Efficiency = hw.b1Efficiency / 90 * pi_half_fa
        b1_max_kHz = hw.b1Efficiency / (2 * np.pi) * 1e3  # kHz
        b1_max_uT = hw.b1Efficiency / (2 * np.pi) / hw.gammaB * 1e12
        print("RF coil efficiency: %0.1f kHz / %0.1f uT" % (b1_max_kHz, b1_max_uT))

        # Signal vs rf time
        result1 = {'widget': 'curve',
                   'xData': timeVector * 1e6,
                   'yData': [np.abs(rabiFID), np.real(rabiFID), np.imag(rabiFID)],
                   'xLabel': 'Time (us)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Rabi Flops with FID',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 0,
                   'col': 0}

        result2 = {'widget': 'curve',
                   'xData': timeVector * 1e6,
                   'yData': [np.abs(rabiEcho), np.real(rabiEcho), np.imag(rabiEcho)],
                   'xLabel': 'Time (us)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Rabi Flops with Spin Echo',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 1,
                   'col': 0}

        self.output = [result1, result2]

        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()

        return self.output

if __name__ == '__main__':
    seq = RabiFlops()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
