"""
@author: J.M. Algarín

@summary: increase the pulse width and plot the peak value of the signal received 
@status: under development
@todo:

"""
import numpy as np
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import marge.configs.hw_config as hw
import marge.configs.units as units
import marge.controller.controller_device as device
import marge.controller.experiment_gui as ex
from marga_pulseq.interpreter import PSInterpreter  # Import the marga-pulseq interpreter
import pypulseq as pp  # Import PyPulseq


class RabiFlops(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(RabiFlops, self).__init__()
        # Input the parameters
        self.decimation_factor = None
        self.oversampling_factor = None
        self.larmorFreq = None
        self.rfExAmp = None
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
        self.addParameter(key='seqName', string='RabiFlopsInfo', val='RabiFlops')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='pypulseq', string='PyPulseq', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.06, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=0.0, field='RF', units=units.us)
        self.addParameter(key='echoTime', string='Echo time (ms)', val=2.0, field='SEQ', units=units.ms)
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=5., units=units.ms, field='SEQ')
        self.addParameter(key='nPoints', string='nPoints', val=60, field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=0.5, field='SEQ', units=units.ms)
        self.addParameter(key='shimming', string='Shimming', val=[0.0, 0.0, 0.0], field='OTH', units=units.sh)
        self.addParameter(key='rfExTime0', string='RF pulse time, Start (us)', val=50.0, units=units.us, field='RF')
        self.addParameter(key='rfExTime1', string='RF pulse time, End (us)', val=100.0, units=units.us, field='RF')
        self.addParameter(key='nSteps', string='Number of steps', val=5, field='RF')
        self.addParameter(key='deadTime', string='Dead time (us)', val=100, field='SEQ', units=units.us)
        self.addParameter(key='rfRefPhase', string='Refocusing phase (degrees)', val=0.0, field='RF')
        self.addParameter(key='method', string='Rephasing method: 0->Amp, 1->Time', val=0, field='RF')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=0, field='SEQ')
        self.addParameter(key='cal_method', string='Calibration method', val='FID', tip='FID or ECHO', field='OTH')
        self.addParameter(key='discriminator', string='Calibration point', val='max', field='OTH',
                          tip="'max' to use maximum for 90º or 'min' to use zero-crossing for 180º")
        self.addParameter(key='oversampling_factor', string='Oversampling factor', val=6, field='OTH',
                          tip='Oversampling factor applied during readout')
        self.addParameter(key='decimation_factor', string='Decimation factor', val=3, field='OTH',
                          tip='Decimation applied to acquired data')
        self.addParameter(key='add_rd_points', string='Add RD points', val=10, field='OTH',
                          tip='Add RD points to avoid CIC and FIR filters issues')

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
            rf_raster_time=1e-9,
            block_duration_raster=1e-9
        )

        '''
        Step 3: Perform any calculations required for the sequence.
        In this step, students can implement the necessary calculations, such as timing calculations, RF amplitudes, and
        gradient strengths, before defining the sequence blocks.
        '''

        # RF excitation time vector
        rf_time = np.linspace(self.rfExTime0, self.rfExTime1, num=self.nSteps, endpoint=True)  # s
        self.mapVals['rfTime'] = rf_time  # s

        # Estimated flip angles
        rf_fa = hw.b1Efficiency * self.rfExAmp * rf_time * 1e6
        self.mapVals['rf_fa'] = rf_fa * 180 / np.pi

        # Bandwidth, sampling period, and true acquisition time
        bw = self.nPoints / self.acqTime * 1e-6  # MHz
        sampling_period = 1 / bw / self.oversampling_factor  # us
        sampling_period = np.round(np.array(sampling_period) * hw.fpga_clk_freq_MHz).astype(
            np.uint32) / hw.fpga_clk_freq_MHz * self.oversampling_factor
        bw = 1 / sampling_period
        acq_time = self.nPoints * sampling_period * 1e-6  # s
        print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))
        self.mapVals['bw_MHz'] = bw
        self.mapVals['sampling_period_us'] = sampling_period

        '''
        Step 5: Define sequence blocks.
        In this step, you will define the building blocks of the MRI sequence, including the RF pulses, gradient pulses,
        and ADC blocks.
        '''

        round_rf = int(np.abs(np.log10(np.abs(system.rf_raster_time))))
        round_bl = int(np.abs(np.log10(np.abs(system.block_duration_raster))))

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

            # First delay
            delay = np.round(self.repetitionTime - rf_time[0] / 2 - system.rf_dead_time, decimals=round_bl)
            batch.add_block(pp.make_delay(delay))

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
                rf_time_ex = np.round(rf_time[step], decimals=round_rf)
                event_rf_ex = pp.make_block_pulse(flip_angle=rf_fa[step],
                                                  duration=rf_time_ex,
                                                  delay=0,
                                                  system=system,
                                                  use="excitation")

                # Create first adc
                delay = system.rf_dead_time + rf_time_ex + self.deadTime
                block_duration = np.round(acq_time + delay, decimals=round_bl)
                delay = block_duration - acq_time
                event_adc_a = pp.make_adc(
                    num_samples=self.nPoints,
                    duration=acq_time,
                    delay=delay,
                    system=system,
                )

                # Create refocusing rf event
                rf_time_re = np.round(2 * rf_time[step], decimals=round_rf)
                event_rf_re = pp.make_block_pulse(flip_angle=rf_fa[step] * 2,
                                                  duration=rf_time_re,
                                                  delay=0,
                                                  system=system,
                                                  use="refocusing")

                # Create second adc
                delay = system.rf_dead_time + rf_time_re / 2 + self.echoTime / 2 - acq_time / 2
                block_duration = np.round(acq_time + delay, decimals=round_bl)
                delay = block_duration - acq_time
                event_adc_b = pp.make_adc(
                    num_samples=self.nPoints,
                    duration=acq_time,
                    delay=delay,
                    system=system,
                )

                if step < self.nSteps - 1:
                    delay_ex = np.round(rf_time_ex / 2 + self.echoTime / 2 - rf_time_re / 2, decimals=round_bl)
                    batches[batch_name].add_block(
                        event_rf_ex,
                        event_adc_a,
                        pp.make_delay(delay_ex),
                    )
                    delay_re = np.round(self.repetitionTime - self.echoTime / 2 + rf_time[step] - rf_time[step + 1] / 2, decimals=round_bl)
                    batches[batch_name].add_block(
                        event_rf_re,
                        event_adc_b,
                        pp.make_delay(delay_re)
                    )

                else:
                    delay_ex = np.round(self.echoTime / 2 + rf_time_ex / 2 - rf_time_re / 2, decimals=round_bl)
                    batches[batch_name].add_block(
                        event_rf_ex,
                        event_adc_a,
                        pp.make_delay(delay_ex),
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
                               hardware=True,
                               oversampling_factor=self.oversampling_factor,
                               decimation_factor=self.decimation_factor,
                               )

if __name__ == '__main__':
    seq = RabiFlops()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
