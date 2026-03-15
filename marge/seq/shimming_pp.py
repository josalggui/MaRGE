"""
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia, Spain
@date: 19 tue Apr 2022
@email: josalggui@i3m.upv.es
"""

import numpy as np
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import marge.configs.hw_config as hw
import marge.configs.units as units
from marga_pulseq.interpreter import PSInterpreter  # Import the flocra-pulseq interpreter
import pypulseq as pp  # Import PyPulseq


class ShimmingSweep(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(ShimmingSweep, self).__init__()
        # Input the parameters
        self.decimation_factor = None
        self.oversampling_factor = None
        self.plot_seq = None
        self.acqTime = None
        self.freqOffset = None
        self.dummyPulses = None
        self.sequence_list = None
        self.nPoints = None
        self.echoTime = None
        self.repetitionTime = None
        self.dShimming = None
        self.rfReTime = None
        self.shimming0 = None
        self.rfExTime = None
        self.rfExFA = None
        self.rfReFA = None
        self.nShimming = None
        self.rfReAmp = None
        self.standalone = None
        self.plotSeq = None
        self.rfExAmp = None
        self.addParameter(key='seqName', string='ShimmingSweepInfo', val='Shimming')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='pypulseq', string='PyPulseq', val=True)
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, units=units.kHz,
                          field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90.0, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (º)', val=180.0, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=50.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=100.0, units=units.us, field='RF')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=10., units=units.ms, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=100., units=units.ms, field='SEQ')
        self.addParameter(key='nPoints', string='nPoints', val=64, field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, units=units.ms, field='SEQ')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=0, field='SEQ')
        self.addParameter(key='shimming0', string='Shimming 0', val=[-110.0, 50.0, 10.0], units=units.sh, field='OTH')
        self.addParameter(key='nShimming', string='n Shimming steps', val=4, field='OTH')
        self.addParameter(key='dShimming', string='Shimming step', val=[2.5, 2.5, 2.5], units=units.sh, field='OTH')
        self.addParameter(key='oversampling_factor', string='Oversampling factor', val=6, field='OTH',
                          tip='Oversampling factor applied during readout')
        self.addParameter(key='decimation_factor', string='Decimation factor', val=3, field='OTH',
                          tip='Decimation applied to acquired data')
        self.addParameter(key='add_rd_points', string='Add RD points', val=10, field='OTH',
                          tip='Add RD points to avoid CIC and FIR filters issues')

    def sequenceInfo(self):

        print("Shimming")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence sweep the shimming in the three axis\n")

    def sequenceTime(self):
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        nShimming = self.mapVals['nShimming']
        return (repetitionTime * nShimming * 3 / 60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        self.demo = demo  # Set demo mode
        self.plotSeq = plotSeq
        self.standalone = standalone
        self.shimming = np.array([0.0, 0.0, 0.0])

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
            adc_raster_time=1e-9,
            rf_raster_time=1e-9,
            block_duration_raster=1e-9,
        )

        '''
        Step 3: Perform any calculations required for the sequence.
        In this step, students can implement the necessary calculations, such as timing calculations, RF amplitudes, and
        gradient strengths, before defining the sequence blocks.
        '''

        # Save larmor frequency
        self.mapVals['larmorFreq'] = hw.larmorFreq + self.freqOffset * 1e-6  # MHz

        # Shimming vectors
        dsx = self.nShimming * self.dShimming[0]
        dsy = self.nShimming * self.dShimming[1]
        dsz = self.nShimming * self.dShimming[2]
        sx_vector = np.reshape(
            np.linspace(self.shimming0[0] - dsx / 2, self.shimming0[0] + dsx / 2, num=self.nShimming, endpoint=False),
            (self.nShimming, 1))
        sx_vector = np.vstack([[0.0], sx_vector, [0.0]])
        sy_vector = np.reshape(
            np.linspace(self.shimming0[1] - dsy / 2, self.shimming0[1] + dsy / 2, num=self.nShimming, endpoint=False),
            (self.nShimming, 1))
        sy_vector = np.vstack([[0.0], sy_vector, [0.0]])
        sz_vector = np.reshape(
            np.linspace(self.shimming0[2] - dsz / 2, self.shimming0[2] + dsz / 2, num=self.nShimming, endpoint=False),
            (self.nShimming, 1))
        sz_vector = np.vstack([[0.0], sz_vector, [0.0]])
        self.mapVals['sx_vector'] = sx_vector[1:-1]
        self.mapVals['sy_vector'] = sy_vector[1:-1]
        self.mapVals['sz_vector'] = sz_vector[1:-1]

        # Bandwidth, sampling period, and true acquisition time
        bw = self.nPoints / self.acqTime * 1e-6  # MHz
        sampling_period = 1 / bw / self.oversampling_factor  # us
        sampling_period = np.round(np.array(sampling_period) * hw.fpga_clk_freq_MHz).astype(np.uint32) / hw.fpga_clk_freq_MHz * self.oversampling_factor
        bw = 1 / sampling_period
        self.acqTime = self.nPoints * sampling_period * 1e-6  # s
        print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))
        self.mapVals['bw_MHz'] = bw
        self.mapVals['sampling_period_us'] = sampling_period

        '''
        Step 5: Define sequence blocks.
        In this step, you will define the building blocks of the MRI sequence, including the RF pulses, gradient pulses,
        and ADC blocks.
        '''

        # Round some parameters
        round_rf = int(np.abs(np.log10(np.abs(system.rf_raster_time))))
        round_bl = int(np.abs(np.log10(np.abs(system.block_duration_raster))))
        self.rfExTime = np.round(self.rfExTime, round_rf)


        # First delay
        delay_first = pp.make_delay(self.repetitionTime)

        # Block durations
        block_a_duration = (self.repetitionTime - self.acqTime / 2 - self.echoTime +
                            ((self.rfExTime / 2) // hw.grad_raster_time + 1) * hw.grad_raster_time)
        block_b_duration = self.repetitionTime - block_a_duration
        block_a_duration = np.round(block_a_duration, decimals=round_bl).astype(float)
        block_b_duration = np.round(block_b_duration, decimals=round_bl).astype(float)
        delay_a = pp.make_delay(block_a_duration)
        delay_b = pp.make_delay(block_b_duration)

        # Create excitation rf event
        delay_rf_ex = self.repetitionTime - self.acqTime / 2 - self.echoTime - self.rfExTime / 2
        event_duration = np.round(delay_rf_ex + self.rfExTime, decimals=round_rf)
        delay_rf_ex = event_duration - self.rfExTime
        event_rf_ex = pp.make_block_pulse(flip_angle=self.rfExFA * np.pi / 180,
                                          duration=self.rfExTime,
                                          delay=delay_rf_ex,
                                          system=system,
                                          use="excitation")

        # Create refocusing rf event
        delay_rf_re = block_b_duration - self.acqTime / 2 - self.echoTime / 2 - self.rfReTime / 2
        event_duration = np.round(delay_rf_re + self.rfReTime, decimals=round_rf)
        delay_rf_re = event_duration - self.rfReTime
        event_rf_re = pp.make_block_pulse(flip_angle=self.rfReFA * np.pi / 180,
                                          duration=self.rfReTime,
                                          delay=delay_rf_re,
                                          system=system,
                                          use="refocusing")

        # Create ADC event
        delay_adc = block_b_duration - self.acqTime
        event_adc = pp.make_adc(num_samples=self.nPoints * hw.oversamplingFactor,
                                duration=self.acqTime,  # s
                                delay=delay_adc,  # s
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

            # Add dummy pulses
            if self.dummyPulses == 0:
                batch.add_block(delay_first)
            for _ in range(self.dummyPulses):
                batch.add_block(
                    event_rf_ex,
                    delay_a,
                )
                batch.add_block(
                    event_rf_re,
                    delay_b
                )

            return batch, n_rd_points, n_adc

        '''
        Step 7: Define your createBatches method.
        In this step you will populate the batches adding the blocks previously defined in step 4, and accounting for
        number of acquired points to check if a new batch is required.
        '''

        def create_batches(case='x'):
            batches = {}  # Dictionary to save batches PyPulseq sequences
            waveforms = {}  # Dictionary to store generated waveforms per each batch
            n_rd_points_dict = {}  # Dictionary to track readout points for each batch
            n_adc = 0  # To account for number of adc windows
            batch_name = f'sequence_{case}'

            # Initialize new batch
            batches[batch_name], n_rd_points, n_adc_0 = initialize_batch()
            n_adc += n_adc_0

            # Select gradient vector
            sh_vector = []
            if case == 'x':
                sh_vector = sx_vector * hw.gFactor[0]
            elif case == 'y':
                sh_vector = sy_vector * hw.gFactor[1]
            elif case == 'z':
                sh_vector = sz_vector * hw.gFactor[2]

            # Populate the sequence
            for ii in range(1, np.size(sh_vector)):
                # Gradient ramp, up to excitation pulse
                g_amp_a = np.array([sh_vector[ii - 1, 0], sh_vector[ii, 0], sh_vector[ii, 0]]) * hw.gammaB
                t0 = 0.0
                t1 = hw.grad_rise_time
                t2 = (self.repetitionTime - self.acqTime / 2 - self.echoTime +
                      ((self.rfExTime / 2) // hw.grad_raster_time + 1) * hw.grad_raster_time)
                g_time_a = np.array([t0, t1, t2])

                # Gradient flat, up to the end of the repetition
                g_amp_b = np.array([sh_vector[ii, 0], sh_vector[ii, 0]]) * hw.gammaB
                g_time_b = np.array([0, self.repetitionTime - t2])

                if ii == np.size(sh_vector) - 1:
                    batches[batch_name].add_block(
                        pp.make_extended_trapezoid(channel=case, amplitudes=g_amp_a, times=g_time_a, system=system,
                                                   skip_check=True),
                        delay_a,
                    )

                    batches[batch_name].add_block(
                        pp.make_extended_trapezoid(channel=case, amplitudes=g_amp_b, times=g_time_b, system=system,
                                                   skip_check=True),
                        delay_b
                    )
                else:
                    batches[batch_name].add_block(
                        pp.make_extended_trapezoid(channel=case, amplitudes=g_amp_a, times=g_time_a, system=system,
                                                   skip_check=True),
                        event_rf_ex,
                    )

                    batches[batch_name].add_block(
                        pp.make_extended_trapezoid(channel=case, amplitudes=g_amp_b, times=g_time_b, system=system,
                                                   skip_check=True),
                        event_rf_re,
                        event_adc,
                    )
                    n_adc += 1
                    n_rd_points += self.nPoints

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

        waveforms_x, n_readouts_x, n_adc_x = create_batches(case='x')
        waveforms_y, n_readouts_y, n_adc_y = create_batches(case='y')
        waveforms_z, n_readouts_z, n_adc_z = create_batches(case='z')

        # Run sequence x
        if self.runBatches(waveforms=waveforms_x,
                           n_readouts=n_readouts_x,
                           n_adc=n_adc_x,
                           frequency=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                           bandwidth=bw,  # MHz
                           decimate='Normal',
                           output='x',
                           hardware=False,
                           oversampling_factor=self.oversampling_factor,
                           decimation_factor=self.decimation_factor,
                           ):
            pass
        else:
            return False

        # Run sequence y
        if self.runBatches(waveforms=waveforms_y,
                           n_readouts=n_readouts_y,
                           n_adc=n_adc_y,
                           frequency=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                           bandwidth=bw,  # MHz
                           decimate='Normal',
                           output='y',
                           hardware=False,
                           oversampling_factor=self.oversampling_factor,
                           decimation_factor=self.decimation_factor,
                           ):
            pass
        else:
            return False

        # Run sequence z
        return self.runBatches(waveforms=waveforms_z,
                        n_readouts=n_readouts_z,
                        n_adc=n_adc_z,
                        frequency=hw.larmorFreq + self.freqOffset * 1e-6,  # MHz
                        bandwidth=bw,  # MHz
                        decimate='Normal',
                        output='z',
                        hardware=False,
                        oversampling_factor=self.oversampling_factor,
                        decimation_factor=self.decimation_factor,
                        )

    def sequenceAnalysis(self, mode=None):
        super().sequenceAnalysis(mode=mode)

        # Update the shimming in hw_config
        if mode != "Standalone":
            for seqName in self.sequence_list:
                self.sequence_list[seqName].mapVals['shimming'] = [
                    float(np.round(self.mapVals['shimming0'][0], decimals=1)),
                    float(np.round(self.mapVals['shimming0'][1], decimals=1)),
                    float(np.round(self.mapVals['shimming0'][2], decimals=1))]
        return self.output


if __name__ == '__main__':
    seq = ShimmingSweep()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
