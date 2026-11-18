"""
MIMO coherence test sequence.

The SDRLab (master) transmits an RF burst at the Larmor frequency (3.04 MHz). An external RF splitter sends
the burst to RX1 (fixed) and to RXn (moved by hand from RX2 to RX12 between runs). All RX channels are
acquired simultaneously. The acquisition is repeated n_repetitions times in a single run.

Based on noise.py (MIMO device handling) and larmor.py (TX burst + rx gate).
"""

import os
import sys
import time

# *****************************************************************************
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory = os.path.dirname(parent_directory)
for subdir in ['MaRGE', 'marcos_client']:
    sys.path.append(os.path.join(parent_directory, subdir))
# ******************************************************************************
import numpy as np
import marge.configs.hw_config as hw
import marge.controller.controller_device as device
import marge.configs.units as units
from marge.seq.mriBlankSeq import MRIBLANKSEQ


class MimoCoherence(MRIBLANKSEQ):
    def __init__(self):
        super().__init__()
        self.larmorFreq = None
        self.rfAmp = None
        self.settleTime = None
        self.burstTime = None
        self.nPoints = None
        self.bw = None
        self.nRepetitions = None
        self.repetitionTime = None
        self.rxSplitter = None
        self.burstsPerRun = None
        self.addParameter(key='seqName', string='MimoCoherenceInfo', val='MimoCoherence')
        self.addParameter(key='toMaRGE', string='to MaRGE', val=True)
        self.addParameter(key='larmorFreq', string='RF frequency (MHz)', val=3.04, units=units.MHz, field='RF')
        self.addParameter(key='rfAmp', string='TX amplitude (0-1)', val=0.05, field='RF',
                          tip='Keep it low: TX goes directly to the RX inputs through the splitter')
        self.addParameter(key='nPoints', string='Number of points', val=100, field='RF')
        self.addParameter(key='bw', string='Acquisition bandwidth (kHz)', val=50.0, units=units.kHz, field='RF')
        self.addParameter(key='settleTime', string='Burst settle time (us)', val=100.0, units=units.us, field='RF',
                          tip='Burst margin before and after the acquisition window')
        self.addParameter(key='burstTime', string='Burst duration (us), 0 = cover the window', val=0.0, units=units.us,
                          field='RF',
                          tip='0: the burst covers the whole acquisition window (plateau only). >0: shorter burst '
                              'centred in the window, so the rising and falling edges are inside the acquired data. '
                              'Use about 30-70% of the window (window = points / bandwidth).')
        self.addParameter(key='nRepetitions', string='Repetitions', val=100, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=50.0, units=units.ms, field='SEQ')
        self.addParameter(key='burstsPerRun', string='Bursts per hardware run', val=10, field='SEQ',
                          tip='Bursts acquired in one expt.run(). Fewer host round trips = more stable')
        self.addParameter(key='rxSplitter', string='Compare RX1 with (0 = all, 2-12 = that RX)', val=0, field='SEQ',
                          tip='0 = splitter 1:12, all channels against RX1. 2-12 = splitter 1:2, RX1 against that RX '
                              '(change it every time you move the cable). It only selects the analysis, the acquisition is '
                              'the same. It is saved in the .mat')

    def sequenceInfo(self):
        print("MIMO coherence")
        print("TX burst split to RX1 and RXn, 100 repetitions, amplitude and relative phase stability\n")

    def sequenceTime(self):
        return self.mapVals['nRepetitions'] * self.mapVals['repetitionTime'] * 1e-3 / 60  # minutes

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        self.demo = demo

        # Units: MHz and us
        bw = self.bw * 1e-6  # MHz
        repetition_time = self.repetitionTime * 1e6  # us
        settle = self.settleTime * 1e6  # us
        sampling_period = 1 / bw  # us
        lo_freq = self.larmorFreq * 1e-6  # MHz
        n_rep = int(self.nRepetitions)
        n_points = int(self.nPoints)

        if self.rfAmp > 1 or self.rfAmp <= 0:
            print("ERROR: TX amplitude must be in (0, 1]")
            return False

        # Optional short burst (edges inside the window). 0 keeps the original behaviour (burst covers the window).
        burst_short = float(self.burstTime) * 1e6  # us
        if burst_short < 0:
            print("ERROR: burst duration must be >= 0")
            return False
        if burst_short > 0:
            acq_nominal = n_points / bw  # us
            if burst_short < 0.2 * acq_nominal or burst_short > 0.9 * acq_nominal:
                print("ERROR: burst duration (%0.0f us) must be between 20%% and 90%% of the acquisition window "
                      "(%0.0f us) so that both edges are inside the window" % (burst_short, acq_nominal))
                return False
        self.mapVals['burstTime_us'] = burst_short

        if int(self.rxSplitter) != 0 and not 2 <= int(self.rxSplitter) <= 12:
            print("ERROR: 'Compare RX1 with' must be 0 (all channels) or 2-12")
            return False
        self.mapVals['rxSplitter'] = int(self.rxSplitter)
        self.mapVals['nRepetitions'] = n_rep
        self.mapVals['rfFreq_MHz'] = lo_freq

        if self.demo:
            channels = list(range(1, 13))
            data = np.random.randn(len(channels), n_rep, n_points) + 1j * np.random.randn(len(channels), n_rep, n_points)
            self.mapVals['channels'] = channels
            self.mapVals['data_decimated'] = data
            self.mapVals['data'] = data
            self.mapVals['bw'] = self.bw * 1e-3  # kHz
            return True

        # Define device arguments (same as noise.py)
        dev_kwargs = {
            "lo_freq": lo_freq,
            "rx_t": sampling_period,
            "print_infos": True,
            "assert_errors": True,
            "halt_and_reset": False,
            "fix_cic_scale": True,
            "set_cic_shift": False,  # needs to be true for open-source cores
            "flush_old_rx": True,  # Limpa o buffer antes de cada run (evita deadlock)
            "init_gpa": False,
            "gpa_fhdo_offset_time": 1 / 0.2 / 3.1,
            "auto_leds": True,
        }
        master_kwargs = {
            'mimo_master': True,
            'trig_output_time': 1e5,
            'slave_trig_latency': 6.079
        }

        # Inicializa os dispositivos
        self.expt = device.MimoDevices(ips=hw.rp_ip_list, ports=hw.rp_port, **(master_kwargs | dev_kwargs))
        self.devices = self.expt.dev_list()
        sampling_period = self.expt.getSamplingRate()  # us
        bw = 1 / sampling_period  # MHz
        acq_time = n_points / bw  # us
        self.mapVals['bw'] = bw * 1e3  # kHz
        self.mapVals['samplingPeriod'] = sampling_period

        # RX channels (1-based): two per Red Pitaya
        channels = getattr(self, 'channels', None)
        if channels is None or len(channels) == 0:
            channels = list(range(1, 2 * len(self.devices) + 1))
        channels = list(channels)
        self.mapVals['channels'] = channels

        # Sequence definition: several TX bursts per hardware run, each one covering its own acquisition window
        bursts = max(1, min(int(self.burstsPerRun), n_rep))
        n_runs = int(np.ceil(n_rep / bursts))
        self.mapVals['burstsPerRun'] = bursts
        pad = hw.addRdPoints * hw.oversamplingFactor / bw  # us
        lead = max(float(hw.blkTime), float(getattr(hw, 'blkOffTime', 0.0)))
        rf_start = max(100.0, 30.0 + lead - settle - pad)  # us, first burst start
        burst_time = settle + pad + acq_time + pad + settle  # us
        tr = max(repetition_time, burst_time + 200.0)  # us between bursts

        self.iniSequence(20, np.array((0, 0, 0)))
        for k in range(bursts):
            t_k = rf_start + k * tr
            gate_t0 = t_k + settle + pad
            if burst_short > 0:
                pulse_t0 = gate_t0 + (acq_time - burst_short) / 2
                pulse_dur = burst_short
            else:
                pulse_t0 = t_k
                pulse_dur = burst_time
            self.rfRecPulse(pulse_t0 - hw.blkTime, pulse_dur, self.rfAmp, 0)
            self.ttlOffRecPulse(gate_t0, acq_time)
            for rx in (0, 1):
                self.rxGateSync(gate_t0, acq_time, channel=rx)
        self.endSequence(rf_start + bursts * tr)

        if self.floDict2Exp():
            print("Sequence waveforms loaded successfully")
        else:
            print("ERROR: sequence waveforms out of hardware bounds")
            for dev in self.devices:
                try:
                    dev.__del__()
                except:
                    pass
            return False

        data = [[] for _ in channels]
        complete = False
        expected = bursts * (n_points + 2 * hw.addRdPoints) * hw.oversamplingFactor

        if not plotSeq:
            try:
                for run in range(n_runs):
                    # Tenta rodar até 3 vezes, mas sem reiniciar as placas
                    success = False
                    for attempt in range(3):
                        try:
                            result = self.expt.run()
                            prov = [tup[0] for tup in result]
                            if np.size(prov[0]['rx0']) == expected:
                                success = True
                                break
                            print("WARNING: expected %i points, got %i. Repeating run..." %
                                  (expected, np.size(prov[0]['rx0'])))
                        except Exception as e:
                            print("ERROR on run %i: %s" % (run + 1, e))
                            # Se der erro, não tenta recuperar. Apenas sai.
                            raise

                    if not success:
                        raise RuntimeError("Run %i failed after 3 attempts" % (run + 1))

                    for ii, ch in enumerate(channels):
                        raw = prov[(ch - 1) // 2][f'rx{(ch - 1) % 2}']
                        dec = self.decimate(raw, bursts, option='Normal')
                        data[ii].append(np.reshape(dec, (bursts, -1)))
                    print("Run %i/%i ready (%i repetitions)" % (run + 1, n_runs, min((run + 1) * bursts, n_rep)))
                complete = True
            except Exception as e:
                print("ERROR during acquisition: %s" % e)
            finally:
                # Fecha os dispositivos no final, sempre
                for dev in self.devices:
                    try:
                        dev.__del__()
                    except:
                        pass

            if len(data[0]) > 0:
                data = np.array([np.concatenate(d, axis=0)[:n_rep] for d in data])
                self.mapVals['data_decimated'] = data
                self.mapVals['data'] = data
                self.mapVals['nRepetitions'] = data.shape[1]
            return complete

        # Se plotSeq for True, fecha e retorna
        for dev in self.devices:
            try:
                dev.__del__()
            except:
                pass
        return True

    def sequenceAnalysis(self, mode=None):
        super().sequenceAnalysis(mode=mode)
        return self.output


if __name__ == '__main__':
    seq = MimoCoherence()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')