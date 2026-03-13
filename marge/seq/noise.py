"""
@author: J.M. Algarín, february 03th 2022
MRILAB @ I3M
"""


import time
import marge.configs.hw_config as hw
import marge.controller.experiment_gui as ex
import marge.controller.controller_device as device
import numpy as np
from marge.seq.mriBlankSeq import MRIBLANKSEQ
import marge.configs.units as units

class Noise(MRIBLANKSEQ):
    def __init__(self):
        super().__init__()

        # Input the parameters
        self.repetitionTime = None
        self.rxChannel = None
        self.nPoints = None
        self.bw = None
        self.freqOffset = None
        self.addParameter(key='seqName', string='NoiseInfo', val='Noise')
        self.addParameter(key='toMaRGE', val=False)
        self.addParameter(key='freqOffset', string='RF frequency offset (kHz)', val=0.0, units=units.kHz, field='RF')
        self.addParameter(key='nPoints', string='Number of points', val=2500, field='RF')
        self.addParameter(key='bw', string='Acquisition bandwidth (kHz)', val=50.0, units=units.kHz, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=500.0, field='RF', units=units.ms)
        self.addParameter(key='sleepTime', string='Sleep Time (s)', val=0.0, field='OTH')

    def sequenceInfo(self):
        print("Noise")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Acquires a noise measurement\n")

    def sequenceTime(self):
        return 0  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False
        self.demo = demo

        # Fix units to MHz and us
        self.freqOffset *= 1e-6  # MHz
        self.bw *= 1e-6  # MHz
        self.repetitionTime *= 1e6  # us

        self.mapVals['larmorFreq'] = hw.larmorFreq
        samplingPeriod = 1 / self.bw

        if self.demo:
            n_samples = (self.nPoints + 2 * hw.addRdPoints) * hw.oversamplingFactor
            data_real = np.random.randn(n_samples)
            data_imag = np.random.randn(n_samples)
            data = data_real + 1j * data_imag
            data = self.decimate(data_over=data, n_adc=1, option="Normal")
            self.mapVals["data"] = data
            time.sleep(self.repetitionTime * 1e-6)
            return True

        if hw.marcos_version=="MaRCoS":
            self.expt = ex.Experiment(
                lo_freq=hw.larmorFreq + self.freqOffset,
                rx_t=samplingPeriod,
                init_gpa=init_gpa,
                gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                print_infos=False,
            )

        if hw.marcos_version=="MIMO":
            # Define device arguments
            dev_kwargs = {
                "lo_freq": hw.larmorFreq,
                "rx_t": samplingPeriod,
                "print_infos": True,
                "assert_errors": True,
                "halt_and_reset": False,
                "fix_cic_scale": True,
                "set_cic_shift": False,  # needs to be true for open-source cores
                "flush_old_rx": False,
                "init_gpa": False,
                "gpa_fhdo_offset_time": 1 / 0.2 / 3.1,
                "auto_leds": True,
            }

            # Define master arguments
            master_kwargs = {
                'mimo_master': True,
                'trig_output_time': 1e5,
                'slave_trig_latency': 6.079
            }

            # Define experiment
            self.expt = device.MimoDevices(ips=hw.rp_ip_list, ports=hw.rp_port, **(master_kwargs | dev_kwargs))
            self.devices = self.expt.dev_list()
            samplingPeriod = self.expt.getSamplingRate()
            self.bw = 1/samplingPeriod
            acqTime = self.nPoints/self.bw

            # Sequence definition
            self.iniSequence(20, np.array((0, 0, 0)))
            t0 = 30 + hw.addRdPoints*hw.oversamplingFactor/self.bw
            self.ttlOffRecPulse(t0, acqTime)
            self.rxGateSync(t0, acqTime, channel=self.rxChannel)
            t0 = t0 + acqTime + hw.addRdPoints*hw.oversamplingFactor/self.bw
            if t0 < self.repetitionTime:
                self.endSequence(self.repetitionTime)
            else:
                self.endSequence(t0+20)

            # Load sequence into red pitaya
            if self.floDict2Exp():
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("ERROR: sequence waveforms out of hardware bounds")
                return False

            if plotSeq == 0:
                if hw.marcos_version=="MaRCoS":
                    rxd, msgs = self.expt.run()
                    data = self.decimate(rxd['rx%i' % self.rxChannel], 1, option='Normal')

                elif hw.marcos_version=="MIMO":
                    data = [[] for _ in range(len(self.channels))]
                    result = self.expt.run()  # Run the experiment and collect data
                    prov = [tup[0] for tup in result]  # List of rx results for each device

                    results = {}
                    for channel in self.channels:
                        results[f'rx{channel}'] = prov[(channel - 1) // 2][f'rx{(channel - 1) % 2}']

                    for ii in range(len(self.channels)):
                        data_decimated = self.decimate(results[f'rx{self.channels[ii]}'], 1, option='Normal')
                        data[ii] = np.concatenate((data[ii], data_decimated), axis=0)

                self.mapVals['data'] = data

            if hw.marcos_version == "MaRCoS":
                self.expt.__del__()
            elif hw.marcos_version == "MIMO":
                for dev in self.devices:
                    dev.__del__()

        return True


if __name__=='__main__':
    seq = Noise()
    seq.sequenceAtributes()
    seq.sequenceRun(demo=False)
    seq.sequenceAnalysis(mode='Standalone')

