import os
import sys

import numpy as np

# *****************************************************************************
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
# ******************************************************************************

import seq.larmor as larmor
import configs.hw_config as hw
import autotuning.autotuning as autotuning


class FixGain(larmor.Larmor):
    def __init__(self):
        super(FixGain, self).__init__()
        self.attenuation = None
        self.mode = None
        self.setParameter(key='seqName', string='FixGain', val='FixGain')
        self.addParameter(key='gain', string='Attenuation (dB)', val=50, field='OTH',
                          tip='Integer from %i to %i' % (hw.rf_min_gain, hw.rf_max_gain))
        self.addParameter(key='mode', string='Mode', val='AUTO', field='OTH', tip="'AUTO' or 'MANUAL'")

        # Connect to Arduino and set the initial state
        self.arduino = autotuning.Arduino(name="attenuator", serial_number=hw.ard_sn_attenuator)
        self.arduino.connect()
        gain_binary = bin(self.mapVals['gain']-hw.rf_min_gain)[2:].zfill(5)
        self.arduino.send("1" + gain_binary)
        print("\nRF gain: %i dB" % self.mapVals['gain'])

    def sequenceInfo(self):
        print("\nSet RF gain of the scanner")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Set the gain of the RF chain.")
        print("Specific hardware from MRILab @ i3M is required.\n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        return (repetitionTime * nScans / 60)  # minutes, scanTime

    def sequenceAnalysis(self, mode=None, save=True):
        super().sequenceAnalysis(mode=mode, save=False)

        # Get echo amplitude
        data = self.mapVals['data']
        echo_amplitude = np.max(np.abs(data))

        # Get desired amplification
        desired_gain = 0.9 * hw.rp_max_input_voltage / echo_amplitude
        desired_gain = int(20 * np.log10(desired_gain))

        # Set gain
        self.mapVals['gain'] += desired_gain
        gain_binary = bin(self.mapVals['gain'] - hw.rf_min_gain)[2:].zfill(5)
        self.arduino.send("1" + gain_binary)
        print("\nRF gain: %i dB" % self.mapVals['gain'])
        hw.lnaGain = self.mapVals['gain']

        # save data once self.output is created
        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()

        return self.output

if __name__ == '__main__':
    seq = FixGain()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
