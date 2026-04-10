"""Pulse sequence for automatic receiver gain calibration."""

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

import marge.seq.larmor as larmor
import marge.configs.hw_config as hw
from marge.utils.SerialDevice import SerialDevice


class FixGain(larmor.Larmor):
    def __init__(self):
        super(FixGain, self).__init__()
        self.attenuation = None
        self.mode = None
        self.setParameter(key='seqName', string='FixGain', val='FixGain')
        self.addParameter(key='toMaRGE', val=False)
        self.addParameter(key='gain', string='Attenuation (dB)', val=45, field='OTH',
                          tip='Integer from %i to %i' % (hw.rf_min_gain, hw.rf_max_gain))
        self.addParameter(key='mode', string='Mode', val='AUTO', field='OTH', tip="'AUTO' or 'MANUAL'")
        self.arduino = None

    def _open_arduino(self):
        if self.arduino is None:
            self.arduino = SerialDevice(name="Arduino attenuator")
            self.arduino.connect(serial_number=hw.ard_sn_attenuator)
        return self.arduino.device is not None

    def _close_arduino(self):
        if self.arduino is not None:
            self.arduino.disconnect()
            self.arduino = None

    def _set_gain(self):
        if not self._open_arduino():
            print("WARNING: No Arduino found for attenuator.")
            return False

        gain_binary = bin(self.mapVals['gain'] - hw.rf_min_gain)[2:].zfill(5)
        try:
            self.arduino.send("1" + gain_binary)
            print("RF gain: %i dB" % self.mapVals['gain'])
            return True
        finally:
            self._close_arduino()

    def sequenceInfo(self):
        print("Set RF gain of the scanner")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Set the gain of the RF chain.")
        print("Specific hardware from MRILab @ i3M is required.\n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        return (repetitionTime * nScans / 60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        if self.mode == 'MANUAL':
            if hw.rf_min_gain > self.mapVals['gain'] or hw.rf_max_gain < self.mapVals['gain']:
                print("ERROR: Gain must be between %i and %i dB" % (hw.rf_min_gain, hw.rf_max_gain))
                return False
            else:
                return True
        elif self.mode == 'AUTO':
            return super().sequenceRun(plotSeq=plotSeq, demo=demo, standalone=standalone)
        else:
            print("ERROR: Mode must be AUTO or MANUAL")
            return False

    def sequenceAnalysis(self, mode=None, save=True):
        """
        Process raw acquired data and compute the output images and metrics.

        Reconstructs the image from k-space, computes SNR or other sequence-specific
        figures of merit, populates output_dict with result arrays, and fills
        dicom_meta_data with the relevant DICOM tags for saving.

        Args:
            mode (str, optional): Processing mode selector (sequence-dependent). Defaults to None.
            save (bool, optional): Whether to save results to disk. Defaults to True.

        Returns:
            tuple: (output_dict, dicom_meta_data) with processed results and metadata.
        """
        if self.mode == 'MANUAL':
            if not self._set_gain():
                return []

            # save data once self.output is created
            self.saveRawData()

            return []

        elif self.mode == 'AUTO':
            super().sequenceAnalysis(mode=mode, save=False)

            # Get echo amplitude
            data = self.mapVals['data']
            echo_amplitude = np.max(np.abs(data))

            # Get desired gain
            desired_gain = 0.9 * hw.rp_max_input_voltage / echo_amplitude
            desired_gain = int(20 * np.log10(desired_gain))
            self.mapVals['gain'] += desired_gain
            if self.mapVals['gain'] > hw.rf_max_gain:
                print("WARNING: Required gain larger than maximum gain.")
                print("Gain will be set to maximum gain.")
                self.mapVals['gain'] = hw.rf_max_gain
            elif self.mapVals['gain'] < hw.rf_min_gain:
                print("WARNING: Required gain smaller than minimum gain.")
                print("Gain will be set to minimum gain.")
                self.mapVals['gain'] = hw.rf_min_gain

            # Set gain
            if not self._set_gain():
                return []
            hw.lnaGain = self.mapVals['gain']

            # save data once self.output is created
            self.saveRawData()

            if self.mode == 'Standalone':
                self.plotResults()

            return self.output

        else:
            # save data once self.output is created
            self.saveRawData()

            return []

if __name__ == '__main__':
    seq = FixGain()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')
