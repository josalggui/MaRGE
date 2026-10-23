"""
SDRO sequence file for MaRGE.
Controls RF gain/attenuation via Arduino-controlled attenuator + 26 dB amplifier.

Two modes available via MaRGE GUI:
    - auto:       one-shot deterministic gain from Larmor echo amplitude
    - manual:     user inputs gain directly (use when signal saturates)

Author: MRILab, i3M, CSIC, Valencia, Spain
"""

import marge.seq.mriBlankSeq as blankSeq
import marge.configs.hw_config as hw
from marge.controller.controller_sdro import SDRO


class SDROSeq(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(SDROSeq, self).__init__()

        self.sdro = None

        # Parameters visible in MaRGE GUI
        self.addParameter(key='seqName', string='SDROInfo', val='SDRO')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='targetAmplitude', string='Target amplitude (mV)', val=90.0, field='IM',
                          tip='Target echo amplitude. Default 90 mV, based on measured ACR phantom signal '
                              '(~25 mV unattenuated, saturating at 136 mV with SDRO at no attenuation).')
        self.addParameter(key='gainTolerance', string='Accuracy (+/- mV)', val=10.0, field='IM',
                          tip='Accuracy band around the target amplitude. If echo is already within '
            'target +/- this value, auto mode applies no correction.')
        self.addParameter(key='mode', string='Mode (auto/manual)', val='auto', field='IM',
                          tip='auto: one-shot gain from Larmor echo. '
                              'manual: user sets gain directly.')
        self.addParameter(key='manualGain', string='Manual gain (dB)', val=0, field='IM',
                          tip='Only used in manual mode. Valid range: -5 to 26 dB.')

    def sequenceInfo(self):
        print("Signal Dynamic Range Optimization (SDRO)")
        print("Author: MRILab, i3M, CSIC, Valencia, Spain")
        print("Controls RF gain via Arduino attenuator + 26 dB amplifier.")
        print("auto:       one-shot deterministic gain from Larmor echo amplitude.")
        print("manual:     user inputs gain directly.\n")

    def sequenceTime(self):
        return 0  # minutes

    def sequenceRun(self, plotSeq=0, demo=False):
        self.demo = demo

        # Update target amplitude and accuracy band from GUI
        hw.targetAmplitude = self.mapVals['targetAmplitude']
        hw.gainTolerance = self.mapVals['gainTolerance']

        if self.sdro.arduino.device is None:
            print("WARNING: No Arduino found for SDRO.")
            return False

        # Set initial gain to 0 dB on first run
        if self.sdro.current_gain is None:
            self.sdro.set_gain(0)

        # Always print suggested gain so user knows what manual value to use
        if hw.echoAmplitude > 0.0:
            self.sdro.suggest_gain()

        # Run in selected mode
        mode = self.mapVals['mode']

        if mode == 'manual':
            self.sdro.adjust_gain(manual_gain=self.manualGain, mode='manual')

        elif mode == 'auto':
            if hw.echoAmplitude > 0.0:
                self.sdro.adjust_gain(mode='auto')
            else:
                print("WARNING: hw.echoAmplitude is 0. Run Larmor first or switch to manual mode.")
                return False
        else:
            print("ERROR: mode must be auto or manual.")
            self.sdro.disconnect()  # always close before returning
            return False

        # Read back and confirm gain
        current_gain = self.sdro.get_gain()
        self.mapVals['current_gain'] = current_gain
        hw.sdrolnaGain = current_gain

        return True




































































#
#
# """
# SDRO sequence file for MaRGE.
# Controls RF gain/attenuation via Arduino-controlled attenuator + 26 dB amplifier.
#
# Two modes available via MaRGE GUI:
#     - auto:       one-shot deterministic gain from Larmor echo amplitude
#     - manual:     user inputs gain directly (use when signal saturates)
#
# Author: MRILab, i3M, CSIC, Valencia, Spain
# """
#
# import marge.seq.mriBlankSeq as blankSeq
# import marge.configs.hw_config as hw
# from marge.controller.controller_sdro import SDRO
#
#
# class SDROSeq(blankSeq.MRIBLANKSEQ):
#     def __init__(self):
#         super(SDROSeq, self).__init__()
#
#         self.sdro = None
#
#         # Parameters visible in MaRGE GUI
#         self.addParameter(key='seqName', string='SDROInfo', val='SDRO')
#         self.addParameter(key='toMaRGE', val=True)
#         self.addParameter(key='targetAmplitude', string='Target amplitude (mV)', val=125.0, field='IM',
#                           tip='Target echo amplitude. Default 125 mV = 50% of ADC range (250 mV).')
#         self.addParameter(key='mode', string='Mode (auto/manual)', val='auto', field='IM',
#                           tip='auto: one-shot gain from Larmor echo. '
#                               'manual: user sets gain directly.')
#         self.addParameter(key='manualGain', string='Manual gain (dB)', val=0, field='IM',
#                           tip='Only used in manual mode. Valid range: -5 to 26 dB.')
#
#     def sequenceInfo(self):
#         print("Signal Dynamic Range Optimization (SDRO)")
#         print("Author: MRILab, i3M, CSIC, Valencia, Spain")
#         print("Controls RF gain via Arduino attenuator + 26 dB amplifier.")
#         print("auto:       one-shot deterministic gain from Larmor echo amplitude.")
#         print("manual:     user inputs gain directly.\n")
#
#     def sequenceTime(self):
#         return 0  # minutes
#
#     def sequenceRun(self, plotSeq=0, demo=False):
#         self.demo = demo
#
#         # Update target amplitude from GUI
#         hw.targetAmplitude = self.mapVals['targetAmplitude']
#
#         if self.sdro.arduino.device is None:
#             print("WARNING: No Arduino found for SDRO.")
#             return False
#
#         # Set initial gain to 0 dB on first run
#         if self.sdro.current_gain is None:
#             print("Setting initial gain to 0 dB...")
#             self.sdro.set_gain(0)
#
#         # Always print suggested gain so user knows what manual value to use
#         if hw.echoAmplitude > 0.0:
#             self.sdro.suggest_gain()
#
#         # Run in selected mode
#         mode = self.mapVals['mode']
#
#         if mode == 'manual':
#             manual_gain = int(self.mapVals['manualGain'])
#             print(f"Manual mode selected: gain = {manual_gain} dB")
#             self.sdro.adjust_gain(manual_gain=manual_gain, mode='manual')
#
#         elif mode == 'auto':
#             if hw.echoAmplitude > 0.0:
#                 print(f"Auto mode: echo = {hw.echoAmplitude:.3f} mV")
#                 self.sdro.adjust_gain(mode='auto')
#             else:
#                 print("WARNING: hw.echoAmplitude is 0. Run Larmor first or switch to manual mode.")
#                 return False
#         else:
#             print("ERROR: mode must be auto or manual.")
#             self.sdro.disconnect()  # always close before returning
#             return False
#
#         # Read back and confirm gain
#         current_gain = self.sdro.get_gain()
#         self.mapVals['current_gain'] = current_gain
#
#         return True
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

#
# """
# SDRO sequence file for MaRGE.
# Controls RF gain/attenuation via Arduino-controlled attenuator + 26 dB amplifier.
#
# Three modes available via MaRGE GUI:
#     - auto:       one-shot deterministic gain from Larmor echo amplitude
#     - manual:     user inputs gain directly (use when signal saturates)
#     - iterative:  steps down gain until safe, then applies deterministic correction
#                   (use when ADC is clipping and real amplitude is unknown)
#
# Author: MRILab, i3M, CSIC, Valencia, Spain
# """
#
# import marge.seq.mriBlankSeq as blankSeq
# import marge.configs.hw_config as hw
# from marge.controller.controller_sdro import SDRO
#
#
# class SDROSeq(blankSeq.MRIBLANKSEQ):
#     def __init__(self):
#         super(SDROSeq, self).__init__()
#
#         self.sdro = None
#
#         # Parameters visible in MaRGE GUI
#         self.addParameter(key='seqName', string='SDROInfo', val='SDRO')
#         self.addParameter(key='toMaRGE', val=True)
#         self.addParameter(key='targetAmplitude', string='Target amplitude (mV)', val=125.0, field='IM',
#                           tip='Target echo amplitude. Default 125 mV = 50% of ADC range (250 mV).')
#         self.addParameter(key='mode', string='Mode (auto/manual/iterative)', val='auto', field='IM',
#                           tip='auto: one-shot gain from Larmor echo. '
#                               'manual: user sets gain directly. '
#                               'iterative: steps down gain until safe, then corrects. '
#                               'Use iterative when ADC is clipping.')
#         self.addParameter(key='manualGain', string='Manual gain (dB)', val=0, field='IM',
#                           tip='Only used in manual mode. Valid range: -5 to 26 dB.')
#
#     def sequenceInfo(self):
#         print("Signal Dynamic Range Optimization (SDRO)")
#         print("Author: MRILab, i3M, CSIC, Valencia, Spain")
#         print("Controls RF gain via Arduino attenuator + 26 dB amplifier.")
#         print("auto:       one-shot deterministic gain from Larmor echo amplitude.")
#         print("manual:     user inputs gain directly.")
#         print("iterative:  steps down gain until safe, then applies correction.\n")
#
#     def sequenceTime(self):
#         return 0  # minutes
#
#     def sequenceRun(self, plotSeq=0, demo=False):
#         self.demo = demo
#
#         # Update target amplitude from GUI
#         hw.targetAmplitude = self.mapVals['targetAmplitude']
#
#         if self.sdro.arduino.device is None:
#             print("WARNING: No Arduino found for SDRO.")
#             return False
#
#         # Set initial gain to 0 dB on first run
#         if self.sdro.current_gain is None:
#             print("Setting initial gain to 0 dB...")
#             self.sdro.set_gain(0)
#
#         # Always print suggested gain so user knows what manual value to use
#         if hw.echoAmplitude > 0.0:
#             self.sdro.suggest_gain()
#
#         # Run in selected mode
#         mode = self.mapVals['mode']
#
#         if mode == 'manual':
#             manual_gain = int(self.mapVals['manualGain'])
#             print(f"Manual mode selected: gain = {manual_gain} dB")
#             self.sdro.adjust_gain(manual_gain=manual_gain, mode='manual')
#
#         elif mode == 'auto':
#             if hw.echoAmplitude > 0.0:
#                 print(f"Auto mode: echo = {hw.echoAmplitude:.3f} mV")
#                 self.sdro.adjust_gain(mode='auto')
#             else:
#                 print("WARNING: hw.echoAmplitude is 0. Run Larmor first or switch to manual mode.")
#                 return False
#         else:
#             print("ERROR: mode must be auto, manual, or iterative.")
#             self.sdro.disconnect()  # always close before returning
#             return False
#
#         # Read back and confirm gain
#         current_gain = self.sdro.get_gain()
#         self.mapVals['current_gain'] = current_gain
#
#         return True
#
#     # def sequenceAnalysis(self, mode=None):
#     #         """
#     #         Required by MaRGE after sequenceRun.
#     #         SDRO has no image output, so return empty results.
#     #         """
#     #         output_dict = {
#     #             'current_gain': self.mapVals.get('current_gain', None),
#     #             'targetAmplitude': self.mapVals.get('targetAmplitude', 125.0),
#     #             'echoAmplitude': hw.echoAmplitude,
    #         }
    #
    #         # No plots needed for SDRO
    #         outputs = []
    #         dicom_meta_data = {}
    #
    #         return output_dict, outputs, dicom_meta_data

    # def sequenceRun(self, plotSeq=0, demo=False):
    #     self.demo = demo
    #
    #     # Update target amplitude from GUI
    #     hw.targetAmplitude = self.mapVals['targetAmplitude']
    #
    #     # Connect to SDRO Arduino if not already connected
    #     if self.sdro is None:
    #         self.sdro = SDRO(serial_number=hw.ard_sn_sdro)
    #         self.sdro.connect()
    #
    #     if self.sdro.arduino.device is None:
    #         print("WARNING: No Arduino found for SDRO.")
    #         return False
    #
    #     # Set initial gain to 0 dB on first run
    #     if self.sdro.current_gain is None:
    #         print("Setting initial gain to 0 dB...")
    #         self.sdro.set_gain(0)
    #
    #     # Always print suggested gain so user knows what manual value to use
    #     if hw.echoAmplitude > 0.0:
    #         self.sdro.suggest_gain()
    #
    #     # Run in selected mode
    #     mode = self.mapVals['mode']
    #
    #     if mode == 'manual':
    #         manual_gain = int(self.mapVals['manualGain'])
    #         print(f"Manual mode selected: gain = {manual_gain} dB")
    #         self.sdro.adjust_gain(manual_gain=manual_gain, mode='manual')
    #
    #     elif mode == 'auto':
    #         if hw.echoAmplitude > 0.0:
    #             print(f"Auto mode: echo = {hw.echoAmplitude:.3f} mV")
    #             self.sdro.adjust_gain(mode='auto')
    #         else:
    #             print("WARNING: hw.echoAmplitude is 0. Run Larmor first or switch to manual mode.")
    #             return False
    #
    #     elif mode == 'iterative':
    #         if hw.echoAmplitude > 0.0:
    #             print(f"Iterative mode: echo = {hw.echoAmplitude:.3f} mV")
    #             self.sdro.adjust_gain(mode='iterative')
    #         else:
    #             print("WARNING: hw.echoAmplitude is 0. Run Larmor first.")
    #             return False
    #
    #     else:
    #         print("ERROR: mode must be auto, manual, or iterative.")
    #         return False
    #
    #     # Read back and confirm gain
    #     current_gain = self.sdro.get_gain()
    #     self.mapVals['current_gain'] = current_gain
    #
    #     return True





























#
#
# """
# SDRO sequence file for MaRGE.
# Controls RF gain/attenuation via Arduino-controlled attenuator + 26 dB amplifier.
#
# Two modes available via MaRGE GUI:
#     - auto:   gain computed deterministically from Larmor echo amplitude
#     - manual: user inputs gain directly (use when signal saturates)
#
# Author: MRILab, i3M, CSIC, Valencia, Spain
# """
#
# import marge.seq.mriBlankSeq as blankSeq
# import marge.configs.hw_config as hw
# from marge.controller.SDRO import SDRO
#
#
# class SDROSeq(blankSeq.MRIBLANKSEQ):
#     def __init__(self):
#         super(SDROSeq, self).__init__()
#
#         self.sdro = None
#
#         # Parameters visible in MaRGE GUI
#         self.addParameter(key='seqName', string='SDROInfo', val='SDRO')
#         self.addParameter(key='toMaRGE', val=True)
#         self.addParameter(key='targetAmplitude', string='Target amplitude (mV)', val=125.0, field='IM',
#                           tip='Target echo amplitude. Default 125 mV = 50% of ADC range (250 mV).')
#         self.addParameter(key='mode', string='Mode (auto/manual)', val='auto', field='IM',
#                           tip='auto: gain computed from Larmor echo amplitude. '
#                               'manual: user sets gain directly. Use manual when signal saturates.')
#         self.addParameter(key='manualGain', string='Manual gain (dB)', val=0, field='IM',
#                           tip='Only used when mode is set to manual. Valid range: -5 to 26 dB.')
#
#         # Connect to SDRO Arduino if not already connected
#         if self.sdro is None:
#             self.sdro = SDRO(serial_number=hw.ard_sn_sdro)
#             self.sdro.connect()
#
#     def sequenceInfo(self):
#         print("Signal Dynamic Range Optimization (SDRO)")
#         print("Author: MRILab, i3M, CSIC, Valencia, Spain")
#         print("Controls RF gain via Arduino attenuator + 26 dB amplifier.")
#         print("Auto mode: gain set deterministically from Larmor echo amplitude.")
#         print("Manual mode: user inputs gain directly.\n")
#
#     def sequenceTime(self):
#         return 0  # minutes
#
#     def sequenceRun(self, plotSeq=0, demo=False):
#         self.demo = demo
#
#         # Update target amplitude from GUI
#         hw.targetAmplitude = self.mapVals['targetAmplitude']
#
#         if self.sdro.arduino.device is None:
#             print("WARNING: No Arduino found for SDRO.")
#             return False
#
#         # Set initial gain to 0 dB on first run
#         if self.sdro.current_gain is None:
#             print("Setting initial gain to 0 dB...")
#             self.sdro.set_gain(0)
#
#         # Run in selected mode
#         mode = self.mapVals['mode']
#
#         if mode == 'manual':
#             manual_gain = int(self.mapVals['manualGain'])
#             print(f"Manual mode selected: gain = {manual_gain} dB")
#             self.sdro.adjust_gain(manual_gain=manual_gain)
#
#         elif mode == 'auto':
#             if hw.echoAmplitude > 0.0:
#                 print(f"Auto mode: echo amplitude = {hw.echoAmplitude:.3f} mV")
#                 self.sdro.adjust_gain()
#             else:
#                 print("WARNING: hw.echoAmplitude is 0. Run Larmor first or switch to manual mode.")
#                 return False
#
#         else:
#             print("ERROR: mode must be 'auto' or 'manual'.")
#             return False
#
#         # Read back and confirm gain
#         current_gain = self.sdro.get_gain()
#         self.mapVals['current_gain'] = current_gain
#
#         return True
#

















































#
# """
# SDRO sequence file for MaRGE.
# Controls RF gain/attenuation via Arduino-controlled attenuator + 26 dB amplifier.
# Author: MRILab, i3M, CSIC, Valencia
# """
#
# import numpy as np
# import marge.seq.mriBlankSeq as blankSeq
# import marge.configs.hw_config as hw
# import marge.configs.units as units
# from marge.controller.SDRO import SDRO
#
#
# class SDROSeq(blankSeq.MRIBLANKSEQ):
#     def __init__(self):
#         super(SDROSeq, self).__init__()
#
#         self.sdro = None
#
#         # Parameters visible in MaRGE GUI
#         self.addParameter(key='seqName', string='SDROInfo', val='SDRO')
#         self.addParameter(key='toMaRGE', val=True)
#         self.addParameter(key='targetAmplitude', string='Target amplitude (mV)', val=125.0, field='IM')
#
#     def sequenceInfo(self):
#         print("Signal Dynamic Range Optimization (SDRO)")
#         print("Author: MRILab, i3M, CSIC, Valencia")
#         print("Controls RF gain via Arduino attenuator + 26 dB amplifier.")
#         print("Gain is set automatically based on Larmor echo amplitude.\n")
#
#     def sequenceTime(self):
#         return 0  # minutes
#
#     def sequenceRun(self, plotSeq=0, demo=False):
#         self.demo = demo
#
#         # Update target amplitude from GUI parameter
#         hw.targetAmplitude = self.mapVals['targetAmplitude']
#
#         # Connect to SDRO Arduino if not already connected
#         if self.sdro is None:
#             self.sdro = SDRO(serial_number=hw.ard_sn_sdro)
#             self.sdro.connect()
#
#         if self.sdro.arduino.device is None:
#             print("WARNING: No Arduino found for SDRO.")
#             return False
#
#         # Set initial gain to 0 dB before Larmor runs
#         if self.sdro.current_gain is None:
#             print("Setting initial gain to 0 dB...")
#             self.sdro.set_gain(0)
#
#         # If Larmor has already run and updated hw.echoAmplitude, adjust gain
#         if hw.echoAmplitude > 0.0:
#             print(f"Echo amplitude from Larmor: {hw.echoAmplitude:.3f} mV")
#             self.sdro.adjust_gain()
#         else:
#             print("WARNING: hw.echoAmplitude is 0. Run Larmor sequence first.")
#
#         # Read back and confirm current gain
#         current_gain = self.sdro.get_gain()
#         self.mapVals['current_gain'] = current_gain
#
#         return True

if __name__ == '__main__':
    import marge.controller.controller_sdro as sdro
    sdro = sdro.SDRO(serial_number=hw.ard_sn_sdro)
    sdro.connect()
    seq = SDROSeq()
    seq.sdro = sdro
    seq.sequenceAtributes()
    seq.sequenceRun(demo=True)
    seq.sequenceAnalysis(mode='Standalone')
    sdro.disconnect()