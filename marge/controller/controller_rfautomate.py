"""
Created on Thu August 17th 2023
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: code to communicate with arduino for autotuning
Specific hardware from MRILab @ i3M is required
"""

from marge.seq.AutoTuning.AutoTuningHardwareInterface import Arduino as BaseArduino
from marge.seq.AutoTuning.AutoTuningHardwareInterface import VNA


class Arduino(BaseArduino):
    def __init__(self, baudrate=115200, timeout=0.1, name='test'):
        super().__init__(
            baudrate=baudrate,
            timeout=timeout,
            name=name,
            receive_timeout=2.0,
            pad_to_length=32,
            clear_input_on_receive=False,
        )


if __name__ == "__main__":
    import random

    arduino = Arduino()
    arduino.connect(serial_number='44234313434351416122')
    n = 0
    while True:
        binary_string = ''.join(random.choice('01') for _ in range(17))
        arduino.send(binary_string)
        n += 1
        print(f"Iteration {n}")
