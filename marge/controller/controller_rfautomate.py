"""
Created on Thu August 17th 2023
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: code to communicate with arduino for autotuning
Specific hardware from MRILab @ i3M is required
"""

import numpy as np
from scipy.interpolate import interp1d

import time

from marge.vna import Hardware
from marge.utils.SerialDevice import SerialDevice


class VNA:
    def __init__(self):
        """
        Initialize a Vectorial Network Analyzer (VNA) object.
        """
        self.connected = None
        self.frequencies = None
        self.interface = None
        self.data = []
        self.device = None

    def connect(self):
        """
        Connect to the nanoVNA device.

        :return: True if connected successfully, otherwise False.
        """
        if not self.device:
            try:
                self.interface = Hardware.get_interfaces()[0]
                self.interface.open()
                self.interface.timeout = 0.05
                time.sleep(0.1)
                self.device = Hardware.get_VNA(self.interface)
                self.frequencies = np.array(self.device.readFrequencies()) * 1e-6  # MHz
                print("Connected to nanoVNA for auto-tuning")
                return True
            except IndexError:
                print("WARNING: No interfaces available for nanoVNA auto-tuning")
                return False
            except Exception as e:
                print(f"WARNING: Failed to connect to nanoVNA for auto-tuning: {e}")
                return False
        else:
            return True

    def getFrequency(self):
        """
        Get the array of frequencies at which measurements were taken.

        :return: Array of frequencies in MHz.
        """
        if self.device is not None:
            return self.frequencies

    def getData(self):
        """
        Get the measurement data.

        :return: List of complex measurement data.
        """
        if self.device is not None:
            return self.data

    def getS11(self, f0=None):
        """
        Get S11 parameter and impedance for a specific frequency.

        :param f0: Frequency at which to get S11 parameter (in MHz).
        :return: Tuple containing S11 parameter and impedance at the given frequency.
        """
        if self.device is not None:
            self.data = []
            for value in self.device.readValues("data 0"):
                self.data.append(float(value.split(" ")[0]) + 1j * float(value.split(" ")[1]))

            # Create a linear interpolation function
            interp_func = interp1d(self.frequencies, self.data, kind='cubic')

            # Perform interpolation
            s11 = interp_func(f0)
            z11 = 50 * (1 + s11) / (1 - s11)

            return s11, z11

        return None, None



if __name__ == "__main__":
    # device = VNA()
    # device.connect()
    # s11, z11 = device.getS11(2.9713)
    # print(s11)
    # print(z11)

    import random

    # Create an instance of the SerialDevice class and connect to an Arduino
    arduino = SerialDevice()
    arduino.connect(port='serial:44234313434351416122')
    n = 0
    while True:
        binary_string = ''.join(random.choice('01') for _ in range(17))
        # Pad to 32 characters with '0' to match original Arduino.send() behavior
        padded_string = binary_string.ljust(32, '0')
        result = arduino.send(padded_string)
        n += 1
        print(f"Iteration {n}")

    # arduino.disconnect()
