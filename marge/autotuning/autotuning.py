"""
Created on Thu August 17th 2023
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: code to communicate with arduino for autotuning
Specific hardware from MRILab @ i3M is required

Note: Serial device communication (previously Arduino class) has been moved to
marge.utils.SerialDevice.SerialDevice. This module now only contains the VNA class.
"""
import threading

import numpy as np
import serial.tools.list_ports
import serial
from scipy.interpolate import interp1d

import time

from marge.vna import Hardware





class VNA:
    def __init__(self):
        """
        Initialize a Vectorial Network Analyzer (VNA) object.
        """
        self.error = None
        self.reading = True
        self.connected = None
        self.frequencies = None
        self.interface = None
        self.data = []
        self.device = None
        self.timeout = 5

    def _get_frequency(self):
        try:
            self.frequencies = np.array(self.device.readFrequencies()) * 1e-6
        except Exception as e:
            self.error = e
        self.reading = False

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

                # Initialize frequency reading
                self.reading = True
                self.error = None
                thread = threading.Thread(target=self._get_frequency)
                thread.start()
                t0 = time.time()
                t1 = time.time()
                while self.reading and t1 - t0 < self.timeout:
                    t1 = time.time()
                    time.sleep(0.01)
                if t1 - t0 >= self.timeout:
                    print("WARNING: nanoVNA timeout reached!")
                    return False
                else:
                    if self.error is None:
                        print("Connected to nanoVNA")
                        return True
                    else:
                        print(f"WARNING: Failed to connect to nanoVNA: {self.error}")
                        return False
            except IndexError:
                print("WARNING: No interfaces available for nanoVNA")
                return False
            except Exception as e:
                print(f"WARNING: Failed to connect to nanoVNA: {e}")
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

    def _get_s11(self):
        try:
            self.data_prov = self.device.readValues("data 0")
        except Exception as e:
            self.error = e
        self.reading = False

    def getS11(self, f0=None):
        """
        Get S11 parameter and impedance for a specific frequency.

        :param f0: Frequency at which to get S11 parameter (in MHz).
        :return: Tuple containing S11 parameter and impedance at the given frequency.
        """
        if self.device is not None:
            # Initialize data reading
            self.reading = True
            self.error = None
            thread = threading.Thread(target=self._get_s11)
            thread.start()
            t0 = time.time()
            t1 = time.time()
            while self.reading and t1 - t0 < self.timeout:
                t1 = time.time()
                time.sleep(0.01)
            if t1 - t0 >= self.timeout:
                print("WARNING: nanoVNA timeout reached!")
                raise IOError("WARNING: nanoVNA timeout reached!")
            else:
                if self.error is None:
                    self.data = []
                    for value in self.data_prov:
                        self.data.append(float(value.split(" ")[0]) + 1j * float(value.split(" ")[1]))

                    # Create a linear interpolation function
                    interp_func = interp1d(self.frequencies, self.data, kind='cubic')

                    # Perform interpolation
                    s11 = interp_func(f0)
                    z11 = 50 * (1 + s11) / (1 - s11)

                    return s11, z11
                else:
                    print(f"WARNING: Failed to connect to nanoVNA: {self.error}")
                    raise IOError(self.error)


if __name__ == "__main__":
    # # Test arduino
    # arduino = Arduino(baudrate=115200, name="interlock")
    # arduino.connect(port="serial:55731323736351611260")
    #
    # string = arduino.send("GPA_SPC:CTL 1;").decode()
    # string = arduino.send("GPA_ERRST;").decode()

    # Test nanoVNA
    nanovna = VNA()
    nanovna.connect()
    try:
        a, b = nanovna.getS11(3.64)
        print(a)
    except Exception as e:
        print(e)
