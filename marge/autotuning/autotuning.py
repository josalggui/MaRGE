"""
Created on Thu August 17th 2023
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: code to communicate with arduino for autotuning
Specific hardware from MRILab @ i3M is required
"""

import numpy as np
import serial.tools.list_ports
import serial
from scipy.interpolate import interp1d

import time

from marge.vna import Hardware


class Arduino:
    def __init__(self, baudrate=115200, timeout=0.1):
        """
        Initialize an Arduino object.

        :param baudrate: Baud rate for communication (default is 115200).
        :param timeout: Timeout for communication operations (default is 0.1 seconds).
        """
        self.device = None
        self.serial = None
        self.port = None
        self.baudrate = baudrate
        self.timeout = timeout

    def findPort(self):
        """
        Find the port of the connected Arduino.

        :return: The port of the Arduino if found, otherwise False.
        """
        arduino_port = None
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if port.serial_number == self.serial_number:
                arduino_port = port.device

        if arduino_port is None:
            return False
        else:
            return arduino_port

    def connect(self, serial_number=None):
        """
        Connect to the Arduino.

        :return: True if connected successfully, otherwise False.
        """
        self.serial_number = serial_number
        if not self.device:
            self.port = self.findPort()
            if not self.port:
                return False
            else:
                self.device = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)
                print("Connected to Arduino")
                time.sleep(1.0)

    def disconnect(self):
        """
        Disconnect from the Arduino.
        """
        if self.device is not None:
            self.device.close()
            print("Disconnected from Arduino")
            self.device = None

    def send(self, data):
        """
        Send data to the Arduino.

        :param data: The data to be sent.
        """
        output = False
        if self.device is not None:
            while output == False:
                self.device.write(data.encode())
                output = self.receive()
                if output == False:
                    print("WARNING: Arduino communication failed...")
                    print("Retrying...")
        return output

    def receive(self):
        """
        Receive data from the Arduino.

        :return: The received data.
        """
        if self.device is not None:
            # Wait for data or timeout
            t0 = time.time()
            self.device.reset_input_buffer()
            while self.device.in_waiting == 0 and time.time() - t0 < 5:
                time.sleep(0.01)

            # If timeout, return False. Otherwise, return received string
            if time.time() - t0 >= 5:
                print("Failed to get data from Arduino...")
                return False
            else:
                return self.device.readline()
        else:
            return "False".encode('utf-8')


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

if __name__ == "__main__":
    # Create an instance of the Arduino to control the interlock
    arduino = Arduino(baudrate=115200, name="interlock")
    arduino.connect(serial_number="55731323736351611260")

    string = arduino.send("GPA_SPC:CTL 1;").decode()
    string = arduino.send("GPA_ERRST;").decode()
