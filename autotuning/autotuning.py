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

import configs.autotuning as autotuning
import time

from vna import Hardware


class Arduino:
    def __init__(self, baudrate=115200, timeout=0.1):
        self.device = None
        self.serial = None
        self.port = self.findPort()
        self.baudrate = baudrate
        self.timeout = timeout

    def findPort(self):
        arduino_port = [
            p.device
            for p in serial.tools.list_ports.comports()
            if autotuning.serial_number in p.serial_number
        ]

        if not arduino_port:
            print("\nNo Arduino found for auto-tuning.")
            return False
        else:
            return arduino_port[0]

    def connect(self):
        if not self.device:
            self.device = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)
            print("\nConnected to Arduino for auto-tuning")
            time.sleep(1.0)

    def disconnect(self):
        if self.device is not None:
            self.device.close()
            print("\nDisconnected from Arduino for auto-tuning")
            self.device = None

    def send(self, data):
        if self.device is not None:
            self.device.write(data.encode())
        self.receive()

    def receive(self):
        if self.device is not None:
            while self.device.in_waiting == 0:
                pass
            return self.device.readline()


class VNA:

    def __init__(self):
        self.frequencies = None
        self.vna = None
        self.interface = None
        self.data = []

    def connect(self):
        self.interface = Hardware.get_interfaces()[0]
        self.interface.open()
        self.interface.timeout = 0.05
        time.sleep(0.1)
        self.vna = Hardware.get_VNA(self.interface)
        self.frequencies = np.array(self.vna.readFrequencies()) * 1e-6  # MHz
        print("\nConnected to nanoVNA for auto-tuning")

    def getFrequency(self):
        return self.frequencies

    def getData(self):
        return self.data

    def getS11(self, f0=None):
        self.data = []
        for value in self.vna.readValues("data 0"):
            self.data.append(float(value.split(" ")[0]) + 1j * float(value.split(" ")[1]))

        # Create a linear interpolation function
        interp_func = interp1d(self.frequencies, self.data, kind='cubic')

        # Perform interpolation
        s11 = interp_func(f0)
        z11 = 50 * (1 + s11) / (1 - s11)

        # # Generate points for plotting
        # x_interp = np.linspace(np.min(self.frequencies), np.max(self.frequencies), 1000)
        # y_interp = interp_func(x_interp)
        #
        # plt.plot(self.frequencies, np.real(data), 'o', label='Original Data')
        # plt.plot(x_interp, np.real(y_interp), label='Interpolation')
        # plt.plot(f0, np.real(s11), 'ro', label='Interpolated')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('Linear Interpolation')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        return s11, z11


if __name__ == "__main__":
    device = VNA()
    device.connect()
    s11, z11 = device.getS11(2.9713)
    print(s11)
    print(z11)

    # # Create an instance of the Arduino class and connect to an Arduino
    # arduino = Arduino()
    #
    # # Disconnect from the Arduino
    # arduino.connect()
    # arduino.disconnect()
