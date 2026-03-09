import time
from threading import RLock

import serial
import serial.tools.list_ports


class SerialDevice:
    def __init__(self, baudrate=115200, timeout=0.1, startup_delay=1.0, announce=True):
        """
        Initialize a SerialDevice object.

        :param baudrate: Baud rate for communication (default is 115200).
        :param timeout: Timeout for communication operations (default is 0.1 seconds).
        :param startup_delay: Delay after opening the port to allow device reset.
        """
        self.device = None
        self.serial = None
        self.port = None
        self.baudrate = baudrate
        self._timeout = timeout
        self.startup_delay = startup_delay
        self.serial_number = None
        self.announce = announce
        self.lock = RLock()

    def findPort(self):
        """
        Find the port of the connected serial device by serial number.

        :return: The port of the device if found, otherwise False.
        """
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if port.serial_number == self.serial_number:
                return port.device

        return False

    def _resolve_connection_string(self, port):
        if not port:
            return None

        port_str = str(port).strip()
        if port_str.lower().startswith("serial:"):
            self.serial_number = port_str[7:].strip()
            found = self.findPort()
            if not found:
                print(f"WARNING: Device with serial number {self.serial_number} not found")
                return None
            return found

        return port_str

    def connect(self, port=None):
        """
        Connect to the serial device.

        The port argument is a single string; the connection type is determined automatically:
        - serial:SERIAL_NUMBER  — find device by serial number (e.g. serial:55731323736351611260)
        - socket://host:port     — TCP socket (e.g. socket://192.168.1.100:5000)
        - rfc2217://host:port    — RFC2217 serial over network
        - /dev/ttyUSB0, COM3     — direct serial port path (Linux or Windows)

        :param port: Connection specifier (serial:..., URL, or direct port path).
        :return: True if connected successfully, otherwise False.
        """
        if self.is_open:
            return True

        if port is None:
            port = self.port

        if not port:
            print("WARNING: No port specified. Use e.g. port='serial:55731323736351611260', port='/dev/ttyUSB0', or port='socket://host:5000'.")
            return False

        connection_string = self._resolve_connection_string(port)
        if not connection_string:
            return False

        try:
            self.device = serial.serial_for_url(connection_string, baudrate=self.baudrate, timeout=self._timeout)
            self.serial = self.device
            self.port = connection_string
            if self.announce:
                print(f"Connected to serial device at {connection_string}")
            time.sleep(self.startup_delay)
            self.reset_buffers()
            return True
        except serial.SerialException as e:
            print(f"WARNING: Failed to connect to serial device at {connection_string}: {e}")
            self.device = None
            self.serial = None
            return False
        except Exception as e:
            print(f"WARNING: Unexpected error connecting to serial device: {e}")
            self.device = None
            self.serial = None
            return False

    def disconnect(self):
        """
        Disconnect from the serial device.
        """
        if self.device is not None:
            try:
                self.device.close()
                if self.announce:
                    print("Disconnected from serial device")
            except Exception as e:
                print(f"WARNING: Error during disconnect: {e}")
            finally:
                self.device = None
                self.serial = None

    def open(self):
        return self.connect(self.port)

    def close(self):
        self.disconnect()

    def send(self, data, deadline_seconds=5.0):
        """
        Send data to the connected serial device and wait for a response.

        :param data: The data to be sent.
        """
        if self.device is None:
            return False

        output = False
        with self.lock:
            while output is False:
                self.write(data)
                output = self.receive(deadline_seconds=deadline_seconds)
                if output is False:
                    print("WARNING: Serial communication failed...")
                    print("Retrying...")
        return output

    def receive(self, deadline_seconds=5.0):
        """
        Receive data from the connected serial device.

        :return: The received data.
        """
        if self.device is None:
            return "False".encode('utf-8')
        response = self.read_line(deadline_seconds=deadline_seconds)
        if response is False:
            print("Failed to get data from serial device...")
        return response

    def reset_buffers(self):
        """
        Reset the device input and output buffers.
        """
        if self.device is not None:
            self.device.reset_input_buffer()
            self.device.reset_output_buffer()

    def reset_input_buffer(self):
        if self.device is not None:
            self.device.reset_input_buffer()

    def reset_output_buffer(self):
        if self.device is not None:
            self.device.reset_output_buffer()

    def write(self, data):
        """
        Write raw bytes or encoded text to the device.
        """
        if self.device is None:
            return False

        payload = data if isinstance(data, bytes) else str(data).encode("ascii")
        with self.lock:
            self.device.write(payload)
            self.device.flush()
        return True

    def read(self, size=1):
        if self.device is None:
            return b""
        with self.lock:
            return self.device.read(size)

    def readline(self):
        if self.device is None:
            return b""
        with self.lock:
            return self.device.readline()

    def write_line(self, line):
        """
        Write a single newline-terminated ASCII line to the device.
        """
        return self.write(f"{line}\n")

    def read_line(self, deadline_seconds=5.0):
        """
        Read a single line from the device until the deadline expires.
        """
        if self.device is None:
            return False

        deadline = time.time() + deadline_seconds
        while time.time() < deadline:
            raw = self.readline()
            if raw:
                return raw
        return False

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._timeout = value
        if self.device is not None:
            self.device.timeout = value

    @property
    def is_open(self):
        return self.device is not None and getattr(self.device, "is_open", False)

    @property
    def in_waiting(self):
        if self.device is None:
            return 0
        return self.device.in_waiting

    @property
    def fd(self):
        if self.device is None:
            raise AttributeError("Serial device is not open")
        if hasattr(self.device, "fd"):
            return self.device.fd
        return self.device.fileno()
