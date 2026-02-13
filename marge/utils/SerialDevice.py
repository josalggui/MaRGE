import serial
import serial.tools.list_ports
import time


class SerialDevice:
    def __init__(self, baudrate=115200, timeout=0.1):
        """
        Initialize a SerialDevice object.

        :param baudrate: Baud rate for communication (default is 115200).
        :param timeout: Timeout for communication operations (default is 0.1 seconds).
        """
        self.device = None
        self.serial = None
        self.port = None
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_number = None

    def findPort(self):
        """
        Find the port of the connected Arduino by serial number.

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

    def connect(self, port):
        """
        Connect to the Arduino.

        The port argument is a single string; the connection type is determined automatically:
        - serial:SERIAL_NUMBER  — find device by serial number (e.g. serial:55731323736351611260)
        - socket://host:port     — TCP socket (e.g. socket://192.168.1.100:5000)
        - rfc2217://host:port    — RFC2217 serial over network
        - /dev/ttyUSB0, COM3     — direct serial port path (Linux or Windows)

        :param port: Connection specifier (serial:..., URL, or direct port path).
        :return: True if connected successfully, otherwise False.
        """
        if self.device is not None:
            return True  # Already connected

        if not port:
            print("WARNING: No port specified. Use e.g. port='serial:55731323736351611260', port='/dev/ttyUSB0', or port='socket://host:5000'.")
            return False

        connection_string = None
        port_str = str(port).strip()

        # Serial number: "serial:55731323736351611260" or legacy "serial: 5573..."
        if port_str.lower().startswith("serial:"):
            self.serial_number = port_str[7:].strip()
            found = self.findPort()
            if not found:
                print(f"WARNING: Arduino with serial number {self.serial_number} not found")
                return False
            connection_string = found
        # URL schemes (socket://, rfc2217://, etc.)
        elif "://" in port_str:
            connection_string = port_str
        # Direct port path (e.g. /dev/ttyUSB0, COM3)
        else:
            connection_string = port_str

        try:
            self.device = serial.serial_for_url(
                connection_string,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.port = connection_string
            print(f"Connected to Arduino at {connection_string}")
            time.sleep(1.0)
            return True
        except serial.SerialException as e:
            print(f"WARNING: Failed to connect to Arduino at {connection_string}: {e}")
            self.device = None
            return False
        except Exception as e:
            print(f"WARNING: Unexpected error connecting to Arduino: {e}")
            self.device = None
            return False

    def disconnect(self):
        """
        Disconnect from the Arduino.
        """
        if self.device is not None:
            try:
                self.device.close()
                print("Disconnected from Arduino")
            except Exception as e:
                print(f"WARNING: Error during disconnect: {e}")
            finally:
                self.device = None
                self.port = None

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