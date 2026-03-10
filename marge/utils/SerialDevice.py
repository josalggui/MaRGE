import time

import serial
import serial.tools.list_ports


class SerialDevice:
    def __init__(
        self,
        connection="",
        baudrate=115200,
        timeout=0.1,
        startup_delay=1.0,
        name="Serial device",
        pad_to_length=None,
        receive_timeout=5.0,
    ):
        self.connection = connection
        self.default_baudrate = baudrate
        self.baudrate = baudrate
        self.timeout = timeout
        self.startup_delay = startup_delay
        self.name = name
        self.pad_to_length = pad_to_length
        self.receive_timeout = receive_timeout

        self.device = None
        self.serial = None
        self.port = None
        self.serial_number = None

    def _parse_connection_spec(self, connection=None):
        connection_spec = str(connection if connection is not None else self.connection).strip()
        baudrate = self.default_baudrate

        if "@" in connection_spec:
            candidate_spec, candidate_baudrate = connection_spec.rsplit("@", 1)
            candidate_baudrate = candidate_baudrate.strip()
            if candidate_baudrate.isdigit():
                connection_spec = candidate_spec.strip()
                baudrate = int(candidate_baudrate)

        if connection_spec.lower().startswith("serial:"):
            self.serial_number = connection_spec[7:].strip()
            target_port = self.find_port(self.serial_number)
        elif (
            "://" in connection_spec
            or connection_spec.startswith("/")
            or connection_spec.startswith("\\\\.\\")
            or connection_spec.upper().startswith("COM")
        ):
            target_port = connection_spec
        else:
            self.serial_number = connection_spec
            target_port = self.find_port(self.serial_number)

        return target_port, baudrate

    def find_port(self, serial_number=None):
        serial_number = self.serial_number if serial_number is None else str(serial_number).strip()
        if not serial_number:
            return False

        for port in serial.tools.list_ports.comports():
            if port.serial_number == serial_number:
                return port.device

        return False

    def connect(self, connection=None):
        if self.device is not None:
            return True

        self.connection = self.connection if connection is None else connection
        target_port, baudrate = self._parse_connection_spec(self.connection)
        self.port = target_port
        if not self.port:
            print(f"WARNING: No serial device found for {self.name}")
            return False

        self.baudrate = baudrate
        self.device = serial.serial_for_url(self.port, baudrate=self.baudrate, timeout=self.timeout)
        self.serial = self.device
        print(f"Connected to {self.name}")
        time.sleep(self.startup_delay)
        return True

    def disconnect(self):
        if self.device is not None:
            self.device.close()
            print(f"Disconnected from {self.name}")
            self.device = None
            self.serial = None

    def send(self, data, deadline_seconds=None):
        output = False
        if self.device is not None:
            payload = str(data)
            if self.pad_to_length is not None:
                payload = payload.ljust(self.pad_to_length, "0")

            while output is False:
                self.device.write(payload.encode())
                output = self.receive(deadline_seconds=deadline_seconds)
                if output is False:
                    print(f"WARNING: {self.name} communication failed...")
                    print("Retrying...")
        return output

    def receive(self, deadline_seconds=None):
        if self.device is not None:
            t0 = time.time()
            timeout_seconds = self.receive_timeout if deadline_seconds is None else deadline_seconds
            while self.device.in_waiting == 0 and time.time() - t0 < timeout_seconds:
                time.sleep(0.01)

            if time.time() - t0 >= timeout_seconds:
                print(f"Failed to get data from {self.name}...")
                return False
            return self.device.readline()

        return "False".encode("utf-8")
