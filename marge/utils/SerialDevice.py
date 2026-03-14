"""
Shared serial device helper for Arduino-like peripherals.
"""

import time

import serial
import serial.tools.list_ports


class SerialDevice:
    _shared_connections = {}

    def __init__(
        self,
        connection="",
        baudrate=115200,
        timeout=0.1,
        startup_delay=1.0,
        name="Serial device",
        pad_to_length=None,
        receive_timeout=5.0,
        pad_char="0",
        clear_input_on_receive=True,
    ):
        self.connection = connection
        self.default_baudrate = baudrate
        self.baudrate = baudrate
        self.timeout = timeout
        self.startup_delay = startup_delay
        self.name = name
        self.pad_to_length = pad_to_length
        self.receive_timeout = receive_timeout
        self.pad_char = pad_char
        self.clear_input_on_receive = clear_input_on_receive

        self.device = None
        self.serial = None
        self.port = None
        self.serial_number = None
        self._shared_key = None

    def find_port(self, serial_number=None):
        serial_number = self.serial_number if serial_number is None else str(serial_number).strip()
        if not serial_number:
            return False

        for port in serial.tools.list_ports.comports():
            if port.serial_number == serial_number:
                return port.device

        return False

    def _parse_connection_spec(self, connection=None, serial_number=None):
        if serial_number is not None:
            connection_spec = f"serial:{serial_number}"
        else:
            connection_spec = connection if connection is not None else self.connection

        connection_spec = str(connection_spec).strip() if connection_spec is not None else ""
        baudrate = self.default_baudrate

        if "@" in connection_spec:
            candidate_spec, candidate_baudrate = connection_spec.rsplit("@", 1)
            candidate_baudrate = candidate_baudrate.strip()
            if candidate_baudrate.isdigit():
                connection_spec = candidate_spec.strip()
                baudrate = int(candidate_baudrate)

        if not connection_spec:
            return None, baudrate, None

        resolved_serial_number = None
        if connection_spec.lower().startswith("serial:"):
            resolved_serial_number = connection_spec[7:].strip()
            target_port = self.find_port(resolved_serial_number)
        elif (
            connection_spec.startswith(("socket://", "rfc2217://", "loop://"))
            or connection_spec.startswith("/")
            or connection_spec.startswith("\\\\.\\")
            or "\\" in connection_spec
            or connection_spec.upper().startswith("COM")
        ):
            target_port = connection_spec
        else:
            resolved_serial_number = connection_spec
            target_port = self.find_port(resolved_serial_number)

        return target_port, baudrate, resolved_serial_number

    def connect(self, connection=None, serial_number=None):
        if self.device is not None:
            return True

        if connection is not None:
            self.connection = connection
        if serial_number is not None:
            self.serial_number = str(serial_number).strip()

        self.port, self.baudrate, resolved_serial_number = self._parse_connection_spec(
            connection=connection,
            serial_number=serial_number,
        )
        if resolved_serial_number:
            self.serial_number = resolved_serial_number

        if not self.port:
            print(f"WARNING: No serial device found for {self.name}")
            return False

        self._shared_key = str(self.port)
        shared_entry = self._shared_connections.get(self._shared_key)
        if shared_entry is not None:
            shared_device = shared_entry["device"]
            if shared_device is not None and getattr(shared_device, "is_open", True):
                if shared_entry["baudrate"] != self.baudrate:
                    print(
                        f"WARNING: Reusing {self.name} at {shared_entry['baudrate']} baud "
                        f"instead of requested {self.baudrate}"
                    )
                self.device = shared_device
                self.serial = shared_device
                shared_entry["refcount"] += 1
                print(f"Reusing {self.name}")
                return True
            self._shared_connections.pop(self._shared_key, None)

        try:
            self.device = serial.serial_for_url(
                self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            self.serial = self.device
            self._shared_connections[self._shared_key] = {
                "baudrate": self.baudrate,
                "device": self.device,
                "refcount": 1,
            }
            print(f"Connected to {self.name}")
            time.sleep(self.startup_delay)
            return True
        except Exception as exc:
            print(f"WARNING: Failed to connect to {self.name}: {exc}")
            self.device = None
            self.serial = None
            self._shared_key = None
            return False

    def disconnect(self):
        if self.device is None:
            return

        shared_entry = self._shared_connections.get(self._shared_key)
        if shared_entry is not None and shared_entry["device"] is self.device:
            shared_entry["refcount"] -= 1
            if shared_entry["refcount"] <= 0:
                self.device.close()
                self._shared_connections.pop(self._shared_key, None)
                print(f"Disconnected from {self.name}")
        else:
            self.device.close()
            print(f"Disconnected from {self.name}")

        self.device = None
        self.serial = None
        self._shared_key = None

    close = disconnect

    def _normalize_payload(self, data, pad_to_length=None, pad_char=None):
        if isinstance(data, bytes):
            return data

        payload = str(data)
        final_pad_to_length = self.pad_to_length if pad_to_length is None else pad_to_length
        final_pad_char = self.pad_char if pad_char is None else pad_char
        if final_pad_to_length is not None:
            payload = payload.ljust(final_pad_to_length, final_pad_char)
        return payload.encode()

    def send(
        self,
        data,
        deadline_seconds=None,
        pad_to_length=None,
        pad_char=None,
        clear_input=None,
    ):
        if self.device is None:
            return False

        payload = self._normalize_payload(
            data,
            pad_to_length=pad_to_length,
            pad_char=pad_char,
        )
        output = False
        while output is False and self.device is not None:
            self.device.write(payload)
            output = self.receive(
                deadline_seconds=deadline_seconds,
                clear_input=clear_input,
            )
            if output is False:
                print(f"WARNING: {self.name} communication failed...")
                print("Retrying...")
        return output

    def receive(self, deadline_seconds=None, clear_input=None):
        if self.device is None:
            return "False".encode("utf-8")

        if clear_input is None:
            clear_input = self.clear_input_on_receive

        timeout_seconds = self.receive_timeout if deadline_seconds is None else deadline_seconds
        t0 = time.time()
        if clear_input and hasattr(self.device, "reset_input_buffer"):
            self.device.reset_input_buffer()

        while self.device.in_waiting == 0 and time.time() - t0 < timeout_seconds:
            time.sleep(0.01)

        if time.time() - t0 >= timeout_seconds:
            print(f"Failed to get data from {self.name}...")
            return False

        return self.device.readline()
