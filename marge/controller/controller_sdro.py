import numpy as np
import time
import marge.configs.hw_config as hw
from marge.autotuning.autotuning import Arduino  # import from autotuning, no separate class


class SDRO:
    """
    SDRO controller for RF gain/attenuation hardware.

    Uses the same Arduino class as the autotuning system to avoid
    having a separate Arduino per hardware device.

    Hardware chain: Attenuator (0-31 dB) -> 26 dB Amplifier
    Total gain = 26 - attenuation
    Valid gain range: -5 dB (max attenuation) to 26 dB (no attenuation)

    Two modes of operation:

    - auto:       Deterministic one-shot gain correction from Larmor echo amplitude.
                  G_new = G_current + 20 * log10(targetAmplitude / echoAmplitude)
                  Target = 90 mV, with an accuracy band of +/-10 mV (80-100 mV).
                  If the echo is already within that band, no correction is applied.

    - manual:     User provides gain directly, no calculation performed.
                  Use when signal is saturated or Larmor has not been run.

    Serial number stored in hw_config.py as hw.ard_sn_sdro.
    """

    def __init__(self, serial_number):
        self.arduino = Arduino(baudrate=115200, timeout=0.1)  # same as autotuning
        self.serial_number = serial_number
        self.current_gain = None

    def connect(self):
        """Connect to the SDRO Arduino using autotuning Arduino class."""
        self.arduino.connect(serial_number=self.serial_number)
        if self.arduino.device is None:
            raise RuntimeError("SDRO Arduino not found.")
        print(f"SDRO connected on port: {self.arduino.port}")
        time.sleep(1)
        self.set_gain(hw.sdrolnaGain)

    def disconnect(self):
        """Disconnect from the SDRO Arduino."""
        self.arduino.disconnect()

    def _send(self, data: str):
        """
        Send a command to Arduino and return the response string.
        Adapts autotuning Arduino's send/receive to SDRO communication protocol.

        The autotuning Arduino.send() encodes and sends data, then calls receive()
        which returns raw bytes. We decode and parse the response here.
        """
        if self.arduino.device is None:
            raise RuntimeError("Arduino not connected.")

        # Flush buffers before sending
        self.arduino.device.reset_input_buffer()
        self.arduino.device.reset_output_buffer()

        # Send command with newline terminator
        self.arduino.device.write((data + "\n").encode())

        # Collect response lines until 'Ready' or timeout
        time.sleep(0.3)
        responses = []
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if self.arduino.device.in_waiting > 0:
                line = self.arduino.device.readline().decode(errors='replace').strip()
                if line:
                    responses.append(line)
                if responses and "Ready" in responses[-1]:
                    break
            else:
                time.sleep(0.05)

        return responses

    def set_gain(self, gain: int):
        """
        Set total gain of the SDRO system.

        :param gain: Desired gain in dB. Valid range: -5 to 26 dB.
                     total_gain = 26dB (amp) - attenuation
                     e.g. gain=10 -> attenuation=16, gain=26 -> attenuation=0, gain=-5 -> attenuation=31
        :return: Arduino response.
        """
        attenuation = 26 - gain

        if not (0 <= attenuation <= 31):
            raise ValueError(f"Gain {gain} dB out of range. Valid range is -5 to 26 dB.")

        response = self._send(str(attenuation))
        self.current_gain = gain
        print(f"Gain={gain} dB -> Attenuation={attenuation} dB | Arduino: {response}")
        return response

    def get_gain(self):
        """
        Read back current gain from Arduino and compute total gain.

        :return: Current total gain in dB.
        """
        return self.current_gain

    def suggest_gain(self):
        """
        Suggest the correct gain based on current hw.echoAmplitude without setting it.
        Use this to know what manual gain value to input when signal is saturated.

        :return: Suggested gain in dB, or None if echo amplitude is unavailable.
        """
        echo = hw.echoAmplitude
        target = hw.targetAmplitude

        if echo is None or echo == 0.0:
            print("WARNING: No echo amplitude available.")
            return None

        if self.current_gain is None:
            print("WARNING: No current gain set. Set an initial gain first.")
            return None

        correction = 20 * np.log10(target / echo)
        suggested_gain = int(np.floor(self.current_gain + correction))
        suggested_gain = max(-5, min(26, suggested_gain))

        print(f"Suggested gain: {suggested_gain} dB (to bring signal to ~{target:.0f} mV)")

        return suggested_gain

    def adjust_gain(self, manual_gain=None, mode='auto'):
        """
        Set SDRO gain in one of two modes.

        :param manual_gain: int, required when mode='manual'. Gain in dB (-5 to 26).
        :param mode: str, one of 'auto', 'manual'.

        auto mode:
            Deterministic one-shot correction from hw.echoAmplitude.
            G_new = G_current + 20 * log10(targetAmplitude / echoAmplitude)

        manual mode:
            Sets gain directly to manual_gain with no calculation.
            Use when signal is saturated. Call suggest_gain() first.
        """

        # MANUAL MODE
        if mode == 'manual':
            if manual_gain is None:
                print("ERROR: manual_gain must be provided in manual mode.")
                return
            if not (-5 <= manual_gain <= 26):
                raise ValueError(f"Manual gain {manual_gain} dB out of range.")
            self.set_gain(manual_gain)
            return

        # AUTO MODE
        if mode == 'auto':
            echo = hw.echoAmplitude
            target = hw.targetAmplitude

            if echo is None or echo == 0.0:
                print("WARNING: No echo amplitude available. Run Larmor first or use manual mode.")
                return

            tolerance = getattr(hw, 'gainTolerance', 10.0)

            if echo > 136.0:
                print("WARNING: Signal near ADC limit. Consider using manual mode with suggest_gain().")

            correction = 20 * np.log10(target / echo)
            new_gain = int(np.round(self.current_gain + correction))
            if new_gain > 26:
                print("WARNING: Required gain larger than the maximum.")
            new_gain = max(-5, min(26, new_gain))
            if new_gain < -5:
                print("WARNING: Required gain lower than the minimum.")
            self.set_gain(new_gain)
            return

        print(f"ERROR: Unknown mode '{mode}'. Use auto or manual.")


if __name__ == '__main__':
    sdro = SDRO(serial_number=hw.ard_sn_sdro)
    sdro.connect()

    sdro.set_gain(0)
    sdro.get_gain()

    print("\n--- Auto Test: baseline signal ---")
    hw.echoAmplitude = 48.0
    sdro.adjust_gain(mode='auto')

    print("\n--- Manual Test ---")
    sdro.adjust_gain(manual_gain=-5, mode='manual')

    print("\n--- Suggest gain ---")
    hw.echoAmplitude = 220.0
    sdro.suggest_gain()

    sdro.disconnect()
