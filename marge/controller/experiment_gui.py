"""
@author: J.M. Algarín, february 03th 2022
MRILAB @ I3M
"""

import time

import marge.marcos.marcos_client.experiment as ex
import marge.marcos.marcos_client.server_comms as sc
import marge.configs.hw_config as hw
import numpy as np

class Experiment(ex.Experiment):
    """
    Custom experiment class that extends the base Experiment class from the 'ex' module.

    Args:
        lo_freq (float): Frequency of the LO (Local Oscillator) in MHz.
        rx_t (float): RX (Receiver) time in microseconds. Should be multiples of 1/122.88, where some values like 3.125 are exact.
        seq_dict (dict): Dictionary containing the sequence information.
        seq_csv (str): Path to a CSV file containing the sequence information.
        rx_lo (int): Specifies which internal NCO (Numerically Controlled Oscillator) local oscillator to use for each channel.
        grad_max_update_rate (float): Maximum update rate of the gradient in MSPS (Mega Samples Per Second) across all channels in parallel.
        gpa_fhdo_offset_time (int): Offset time used when GPA-FHDO (Gradient Pulse Amplitude - Fractional High Dynamic Range Output) is used.
        print_infos (bool): Flag to control the display of server info messages.
        assert_errors (bool): Flag to control whether to halt on server errors.
        init_gpa (bool): Flag to initialize the GPA (Gradient Pulse Amplitude) when the Experiment object is created.
        initial_wait (float): Initial pause before the experiment begins, in microseconds. Required to configure the LOs (Local Oscillators) and RX rate.
        auto_leds (bool): Flag to automatically scan the LED (Light-Emitting Diode) pattern from 0 to 255 as the sequence runs.
        prev_socket (socket): Previously-opened socket to maintain status.
        fix_cic_scale (bool): Flag to scale the RX (Receiver) data precisely based on the rate being used.
        set_cic_shift (bool): Flag to program the CIC (Cascaded Integrator-Comb) internal bit shift to maintain the gain within a factor of 2 independent of the rate.
        allow_user_init_cfg (bool): Flag to allow user-defined alteration of flocra (Field-Programmable Logic Controller for Real-Time Acquisition) configuration set by init.
        halt_and_reset (bool): Flag to halt any existing sequences that may be running upon connecting to the server.
        flush_old_rx (bool): Flag to read out and clear the old RX (Receiver) FIFOs before running a sequence.

    Summary:
        The Experiment class extends the base Experiment class from the 'ex' module and provides additional functionality and customization for experiments.
        It inherits all the attributes and methods from the base class and overrides the __init__() and run() methods.
    """
    def __init__(self,
                 lo_freq=1,  # MHz
                 rx_t=3.125, # us; multiples of 1/122.88, such as 3.125, are exact, others will be rounded to the nearest multiple of the 122.88 MHz clock
                 seq_dict=None,
                 seq_csv=None,
                 rx_lo=0,  # which of internal NCO local oscillators (LOs), out of 0, 1, 2, to use for each channel
                 grad_max_update_rate=0.2,  # MSPS, across all channels in parallel, best-effort
                 gpa_fhdo_offset_time=0, # when GPA-FHDO is used, offset the Y, Z and Z2 gradient times by 1x, 2x and 3x this value to emulate 'simultaneous' updates
                 print_infos=True,  # show server info messages
                 assert_errors=True,  # halt on server errors
                 init_gpa=False,  # initialise the GPA (will reset its outputs when the Experiment object is created)
                 initial_wait=None, # initial pause before experiment begins - required to configure the LOs and RX rate; must be at least a few us. Is suitably set based on grad_max_update_rate by default.
                 auto_leds=True, # automatically scan the LED pattern from 0 to 255 as the sequence runs (set to off if you wish to manually control the LEDs)
                 prev_socket=None,  # previously-opened socket, if want to maintain status etc
                 fix_cic_scale=True, # scale the RX data precisely based on the rate being used; otherwise a 2x variation possible in data amplitude based on rate
                 set_cic_shift=False, # program the CIC internal bit shift to maintain the gain within a factor of 2 independent of rate; required if the open-source CIC is used in the design
                 allow_user_init_cfg=False, # allow user-defined alteration of flocra configuration set by init, namely RX rate, LO properties etc; see the compile() method for details
                 halt_and_reset=False,  # upon connecting to the server, halt any existing sequences that may be running
                 flush_old_rx=False, # when debugging or developing new code, you may accidentally fill up the RX FIFOs - they will not automatically be cleared in case there is important data inside. Setting this true will always read them out and clear them before running a sequence. More advanced manual code can read RX from existing sequences.
                 ):
        """
        Initialize the Experiment object with the specified parameters.
        """
        super(Experiment, self).__init__(lo_freq,
                                         rx_t / hw.oversamplingFactor,
                                         seq_dict,
                                         seq_csv,
                                         rx_lo,
                                         grad_max_update_rate,
                                         gpa_fhdo_offset_time,
                                         print_infos,
                                         assert_errors,
                                         init_gpa,
                                         initial_wait,
                                         auto_leds,
                                         prev_socket,
                                         fix_cic_scale,
                                         set_cic_shift,
                                         allow_user_init_cfg,
                                         halt_and_reset,
                                         flush_old_rx,)

    def getSamplingRate(self):
        """
        Get the sampling rate of the experiment in the sequence sampling rate.

        Returns:
            float: The sampling rate in samples per second.
        """
        return self.get_rx_ts()[0] * hw.oversamplingFactor

    def get_sampling_period(self):
        """
        Get the sampling rate of the experiment in the sequence sampling rate.

        Returns:
            float: The sampling rate in samples per second.
        """
        return self.get_rx_ts()[0] * hw.oversamplingFactor  # s

    def run(self):
        """
        Compile the TX and gradient data and send everything over to the server.
        Returns the resultant data.

        Returns:
            tuple: A tuple containing the resultant data and messages.
                   The resultant data is a dictionary containing the received IQ signals for each channel in mV.
                   The messages are server messages.

        Raises:
            None
        """

        if not self._seq_compiled:
            t1 = time.time()
            self.compile()
            t2 = time.time()
            print(f"Compilation time: {round(t2-t1, 1)} s")

        if self._flush_old_rx:
            rx_data_old, _ = sc.command({'read_rx': 0}, self._s)
            # TODO: do something with RX data previously collected by the server

        rx_data, msgs = sc.command({'run_seq': self._machine_code.tobytes()}, self._s)

        rxd = rx_data[4]['run_seq']
        rxd_iq = {}

        # (1 << 24) just for the int->float conversion to be reasonable - exact value doesn't matter for now
        rx0_norm_factor = self._rx0_cic_factor / (1 << 24)
        rx1_norm_factor = self._rx0_cic_factor / (1 << 24)

        try: # Signal in millivolts with phase as it should be
            rxd_iq['rx0'] = hw.adcFactor * rx0_norm_factor * (np.array(rxd['rx0_i']).astype(np.int32).astype(float) -
                                               1j * np.array(rxd['rx0_q']).astype(np.int32).astype(float))
        except (KeyError, TypeError):
            pass

        try: # Signal in millivolts with phase as it should be
            rxd_iq['rx1'] = hw.adcFactor * rx1_norm_factor * (np.array(rxd['rx1_i']).astype(np.int32).astype(float) -
                                               1j * np.array(rxd['rx1_q']).astype(np.int32).astype(float))
        except (KeyError, TypeError):
            pass

        return rxd_iq, msgs
