"""
@author: J.M. Algarín, february 03th 2022
MRILAB @ I3M
"""

import os
import sys
#*****************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************
import experiment as ex
import server_comms as sc
import configs.hw_config as hw
import numpy as np

class Experiment(ex.Experiment):
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
        return self.get_rx_ts()[0] * hw.oversamplingFactor

    def run(self):
        """ compile the TX and grad data, send everything over.
        Returns the resultant data """

        if not self._seq_compiled:
            self.compile()

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
