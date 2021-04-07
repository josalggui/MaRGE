#!/usr/bin/env python3

import numpy as np
import experiment as ex
import sys
sys.path.append('../marcos_client')

import pdb
st = pdb.set_trace

def radial(self):
  
    tx_gate_pre = 2 # us, time to start the TX gate before each RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after an RF pulse ends
    angles = np.linspace(0, 2*np.pi, self.trs) # angle

    def radial_tr(tstart, th):

        gx = self.G * np.cos(th)
        gy = self.G * np.sin(th)

        value_dict = {
            # second tx0 pulse and tx1 pulse purely for loopback debugging
            'tx0': ( np.array([self.rf_tstart, self.rf_tend,    self.rx_tstart + 15, self.rx_tend - 15]),
                     np.array([self.rf_amp, 0,    self.dbg_sc * (gx+gy*1j), 0]) ),
            'tx1': ( np.array([self.rx_tstart + 15, self.rx_tend - 15]), np.array([self.dbg_sc * (gx+gy*1j), 0]) ),
            'grad_vz': ( np.array([self.grad_tstart]),
                         np.array([gx]) ),
            'grad_vy': ( np.array([self.grad_tstart]),
                         np.array([gy]) ),
            'rx0_en' : ( np.array([self.rx_tstart, self.rx_tend]),
                            np.array([1, 0]) ),
            'tx_gate' : ( np.array([self.rf_tstart - tx_gate_pre, self.rf_tend + tx_gate_post]),
                          np.array([1, 0]) )
            }

        for k, v in value_dict.items():
            # v for read, value_dict[k] for write
            value_dict[k] = (v[0] + tstart, v[1])

        return value_dict

    expt = ex.Experiment(lo_freq=self.lo_freq, rx_t=self.rx_period, init_gpa=self.init_gpa)

    tr_t = 20 # start the first TR at 20us
    for th in angles:
        expt.add_flodict( radial_tr( tr_t, th ) )
        tr_t += self.tr_total_time

    rxd, msgs = expt.run()
    # expt.close_server(True)

    if self.plot_rx:
        return rxd['rx0'].real, rxd['rx0'].imag

        
