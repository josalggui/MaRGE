#!/usr/bin/env python3

import sys
sys.path.append('../../marcos_client')
import numpy as np
import experiment as ex
import matplotlib.pyplot as plt
import pdb
st = pdb.set_trace

def radial(self, plotSeq):

    tx_gate_pre = 2 # us, time to start the TX gate before each RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after an RF pulse ends
    angles = np.linspace(0, 2*np.pi, self.trs) # angle
    
    #Shimming
#    shim_x: int = self.shim[0]
    shim_y: int = self.shim[1]
    shim_z: int = self.shim[2]

    def radial_tr(tstart, th):

        gx = self.G * np.cos(th)
        gy = self.G * np.sin(th)

        value_dict = {
            # second tx0 pulse and tx1 pulse purely for loopback debugging
            'tx0': ( np.array([self.rf_tstart, self.rf_tend,    self.rx_tstart + 15, self.rx_tend - 15]),
                     np.array([self.rf_amp, 0,    self.dbg_sc * (gx+gy*1j), 0]) ),
            'grad_vz': ( np.array([self.grad_tstart]),
                         np.array([gx]) +shim_z),
            'grad_vy': ( np.array([self.grad_tstart]),
                         np.array([gy]) +shim_y),
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

    if plotSeq==1:
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        rxd, msgs = expt.run()
        expt.__del__()
        return rxd['rx0'], msgs
