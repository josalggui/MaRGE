#!/usr/bin/env python3

import numpy as np
import experiment as ex

import pdb
st = pdb.set_trace

def radial(trs,
           lo_freq=0.2, rf_amp=0.5, G=0.5,
           grad_tstart=0, rf_tstart=5, rf_tend=50,
           rx_tstart=70, rx_tend=180, rx_period=3,
           tr_total_time=220,
           tx_gate_pre=2,
           tx_gate_post=1,
           init_gpa=False, plot_rx=False, dbg_sc=0.5):

    angles = np.linspace(0, 2*np.pi, trs) # angle

    def radial_tr(tstart, th):

        gx = G * np.cos(th)
        gy = G * np.sin(th)

        value_dict = {
            # second tx0 pulse and tx1 pulse purely for loopback debugging
            'tx0': ( np.array([rf_tstart, rf_tend,    rx_tstart + 15, rx_tend - 15]),
                     np.array([rf_amp, 0,    dbg_sc * (gx+gy*1j), 0]) ),
            'tx1': ( np.array([rx_tstart + 15, rx_tend - 15]), np.array([dbg_sc * (gx+gy*1j), 0]) ),
            'grad_vz': ( np.array([grad_tstart]),
                         np.array([gx]) ),
            'grad_vy': ( np.array([grad_tstart]),
                         np.array([gy]) ),
            'rx0_en' : ( np.array([rx_tstart, rx_tend]),
                            np.array([1, 0]) ),
            'tx_gate' : ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post]),
                          np.array([1, 0]) )
            }

        for k, v in value_dict.items():
            # v for read, value_dict[k] for write
            value_dict[k] = (v[0] + tstart, v[1])

        return value_dict

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa)

    tr_t = 20 # start the first TR at 20us
    for th in angles:
        expt.add_flodict( radial_tr( tr_t, th ) )
        tr_t += tr_total_time

    rxd, msgs = expt.run()

    expt.__del__()
    # expt.close_server(True)

    return rxd['rx0'], msgs

    # if plot_rx:
