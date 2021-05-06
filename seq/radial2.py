#!/usr/bin/env python3

import numpy as np
import experiment as ex
from local_config import ip_address, fpga_clk_freq_MHz, grad_board
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

        # RF
    idict = expt._seq
    tx0_i_t, tx0_i_a = idict['tx0_i']
    tx0_q_t, tx0_q_a = idict['tx0_q']
    tx0_t = tx0_i_t / fpga_clk_freq_MHz
    tx0_y = (tx0_i_a + 1j * tx0_q_a)/32767

    # Grad y
    grad_y_t,  grad_y_a = idict['ocra1_vy']
    grad_y_t_float = grad_y_t / fpga_clk_freq_MHz
    grad_y_a_float = (grad_y_a - 32768) / 32768
    
    # Grad z
    grad_z_t,  grad_z_a = idict['ocra1_vz']
    grad_z_t_float = grad_z_t / fpga_clk_freq_MHz
    grad_z_a_float = (grad_z_a - 32768) / 32768
    
    expt.__del__()
    
    return tx0_t, tx0_y,  grad_y_t_float, grad_y_a_float, grad_z_t_float, grad_z_a_float



        
