"""
Shimming

@author:    Yolanda Vives

@summary: increase the pulse width and plot the peak value of the signal received 
@status: under development
@todo:

"""
import sys
sys.path.append('../marcos_client')
import matplotlib.pyplot as plt

#from spinEcho_standalone import spin_echo
import numpy as np
import experiment as ex


def shimming(self, grad):
     
    lo_freq=self.lo_freq # KHz
    rf_amp=self.rf_amp # 1 = full-scale
    rf_duration = self.rf_duration    
    rf_tstart=self.rf_tstart
    N=self.N    #points
    shim_initial = self.shim_initial
    shim_final = self.shim_final    
    tr_duration=self.tr_duration*1e3  # delay after end of RX before start of next TR
    BW=self.BW  # us, 3.333us, 300 kHz rate
    rx_wait=self.rx_wait*1e3
    readout_duration=self.readout_duration*1e3
  
    ## All times are in the context of a single TR, starting at time 0
    init_gpa = False

    rx_period = 1/(BW*1e-3)
    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    
    shim_vector = np.linspace(shim_initial, shim_final, N)
    
    tstart = 20 # start the first TR at 20us
    i=0
    while i < N:     
        
        shim_val = shim_vector[i]
        
        rf_tend = rf_tstart + rf_duration # us
        rx_tstart = rf_tend+rx_wait # us
        rx_tend = rx_tstart + readout_duration  # us
        
        tx_gate_pre = 15 # us, time to start the TX gate before each RF pulse begins
        tx_gate_post = 1 # us, time to keep the TX gate on after an RF pulse ends

        if grad=='x':
            expt.add_flodict({
                # second tx0 pulse purely for loopback debugging
                'tx0': ( np.array([rf_tstart, rf_tend])+tstart, np.array([rf_amp,0]) ),
                'grad_vx': (np.array([1])+tstart, np.array([shim_val])), 
                'rx0_en': ( np.array([rx_tstart, rx_tend])+tstart, np.array([1, 0]) ),
                'tx_gate': ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post])+tstart, np.array([1, 0]) ), 
                'rx_gate': ( np.array([rx_tstart, rx_tend])+tstart, np.array([1, 0]) )
            })
        elif grad=='y':
            expt.add_flodict({
                # second tx0 pulse purely for loopback debugging
                'tx0': ( np.array([rf_tstart, rf_tend])+tstart, np.array([rf_amp,0]) ),
                'grad_vy': (np.array([1])+tstart, np.array([shim_val])), 
                'rx0_en': ( np.array([rx_tstart, rx_tend])+tstart, np.array([1, 0]) ),
                'tx_gate': ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post])+tstart, np.array([1, 0]) ), 
                'rx_gate': ( np.array([rx_tstart, rx_tend])+tstart, np.array([1, 0]) )
            })    
        elif grad=='z':
            expt.add_flodict({
                # second tx0 pulse purely for loopback debugging
                'tx0': ( np.array([rf_tstart, rf_tend])+tstart, np.array([rf_amp,0]) ),
                'grad_vz': (np.array([1])+tstart, np.array([shim_val])), 
                'rx0_en': ( np.array([rx_tstart, rx_tend])+tstart, np.array([1, 0]) ),
                'tx_gate': ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post])+tstart, np.array([1, 0]) ), 
                'rx_gate': ( np.array([rx_tstart, rx_tend])+tstart, np.array([1, 0]) )
            })            
            
        tstart = tstart + tr_duration
        i = i+1

    expt.add_flodict({
        'grad_vx': (np.array([tstart]),np.array([0]) ), 
        'grad_vy': (np.array([tstart]),np.array([0]) ), 
        'grad_vz': (np.array([tstart]),np.array([0]) ), 
    })
    
    expt.plot_sequence()
    plt.show()
 
    rxd, msgs = expt.run()    
        
    expt.__del__()
    return rxd['rx0'], msgs

