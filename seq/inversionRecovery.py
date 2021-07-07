"""
Rabi map

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


def inversionRecovery(self):

    lo_freq = self.lo_freq
    rf_amp = self.rf_amp
    N_ir = self.N_ir
    step=self.step  
    rf_duration = self.rf_duration
    tr_duration=self.tr_duration*1e3  # delay after end of RX before start of next TR
    echo_duration = self.echo_duration*1e3
    BW=self.BW  # us, 3.333us, 300 kHz rate
    rx_wait=self.rx_wait
    readout_duration=self.readout_duration*1e3
       
       
    ## All times are in the context of a single TR, starting at time 0
    init_gpa = True
   
    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends
        
    rx_period = 1/(BW*1e-3)    
    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    tstart = 0
    k = 0
    i=0
    pi2_phase = 1 # x
    pi_phase = 1j # y
    while i < N_ir:     
        rf_tend = tstart+echo_duration+k+rf_duration/2 # us
        rx_tstart = rf_tend+rx_wait # us
        rx_tend = rx_tstart + readout_duration  # us
        expt.add_flodict({
            'tx0': (np.array([tstart + (echo_duration - rf_duration)/2, tstart + (echo_duration + rf_duration)/2,
                         tstart + echo_duration+k - rf_duration/2, rf_tend]), np.array([pi_phase*rf_amp, 0, pi2_phase*rf_amp/2, 0])),
            'rx0_en': ( np.array([rx_tstart, rx_tend]),  np.array([1, 0]) ),
            'tx_gate': (np.array([tstart + (echo_duration - rf_duration)/2- tx_gate_pre, tstart + (echo_duration + rf_duration)/2 + tx_gate_post,
                         tstart + echo_duration+k - rf_duration/2- tx_gate_pre, rf_tend+ tx_gate_post]), np.array([1, 0, 1, 0])),
            'rx_gate': ( np.array([rx_tstart, rx_tend]), np.array([1, 0]) )
        })
        tstart = tstart +tr_duration
        i = i+1
        k=k+step
    
    expt.plot_sequence()
    plt.show()
    
    rxd, msgs = expt.run()    
    
    expt.__del__()
    return rxd['rx0'], msgs


