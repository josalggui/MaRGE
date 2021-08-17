"""
Rabi map

@author:    Yolanda Vives

@summary: increase the pulse width and plot the peak value of the signal received 
@status: under development
@todo:

"""
import sys
sys.path.append('../marcos_client')

#from spinEcho_standalone import spin_echo
import numpy as np
import experiment as ex
import matplotlib.pyplot as plt


def rabi_flops(self):
     
    lo_freq=self.lo_freq # KHz
    rf_amp=self.rf_amp # 1 = full-scale
    N=self.N 
    nScans = self.nScans
    step=self.step  
    rf_pi2_duration0 = self.rf_pi2_duration
    tr_duration=self.tr_duration*1e3  # delay after end of RX before start of next TR
    echo_duration = self.echo_duration*1e3
    BW=self.BW  # us, 3.333us, 300 kHz rate
    rx_wait=self.rx_wait*1e3
    readout_duration=self.readout_duration*1e3
  
    ## All times are in the context of a single TR, starting at time 0
    init_gpa = False

     
    rx_period = 1/(BW*1e-3)
    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    
    
    ##########################################################
    
    def rf_wf(tstart, echo_idx, k):
        rf_pi2_duration = rf_pi2_duration0+k
        rf_pi_duration = rf_pi2_duration*2
        pi2_phase = 1 # x
        pi_phase = 1j # y
        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (echo_duration - rf_pi2_duration)/2, tstart + (echo_duration + rf_pi2_duration)/2,
                             tstart + echo_duration - rf_pi_duration/2]), np.array([pi2_phase*rf_amp, 0, pi_phase*rf_amp])                        
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2]), np.array([0])

#######################################################################

    def tx_gate_wf(tstart, echo_idx):
        tx_gate_pre = 15 # us, time to start the TX gate before each RF pulse begins
        tx_gate_post = 1 # us, time to keep the TX gate on after an RF pulse ends
        rf_pi2_duration = rf_pi2_duration0+k
        rf_pi_duration = rf_pi2_duration*2
        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (echo_duration - rf_pi2_duration)/2 - tx_gate_pre,
                             tstart + (echo_duration + rf_pi2_duration)/2 + tx_gate_post,
                             tstart + echo_duration - rf_pi_duration/2 - tx_gate_pre]), \
                             np.array([1, 0, 1])
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2 + tx_gate_post]), np.array([0])

##############################################################

    def readout_wf(tstart, echo_idx, k):
        rf_pi2_duration = rf_pi2_duration0+k
        rf_pi_duration = rf_pi2_duration*2
        if echo_idx != 0:
            return np.array([tstart + rf_pi_duration/2 + rx_wait, tstart + + rf_pi_duration/2 + rx_wait + readout_duration ]), np.array([1, 0])
        else:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
            
##############################################################    
    global_t = 20 # start the first TR at 20us
    k = 0
    i=0
    while i < N:     
           
        for echo_idx in range(2):
            tx_t, tx_a = rf_wf(global_t, echo_idx, k)
            tx_gate_t, tx_gate_a = tx_gate_wf(global_t, echo_idx)
            readout_t, readout_a = readout_wf(global_t, echo_idx, k)
            rx_gate_t, rx_gate_a = readout_wf(global_t, echo_idx, k)
            
            expt.add_flodict({
                'tx0': (tx_t, tx_a),
                'rx0_en': (readout_t, readout_a),
                'tx_gate': (tx_gate_t, tx_gate_a),
                'rx_gate': (rx_gate_t, rx_gate_a),
            })
            global_t += echo_duration
            
        global_t += tr_duration-echo_duration
        i = i+1
        k=k+step
    
#    expt.plot_sequence()
#    plt.show()   

    for nS in range(nScans):
        print('nScan=%s'%(nS))
        rxd, msgs = expt.run()
        if nS ==0:
            n_rxd = rxd['rx0']
        else:
            n_rxd = np.concatenate((n_rxd, rxd['rx0']), axis=0)
    
    n_rxd = np.reshape(n_rxd, (nScans, len(rxd['rx0'])))
    n_rxd = np.average(n_rxd, axis=0) 
        
    expt.__del__()
    return n_rxd, msgs

