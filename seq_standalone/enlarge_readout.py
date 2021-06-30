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


def enlarge_rd(lo_freq=3.041,  # MHz
             rf_amp=0.4, # 1 = full-scale
             rf_pi2_duration = 50,  # us
             tr_duration = 500e3, 
             echo_duration=10e3, # delay after end of RX before start of next TR
             BW=50,  # us, 3.333us, 300 kHz rate
             readout_duration=5000,
             ):
    
    ## All times are in the context of a single TR, starting at time 0
    init_gpa = True

#    phase_amps = np.linspace(phase_amp, -phase_amp, trs)
    
    rf_pi_duration = rf_pi2_duration
       
    rx_period = 1/(BW*1e-3)    
        
    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    
    ##########################################################
    
    def rf_wf(tstart, echo_idx):
        pi2_phase = 1 # x
        pi_phase = 1j # y
        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (echo_duration - rf_pi2_duration)/2, tstart + (echo_duration + rf_pi2_duration)/2,
                             tstart + echo_duration - rf_pi_duration/2]), np.array([pi2_phase*rf_amp, 0, 2*pi_phase*rf_amp])                        
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2]), np.array([0])

    #######################################################################

    def tx_gate_wf(tstart, echo_idx):
        tx_gate_pre = 15 # us, time to start the TX gate before each RF pulse begins
        tx_gate_post = 1 # us, time to keep the TX gate on after an RF pulse ends

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

    def readout_wf(tstart, echo_idx):
        if echo_idx == 0:
            return np.array([tstart + 3*echo_duration/4 - readout_duration/2, tstart + 3*echo_duration/4 + readout_duration/2 ]), np.array([1, 0])
        else:
            return np.array([tstart+readout_duration/2-echo_duration/4]), np.array([0]) # keep on zero otherwise
            
    ##############################################################    
    
    global_t = 20 # start the first TR at 20us

    for echo_idx in range(2):
        tx_t, tx_a = rf_wf(global_t, echo_idx)
        tx_gate_t, tx_gate_a = tx_gate_wf(global_t, echo_idx)
        readout_t, readout_a = readout_wf(global_t, echo_idx)
        rx_gate_t, rx_gate_a = readout_wf(global_t, echo_idx)
        
        expt.add_flodict({
            'tx0': (tx_t, tx_a),
            'rx0_en': (readout_t, readout_a),
            'tx_gate': (tx_gate_t, tx_gate_a),
            'rx_gate': (rx_gate_t, rx_gate_a),
        })
        global_t += echo_duration

    
    expt.plot_sequence()
    plt.show()
   
    rxd, msg = expt.run()    
    plt.plot( rxd['rx0'])
    plt.show()
    
#    expt.__del__()
#    return rxd['rx0']

if __name__ == "__main__":
    
    enlarge_rd(lo_freq=3.0395, readout_duration=7e3)
       
#    plt.plot(values)
#    plt.show()
