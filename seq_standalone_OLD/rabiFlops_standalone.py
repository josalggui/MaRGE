"""
Rabi map

@author:    Yolanda Vives

@summary: increase the pulse width and plot the peak value of the signal received 
@status: under development
@todo:

"""
import sys
sys.path.append('../marcos_client')
sys.path.append('../manager')
import matplotlib.pyplot as plt
#from spinEcho_standalone import spin_echo
import numpy as np
import experiment as ex
from manager.datamanager import DataManager


def rabi_flops(lo_freq=3.041,  # MHz
             rf_amp=0.62, # 1 = full-scale
             N=10, 
             step=5,  
             rf_pi2_duration0 = 50,  # us
             tr_duration = 20e3, 
             echo_duration=15e3, # delay after end of RX before start of next TR
             BW=31,  # us, 3.333us, 300 kHz rate
             rx_wait = 100, 
             readout_duration=10e3
             ):
    
    ## All times are in the context of a single TR, starting at time 0
    init_gpa = True

      
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
        
#        if fid==1:
#            rf_tend = rf_tstart + rf_duration+k # us
#            rx_tstart = rf_tend+rx_wait # us
#            rx_tend = rx_tstart + readout_duration  # us
#            expt.add_flodict({
#                # second tx0 pulse purely for loopback debugging
#                'tx0': ( np.array([rf_tstart, rf_tend])+tstart, np.array([rf_amp,0]) ),
#                'rx0_en': ( np.array([rx_tstart, rx_tend])+tstart, np.array([1, 0]) ),
#                'tx_gate': ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post])+tstart, np.array([1, 0]) ), 
#                'rx_gate': ( np.array([rx_tstart, rx_tend])+tstart, np.array([1, 0]) )
#            })
#            tstart = tstart + rx_tend+tr_wait
#        else:
            
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
    
    expt.plot_sequence()
    plt.show()
    
    rxd, msg = expt.run()    
    plt.plot( rxd['rx0'])
    plt.show()
    
    expt.__del__()
    return rxd['rx0']

if __name__ == "__main__":
    
    lo_freq=3.041
    BW=31
    N=10
    values=rabi_flops(lo_freq=3.041, N=N, step=20)
    samples = int(len(values)/N)
    
    i=0
    s=0
    peakValsf =[]
    peakValst = []
    while i < N:
        d_cropped = values[s:s+samples-1] 
                
        dataobject: DataManager = DataManager(d_cropped, lo_freq, len(d_cropped), [1,1, 1], BW)
        peakValsf.append(round(np.max(dataobject.f_fftMagnitude), 4))
        peakValst.append(dataobject.t_magnitude[0])

        s=s+samples
        i=i+1
        

    plt.plot(dataobject.f_fftMagnitude)
    plt.show()
    
    plt.plot(peakValst)
    plt.show()
