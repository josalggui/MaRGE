"""
Frequency fit

@author:    Yolanda Vives

@summary: look for larmor frequency
@status: under development
@todo:

"""
import sys
sys.path.append('../marcos_client')
sys.path.append('../manager')
import numpy as np
import experiment as ex
import matplotlib.pyplot as plt
from manager.datamanager import DataManager

def flipAngle(lo_freq=3.041,  # KHz
            rf_amp=0.3,  # 1 = full-scale
            N_amp=10,  
            step=0.05,   
            rf_pi2_duration = 50, 
            echo_duration = 15e3, 
            BW=31,   # us, 3.333us, 300 kHz rate
            rx_wait=100, 
            readout_duration=10e3, 
            tr_duration=20e3
            ):

    rx_period = 1/(BW*1e-3)

    rf_pi_duration = 2*rf_pi2_duration
       
    ## All times are in the context of a single TR, starting at time 0
    init_gpa = False
   
    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends
    tstart = 20   
    pi2_phase = 1 # x
    pi_phase = 1j # y
    
    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    i=0

    amps = np.linspace(rf_amp-N_amp/2*step, rf_amp+N_amp/2*step, N_amp)
    while i<N_amp:
        rf_tend = tstart+echo_duration+rf_pi_duration/2 # us
        rx_tstart = rf_tend+rx_wait # us
        rx_tend = rx_tstart + readout_duration  # us
        expt.add_flodict({
            # second tx0 pulse purely for loopback debugging
            'tx0': (np.array([tstart + (echo_duration - rf_pi2_duration)/2, tstart + (echo_duration + rf_pi2_duration)/2,
                         tstart + echo_duration - rf_pi_duration/2, rf_tend]), np.array([pi2_phase*amps[i], 0, pi_phase*amps[i], 0])),
            'rx0_en': ( np.array([rx_tstart, rx_tend]),  np.array([1, 0]) ),
            'tx_gate': (np.array([tstart + (echo_duration - rf_pi2_duration)/2- tx_gate_pre, tstart + (echo_duration + rf_pi2_duration)/2 + tx_gate_post,
                         tstart + echo_duration - rf_pi_duration/2- tx_gate_pre, rf_tend+ tx_gate_post]), np.array([1, 0, 1, 0])),
            'rx_gate': ( np.array([rx_tstart, rx_tend]), np.array([1, 0]) )
        })
    
        i = i+1
        tstart = tstart + tr_duration
    
    expt.plot_sequence()
    plt.show()
    rxd, msg = expt.run()   
   
    expt.__del__()
    return rxd['rx0'] 

if __name__ == "__main__":
    
    lo_freq = 3.041
    N_amp = 10
    BW=31
    
    rxd=flipAngle(N_amp=N_amp, lo_freq=lo_freq, BW=BW)
    values = rxd
    samples = np.int32(len(values)/N_amp)
    i=0
    s=0
    peakValsf =[]
    while i < N_amp:
        d_cropped = values[s:s+samples] 
        
        dataobject: DataManager = DataManager(d_cropped, lo_freq, len(d_cropped),  [], BW)
        f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency = dataobject.get_peakparameters()
        peakValsf.append(f_signalValue)

        s=s+samples
        i=i+1
    
#    plt.plot(dataobject.f_fftMagnitude)
#    plt.show()
    
    plt.plot(peakValsf)
    plt.show()

