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


def inversionRecovery(lo_freq=3.023, # MHz
             rf_amp=0.62, # 1 = full-scale
             rf_duration=50,
             N=20,  # Number of points
             step=5,  # Step in us 
             tr_wait = 1000, 
             rx_wait = 100,   #us
             tr=50, # ms
             rx_period=1/31e-3,  # us, 3.333us, 300 kHz rate
             readout_duration=5000,
             shimming=(0, 0, 0), 
             echo_duration=4,   #ms
             
             ):
    
    echo_duration = echo_duration*1e3
    tr = tr*1e3
        
    ## All times are in the context of a single TR, starting at time 0
    init_gpa = True
   
    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends
        
    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    tstart = 0
    k = 0
    i=0
    pi2_phase = 1 # x
    pi_phase = 1j # y
    while i < N:     
        rf_tend = tstart+echo_duration+k+rf_duration/2 # us
        rx_tstart = rf_tend+rx_wait # us
        rx_tend = rx_tstart + readout_duration  # us
        expt.add_flodict({
            # second tx0 pulse purely for loopback debugging
            'tx0': (np.array([tstart + (echo_duration - rf_duration)/2, tstart + (echo_duration + rf_duration)/2,
                         tstart + echo_duration+k - rf_duration/2, rf_tend]), np.array([pi_phase*rf_amp, 0, pi2_phase*rf_amp/2, 0])),
            'rx0_en': ( np.array([rx_tstart, rx_tend]),  np.array([1, 0]) ),
            'tx_gate': (np.array([tstart + (echo_duration - rf_duration)/2- tx_gate_pre, tstart + (echo_duration + rf_duration)/2 + tx_gate_post,
                         tstart + echo_duration+k - rf_duration/2- tx_gate_pre, rf_tend+ tx_gate_post]), np.array([1, 0, 1, 0])),
            'rx_gate': ( np.array([rx_tstart, rx_tend]), np.array([1, 0]) )
        })
        tstart = tstart +tr
        i = i+1
        k=k+step
    
    expt.plot_sequence()
    plt.show()
    
    rxd, msgs = expt.run()    
    plt.plot( rxd['rx0'])
    plt.show()
    
    expt.__del__()
    return rxd['rx0']

if __name__ == "__main__":
    
    N=3
    values=inversionRecovery(lo_freq=3.041, rf_amp=0.30,  rf_duration=160, N=N, step=20, rx_wait=200)
    samples = int(len(values)/N)
    
    i=0
    s=0
    peakValsf =[]
    peakValst = []
    while i < N:
        d_cropped = values[s:s+samples-1] 
        
        f_fftData = np.fft.fftshift(np.fft.fft((d_cropped), n=samples))
        f_fftMagnitude = abs(f_fftData)
        f_signalValue: float = round(np.max(f_fftMagnitude), 4)
        peakValsf.append(f_signalValue)
        
        t_magnitude = np.abs(d_cropped)
        t_magnitudeCon = np.convolve(t_magnitude, np.ones((50,)) / 50, mode='same')
        t_signalValue: float = t_magnitudeCon[0]
        peakValst.append(t_signalValue)
        
        s=s+samples
        i=i+1


    plt.plot(f_fftMagnitude)
    plt.show()
    
    plt.plot(t_magnitudeCon)
    plt.show()
    
    plt.plot(peakValst)
    plt.show()
