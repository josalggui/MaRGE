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


def rabi_flops(lo_freq=3.069, # MHz
             rf_amp=0.62, # 1 = full-scale
             rf_duration=50,
             N=10, 
             step=5,  
             rf_tstart = 100,  # us
             rx_wait = 0, 
             tr_wait=0, # delay after end of RX before start of next TR
             rx_period=10/3,  # us, 3.333us, 300 kHz rate
             readout_duration=500,
             shimming=(0, 0, 0)
             ):
        
    ## All times are in the context of a single TR, starting at time 0
    init_gpa = True

#    phase_amps = np.linspace(phase_amp, -phase_amp, trs)
    
    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends
        
    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    tstart = 0
    k = 0
    i=0
    while i < N:     
        rf_tend = rf_tstart + rf_duration+k # us
        rx_tstart = rf_tend+rx_wait # us
        rx_tend = rx_tstart + readout_duration  # us
        expt.add_flodict({
            # second tx0 pulse purely for loopback debugging
            'tx0': ( np.array([rf_tstart, rf_tend])+tstart, np.array([rf_amp,0]) ),
            'rx0_en': ( np.array([rx_tstart, rx_tend])+tstart, np.array([1, 0]) ),
            'tx_gate': ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post])+tstart, np.array([1, 0]) ), 
            'rx_gate': ( np.array([rx_tstart, rx_tend])+tstart, np.array([1, 0]) )
        })
        tstart = tstart + rx_tend+tr_wait
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
    
    N=20
    values=rabi_flops(lo_freq=3.069, rf_amp=0.62,  rf_duration=50, N=N, step=5, tr_wait=1000)
    samples = int(len(values)/N)
    
    i=0
    s=0
    peakVals =[]
    while i < N:
        d_cropped = values[s:s+samples-1] 
        
        f_fftData = np.fft.fftshift(np.fft.fft(np.fft.fftshift(d_cropped), n=samples))
        f_fftMagnitude = abs(f_fftData)
        f_signalValue: float = round(np.max(f_fftMagnitude), 4)
        peakVals.append(f_signalValue)
        
#        t_magnitude = np.abs(d_cropped)
#        t_magnitudeCon = np.convolve(t_magnitude, np.ones((50,)) / 50, mode='same')
#        t_signalValue: float = round(np.max(t_magnitudeCon), 4)
#        peakVals.append(t_signalValue)
        
        s=s+samples
        i=i+1

    plt.plot(peakVals)
    plt.show()
