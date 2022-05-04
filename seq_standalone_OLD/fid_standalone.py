#!/usr/bin/env python3
import sys
sys.path.append('../marcos_client')
sys.path.append('manager')

import numpy as np
import experiment as ex
import matplotlib.pyplot as plt
from datamanager import DataManager
import pdb
st = pdb.set_trace

   
def fid(lo_freq=3.032, # MHz
             rf_amp=0.3, # 1 = full-scale
             rf_duration=100,
             rf_tstart = 100,  # us
             tr_wait=100, # delay after end of RX before start of next TR
             rx_period=0.0323,  # us, 3.333us, 300 kHz rate
             readout_duration=500, 
             shimming=(0.01, 0.01, 0.01)
             ):

    ## All times are in the context of a single TR, starting at time 0
    init_gpa = True

#    phase_amps = np.linspace(phase_amp, -phase_amp, trs)
    rf_tend = rf_tstart + rf_duration # us

    rx_tstart = rf_tend+tr_wait # us
    rx_tend = rx_tstart + readout_duration  # us

    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends


    def fid_tr(tstart):
        value_dict = {
            # second tx0 pulse purely for loopback debugging
            'tx0': ( np.array([rf_tstart, rf_tend]) + tstart,
                     np.array([rf_amp,0]) ),
            'rx0_en': ( np.array([rx_tstart, rx_tend]) + tstart, np.array([1, 0]) ),
             'tx_gate': ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post]) + tstart, np.array([1, 0]) )
        }

        return value_dict

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa)
    # gpa_fhdo_offset_time in microseconds; offset between channels to
    # avoid parallel updates (default update rate is 0.2 Msps, so
    # 1/0.2 = 5us, 5 / 3.1 gives the offset between channels; extra
    # 0.1 for a safety margin)

    tr_t = 0 # start the first TR at 20us
    expt.add_flodict( fid_tr( tr_t) )

#    expt.close_server(True)

    rxd, msgs = expt.run()    
   
    expt.plot_sequence()
    plt.show()     
   
    print(msgs)
    
    expt.__del__()
       
    plt.plot( np.abs(rxd['rx0']) )
#        plt.plot( rxd['rx0'].real )
#        plt.plot( rxd['rx0'].imag )
    plt.show()
    
    recept = rxd['rx0']
    
    dataobject: DataManager = DataManager(recept, lo_freq, len(recept), [], 1/rx_period)
    plt.plot(dataobject.f_axis, dataobject.f_fftMagnitude)
    plt.show()

    
#    d_cropped = rxd[0:len(rxd)]  # * 2000.0
#    f_fftData = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rxd), len(rxd)))
#    f_fftMagnitude = abs(f_fftData) 
#    plt.plot(f_fftMagnitude)
        
if __name__ == "__main__":
    
    fid()
