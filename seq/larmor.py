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
#import matplotlib.pyplot as plt
#from configs.hw_config import fo
#from seq_standalone.fid_standalone import fid
#from seq_standalone.spinEcho_standalone import spin_echo
from manager.datamanager import DataManager
import experiment as ex

def larmorFreq(self):

    lo_freq=self.lo_freq # MHz
    rf_amp=self.rf_amp # 1 = full-scale
    N=self.N 
    step=self.step  
    rf_pi2_duration = self.rf_pi2_duration
    echo_duration = self.echo_duration*1e3
    BW=self.BW  # us, 3.333us, 300 kHz rate
    rx_wait=self.rx_wait*1e3
    readout_duration=self.readout_duration*1e3
    nScans=self.nScans
              

    rx_period = 1/(BW*1e-3)
    step = step*1e-3

    rf_pi_duration = 2*rf_pi2_duration
#    BW = BW*1e3   
       
    ## All times are in the context of a single TR, starting at time 0
    init_gpa = False
   
    tx_gate_pre = 2 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends
    tstart = 0   
    pi2_phase = 1 # x
    pi_phase = 1j # y
    
    i=0
    peakValsf = []
    freqs = np.linspace(lo_freq-N/2*step, lo_freq+N/2*step, N)
    
    while i<N:
    
        expt = ex.Experiment(lo_freq=freqs[i], rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    
        rf_tend = tstart+echo_duration+rf_pi_duration/2 # us
        rx_tstart = rf_tend+rx_wait # us
        rx_tend = rx_tstart + readout_duration  # us
        expt.add_flodict({
            # second tx0 pulse purely for loopback debugging
            'tx0': (np.array([tstart + (echo_duration - rf_pi2_duration)/2, tstart + (echo_duration + rf_pi2_duration)/2,
                         tstart + echo_duration - rf_pi_duration/2, rf_tend]), np.array([pi2_phase*rf_amp, 0, pi_phase*rf_amp, 0])),
            'rx0_en': ( np.array([rx_tstart, rx_tend]),  np.array([1, 0]) ),
            'tx_gate': (np.array([tstart + (echo_duration - rf_pi2_duration)/2- tx_gate_pre, tstart + (echo_duration + rf_pi2_duration)/2 + tx_gate_post,
                         tstart + echo_duration - rf_pi_duration/2- tx_gate_pre, rf_tend+ tx_gate_post]), np.array([1, 0, 1, 0])),
            'rx_gate': ( np.array([rx_tstart, rx_tend]), np.array([1, 0]) )
        })
    
        rxd, msg = expt.run() 
      
        for nS in range(nScans):
            print('nScan=%s'%(nS))
            rxd, msgs = expt.run()
            if nS ==0:
                n_rxd = rxd['rx0']
            else:
                n_rxd = np.concatenate((n_rxd, rxd['rx0']), axis=0)
    
        n_rxd = np.reshape(n_rxd, (nScans, len(rxd['rx0'])))
        n_rxd = np.average(n_rxd, axis=0) 
    
        dataobject:DataManager=DataManager(n_rxd, freqs[i], len(n_rxd), [], BW)
        f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency = dataobject.get_peakparameters()
        peakValsf.append(f_signalValue)
        
        expt.__del__()

        print(i)
        i = i+1
        
    
    return(peakValsf, freqs)
