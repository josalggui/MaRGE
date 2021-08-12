#!/usr/bin/env python3
import sys
sys.path.append('../marcos_client')
import numpy as np
import experiment as ex
import matplotlib.pyplot as plt
import scipy.signal as sig
import pdb
st = pdb.set_trace

def fid(self, plotSeq):

    dbg_sc=self.dbg_sc
    lo_freq=self.lo_freq
    rf_amp=self.rf_amp
    rf_pi2_duration =self.rf_pi2_duration 
    rf_tstart=self.rf_tstart
    rx_wait=self.rx_wait*1e3
    BW=self.BW*1e-3
    readout_duration=self.readout_duration*1e3
    nScans = self.nScans
    oversampling_factor = self.oversampling_factor
        
    rx_period=1/(BW*oversampling_factor)
    
    ## All times are in the context of a single TR, starting at time 0
    init_gpa = True

#    phase_amps = np.linspace(phase_amp, -phase_amp, trs)
    rf_tend = rf_tstart + rf_pi2_duration  # us

    rx_tstart = rf_tend+rx_wait # us
    rx_tend = rx_tstart + readout_duration  # us

    tx_gate_pre = 15 # us, time to start the TX gate before the RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after the RF pulse ends


    def fid_tr(tstart):
        rx_tcentre = (rx_tstart + rx_tend) / 2
        value_dict = {
            # Pulse of 90º + second tx0 pulse purely for loopback debugging
            'tx0': ( np.array([rf_tstart, rf_tend,   rx_tcentre - 10, rx_tcentre + 10]) + tstart,
                     np.array([rf_amp,0,  dbg_sc*(1 + 0.5j),0]) ),
            'rx0_en': ( np.array([rx_tstart, rx_tend]) + tstart, np.array([1, 0]) ),
             'tx_gate': ( np.array([rf_tstart - tx_gate_pre, rf_tend + tx_gate_post]) + tstart, np.array([1, 0]) )
        }

        return value_dict

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa)
    true_rx_period = expt.get_rx_ts()[0]
    true_BW = 1/true_rx_period
    true_BW = true_BW/oversampling_factor
    
    tr_t = 20 # start the first TR at 20us
    expt.add_flodict( fid_tr( tr_t) )

    if plotSeq==1:
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        for nS in range(nScans):
            print('nScan=%s'%(nS))
            rxd, msgs = expt.run()
            #Decimate
            rxd['rx0'] = sig.decimate(rxd['rx0'], oversampling_factor, ftype='fir')
            rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
            
            if nS ==0:
                n_rxd = rxd['rx0']
            else:
                n_rxd = np.concatenate((n_rxd, rxd['rx0']), axis=0)
    
        n_rxd = np.reshape(n_rxd, (nScans, len(rxd['rx0'])))
        data_avg = np.average(n_rxd, axis=0) 
       
        expt.__del__()
        return n_rxd, msgs, data_avg
        
