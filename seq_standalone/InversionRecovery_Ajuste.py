"""
@author: Teresa
@modifield: J.M. algarín, february 25th 2022
"""
import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import date,  datetime 
import os
from scipy.io import savemat

def inversion_recovery_TGN(
    init_gpa= False,                 
    lo_freq=3.07264, 
    rf_amp=0.3, 
    rf_pi_duration=None, 
    rf_pi2_duration=32, 
    tr = 4,                 # s
    t_ini = 10e-3,      # s
    t_fin = 3,             # s
    N = 10):                
    
    BW=60*1e-3,                     # MHz
    n_rd=120,
    plotSeq =0
    point = 10
    
    if rf_pi_duration is None:
        rf_pi_duration = 2 * rf_pi2_duration
    
    rawData = {}
    rawData['larmorFreq'] = lo_freq*1e6
    rawData['rfAmp'] = rf_amp
    rawData['rfInTime'] = rf_pi_duration
    rawData['rfExtime'] = rf_pi2_duration
    rawData['TR'] = tr*1e-6
    rawData['tInvIni'] = t_ini*1e-6
    rawData['tInvFin'] = t_fin*1e-6
    rawData['nRepetitions'] = N
    rawData['bw'] = BW*1e6
    rawData['nRD'] = n_rd
    rawData['point'] = point
    
    t_adq = n_rd/BW
    
    tx_gate_pre = 15 # us, time to start the TX gate before each RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after an RF pulse ends
    
    rx_period = 1/BW

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))

    t_start = 20 
    t = t_ini*1e6
    blk_delay = 300
    tIR = np.geomspace(t_ini, t_fin, N)*1e6             # in us to be used in the sequence
    tr = tr*1e6
    for t in tIR:
        
        tx_t = np.array([t_start, t_start+rf_pi_duration,t_start+t+rf_pi2_duration/2,t_start+t+3*rf_pi2_duration/2])
        tx_a = np.array([1j*rf_amp,0,rf_amp,0])
        
        tx_gate_t = np.array([t_start-tx_gate_pre, t_start+rf_pi_duration+tx_gate_post,t_start+t+rf_pi2_duration/2-tx_gate_pre,t_start+t+3*rf_pi2_duration/2+tx_gate_post])
        tx_gate_a = np.array([1,0,1,0])
    
        readout_t = np.array([t_start, t_start+t+3*rf_pi2_duration/2+blk_delay,t_start+t+3*rf_pi2_duration/2+t_adq+blk_delay])
        readout_a = np.array([0,1,0])
        
        rx_gate_t = readout_t
        rx_gate_a = readout_a
        
        expt.add_flodict({
                        'tx0': (tx_t, tx_a),
                        'tx_gate': (tx_gate_t, tx_gate_a),
                        'rx0_en': (readout_t, readout_a),
                        'rx_gate': (rx_gate_t, rx_gate_a),
                        })
    
        expt.add_flodict({
                        'grad_vx': (np.array([t_start]),np.array([0]) ), 
                        'grad_vy': (np.array([t_start]),np.array([0]) ), 
                        'grad_vz': (np.array([t_start]),np.array([0]) ),
             })
        
        t_start += tr

    expt.add_flodict({
                        'grad_vx': (np.array([t_start]),np.array([0]) ), 
                        'grad_vy': (np.array([t_start]),np.array([0]) ), 
                        'grad_vz': (np.array([t_start]),np.array([0]) ),
             })


# Representar Secuencia o tomar los datos.
    tIR1 = np.geomspace(t_ini, t_fin, N)
    tIR2 = np.geomspace(t_ini, t_fin, 10*N)
    tIR
    if plotSeq==1:                
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        print('Running...')
        rxd, msgs = expt.run()
        print('End')
        data = rxd['rx0']*13.788
        expt.__del__()
        rawData['fullData'] = data
        dataIndiv = np.reshape(data,  (N, n_rd))
        dataIndiv = np.real(dataIndiv[:, point]*np.exp(-1j*(np.angle(dataIndiv[0, 1])+np.pi)))
        results = np.transpose(np.array([tIR1, dataIndiv]))
        rawData['signalVsTime'] = results
        
        # For 1 component
        fitData1, xxx = curve_fit(func1, results[:, 0],  results[:, 1])
        print('For one component:')
        print('mA', round(fitData1[0], 1))
        print('T1', round(fitData1[1]*1e3), ' ms')
        rawData['T11'] = fitData1[1]*1e3
        rawData['M1'] = fitData1[0]
        
        # For 2 components
        fitData2, xxx = curve_fit(func2, results[:, 0],  results[:, 1])
        print('For two components:')
        print('Ma', round(fitData2[0], 1))
        print('Mb', round(fitData2[2], 1))
        print('T1a', round(fitData2[1]*1e3), ' ms')
        print('T1b', round(fitData2[3]*1e3), ' ms')
        rawData['T12'] = [fitData2[1], fitData2[3]]
        rawData['M2'] = [fitData2[0], fitData2[2]]
        
#        # For 3 components
#        fitData3, xxx = curve_fit(func3, results[:, 0],  results[:, 1])
#        print('For two components:')
#        print('Ma', round(fitData3[0], 1), ' ms')
#        print('Mb', round(fitData3[2], 1), ' ms')
#        print('Mc', round(fitData3[4], 1), ' ms')
#        print('T1a', round(fitData3[1]*1e3), ' ms')
#        print('T1b', round(fitData3[3]*1e3), ' ms')
#        print('T1c', round(fitData3[5]*1e3), ' ms')
#        rawData['T13'] = [fitData3[1], fitData3[3], fitData3[5]]
#        rawData['M3'] = [fitData3[0], fitData3[2], fitData3[4]]
        
        # Save data
        name = saveMyData(rawData)
        
        # Plots
        plt.figure(2, figsize=(5, 5))
        plt.plot(results[:, 0], results[:, 1], 'o')
        plt.plot(tIR2, func1(tIR2, *fitData1))
        plt.plot(tIR2, func2(tIR2, *fitData2))
#        plt.plot(tIR2, func3(tIR2, *fitData3))
        plt.title(name)
        plt.xscale('log')
        plt.xlabel('t(ms)')
        plt.ylabel('Signal (mV)')
        plt.legend(['Experimental', 'Fitting 1 component', 'Fitting 2 components'])
        plt.title(name)
        plt.show()
        

def func1(x, m, t1):
    return m*(1-2*np.exp(-x/t1))

def func2(x, ma, t1a, mb, t1b):
    return ma*(1-2*np.exp(-x/t1a))+mb*(1-2*np.exp(-x/t1b))

def func3(x, ma, t1a, mb, t1b, mc, t1c):
    return ma*(1-2*np.exp(-x/t1a))+mb*(1-2*np.exp(-x/t1b))+mc*(1-2*np.exp(-x/t1c))

def saveMyData(rawData):
    # Save data
    dt = datetime.now()
    dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
    dt2 = date.today()
    dt2_string = dt2.strftime("%Y.%m.%d")
    if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
        os.makedirs('experiments/acquisitions/%s' % (dt2_string))
    if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
        os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
    rawData['name'] = "%s.%s.mat" % ("IR",dt_string)
    savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, "IR",dt_string),  rawData)
    return rawData['name']

if __name__ == "__main__":
    inversion_recovery_TGN()

