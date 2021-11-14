"""
@author: Teresa
"""
import sys
sys.path.append('../marcos_client')
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt

def inversion_recovery_TGN(
    init_gpa= True,                 
    lo_freq=3.08, 
    rf_amp=0.3, 
    rf_pi_duration=None, 
    rf_pi2_duration=35, 
    tr = 3000*1e3, 
    t_ini = 200*1e3,
    t_fin = 2000*1e3,
    N = 5,
    BW=60*1e-3, 
    n_rd=120):
    
    if rf_pi_duration is None:
        rf_pi_duration = 2 * rf_pi2_duration
        
    t_adq = n_rd/BW
    
    tx_gate_pre = 15 # us, time to start the TX gate before each RF pulse begins
    tx_gate_post = 1 # us, time to keep the TX gate on after an RF pulse ends
    
    rx_period = 1/BW

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))

    t_start = 20 
    t = t_ini
    blk_delay = 300
    tIR = np.geomspace(t_ini, t_fin, N)
    #tIR = np.linspace(t_ini, t_fin, N)
    
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
    plotSeq =0
    if plotSeq==1:                
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        print('Running...')
        rxd, msgs = expt.run()
        print('End')
        data = rxd['rx0']*13.788
        dataIndiv = np.reshape(data,  (N, n_rd))
        plt.figure(1)
        plt.plot(np.abs(dataIndiv[0, :]),  'o')
#        plt.plot(np.abs(dataIndiv[1, :]),  'go')
#        plt.plot(np.abs(dataIndiv[2, :]),  'bo')
        #Tratamiento resultado
        results = []
        cont = 0
        for t in tIR:
            if t == t_ini: 
                results = [[t, max(np.abs(dataIndiv[cont]))]]
            else: 
                results = np.append(results, [[t, max(np.abs(dataIndiv[cont]))]],axis=0)
            cont += 1
        plt.figure(2)
        plt.plot(results[:, 0]*1e-3, results[:, 1], 'o')
        plt.xscale('log')
        plt.xlabel('t(ms)')
        plt.ylabel('FID max (mV)')
        T1 = results[:,0][np.argmin(results[:,1])]*1e-3/np.log(2)
        plt.suptitle('T1 = '+str(T1))
        plt.show()

if __name__ == "__main__":
    inversion_recovery_TGN()

