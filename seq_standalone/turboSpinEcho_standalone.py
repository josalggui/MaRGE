import sys
sys.path.append('../marcos_client')
import numpy as np
import experiment as ex
from configs.hw_config import Gx_factor
from configs.hw_config import Gy_factor
from configs.hw_config import Gz_factor
import matplotlib.pyplot as plt
import pdb
st = pdb.set_trace


def trapezoid(plateau_a, total_t, ramp_t, ramp_pts, total_t_end_to_end=True, base_a=0):
    """Helper function that just generates a Numpy array starting at time
    0 and ramping down at time total_t, containing a trapezoid going from a
    level base_a to plateau_a, with a rising ramp of duration ramp_t and
    sampling period ramp_ts."""

    # ramp_pts = int( np.ceil(ramp_t/ramp_ts) ) + 1
    rise_ramp_times = np.linspace(0, ramp_t, ramp_pts)
    rise_ramp = np.linspace(base_a, plateau_a, ramp_pts)

    # [1: ] because the first element of descent will be repeated
    descent_t = total_t - ramp_t if total_t_end_to_end else total_t
    t = np.hstack([rise_ramp_times, rise_ramp_times[:-1] + descent_t])
    a = np.hstack([rise_ramp, np.flip(rise_ramp)[1:]])
    return t, a


def trap_cent(centre_t, plateau_a, trap_t, ramp_t, ramp_pts, base_a=0):
    """Like trapezoid, except it generates a trapezoid shape around a centre
    time, with a well-defined area given by its amplitude (plateau_a)
    times its time (trap_t), which is defined from the start of the
    ramp-up to the start of the ramp-down, or (equivalently) from the
    centre of the ramp-up to the centre of the ramp-down. All other
    parameters are as for trapezoid()."""
    t, a = trapezoid(plateau_a, trap_t, ramp_t, ramp_pts, False, base_a)
    return t + centre_t - (trap_t + ramp_t)/2, a


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def rect_cent(centre_t, plateau_a, rect_t, base_a=0):
    """It idefines a rectangle around a centre time, with a given amplitude."""
    
    t = np.array([centre_t-rect_t/2, centre_t+rect_t/2])
    a = np.array([plateau_a, base_a])
    return t, a


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def getIndex(n_ph, g_amps, echos_per_tr, sweep_mode):
#    print(n[1]/2/echos_per_tr)
    n2ETL=np.int32(n_ph/2/echos_per_tr)
    ind:np.int32 = []
#    n_ph = n[1]
    if n_ph==1:
        ind=1
    
    if sweep_mode==0:   # Sequential for T2 contrast
        for ii in range(np.int32(n_ph/echos_per_tr)):
            if ii==0:
                ind = np.linspace(0, n_ph-1, echos_per_tr)+ii
            else:
                ind=np.concatenate((ind, np.linspace(0, n_ph-1, echos_per_tr)+ii), axis=0)
    elif sweep_mode==1: # Center-out for T1 contrast
        if echos_per_tr==n_ph:
            ind = np.linspace(np.int32(n_ph/2)-1, -n2ETL, echos_per_tr)
            ind2 = np.linspace(np.int32(n_ph/2), n_ph-1, np.int32(echos_per_tr/2))
            ind[1::2] = ind2
#            ind = np.concatenate((np.linspace(np.int32(n_ph/2)-1, 0, np.int32(echos_per_tr/2)),np.linspace(np.int32(n_ph/2), n_ph-1, np.int32(echos_per_tr/2))), axis=0)
        else:
            for ii in range(n2ETL):
                if ii==0:
                    ind = np.linspace(np.int32(n_ph/2)-1, 1, echos_per_tr)
                else:
                    ind = np.concatenate((ind, np.linspace(np.int32(n_ph/2)-1-ii, n2ETL-1-ii, echos_per_tr)), axis=0)
                ind = np.concatenate((ind,np.linspace(np.int32(n_ph/2)+ii, n_ph-n2ETL+ii, echos_per_tr)), axis=0)
    elif sweep_mode==2:
        if echos_per_tr==n_ph:
            ind=np.linspace(0, n_ph-1, echos_per_tr)
        else:
            for ii in range(n2ETL):
                if ii==0:
                    ind = np.linspace(0, np.int32(n_ph/2)-1, echos_per_tr)
                else:
                    ind = np.concatenate(ind,np.linspace(ii, ii+np.int32(n_ph/2)-1, echos_per_tr))
                ind = np.concatenate(ind,np.linspace(n_ph-1-ii, np.int32(n_ph/2)-ii, echos_per_tr));

    return np.int32(ind)


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


def turbo_spin_echo(
    init_gpa=True,                 
    lo_freq=3.0395, 
    rf_amp=0.4, 
    rf_pi_duration=None, 
    rf_pi2_duration=45, 
    echo_duration=100*1e3, 
    tr_duration=500*1e3, 
    BW=31, 
    shim_x: float = 0, 
    shim_y: float = 0, 
    shim_z: float = 0, 
    nScans=1, 
    n_rd:int=8000, 
    n_ph:int=1, 
    n_sl:int=1, 
    fov_rd:int=2000000*1e-2, 
    fov_ph:int=20*1e-2, 
    fov_sl:int=20*1e-2, 
    trap_ramp_duration=100, 
    phase_grad_duration=500, 
    echos_per_tr=1, 
    rd_preemph_factor:float=1, 
    sweep_mode = 1, 
    par_acq_factor=0):
   
    BW=BW*1e-3
#    trap_ramp_pts=np.int32(trap_ramp_duration*0.2)    # 0.2 puntos/ms
    trap_ramp_pts = 10
    grad_readout_delay=9   #8.83    # readout amplifier delay
    grad_phase_delay=9      #8.83
    grad_slice_delay=9        #8.83
    rx_period=1/BW
    """
    readout gradient: x
    phase gradient: y
    slice/partition gradient: z
    """

    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    true_rx_period = expt.get_rx_ts()[0]
    true_BW = 1/true_rx_period
    readout_duration = n_rd/true_BW
                    
    if rf_pi_duration is None:
        rf_pi_duration = 2 * rf_pi2_duration
        
#    SweepMode=1
    
    # Calibration constans to change from T/m to DAC amplitude
    
    gammaB = 42.56e6    # Gyromagnetic ratio in Hz/T
    # Get readout, phase and slice amplitudes
    # Readout gradient amplitude
    Grd = true_BW*1e6/(gammaB*fov_rd)
    # Phase gradient amplitude
    if (n_ph==1):   
        Gph=0
    else:
        Gph = n_ph/(2*gammaB*fov_ph*phase_grad_duration*1e-6);
    # Slice gradient amplitude
    if (n_sl==1):
        Gsl=0
    else:
        Gsl = n_sl/(2*gammaB*fov_sl*phase_grad_duration*1e-6);
    
    # Get the phase gradient vector
    if(n_ph>1):
        phase_amps = np.linspace(-Gph, Gph, n_ph+1)
        phase_amps = phase_amps[1:n_ph+1]
    else:
        phase_amps = np.linspace(-Gph, Gph, n_ph)    
    ind = getIndex(n_ph, phase_amps, echos_per_tr, sweep_mode)
    phase_amps=phase_amps[ind]
    
    # Get the slice gradient vector
    if (n_sl>1):
        slice_amps = np.linspace(-Gsl, Gsl,  n_sl+1)
        slice_amps = slice_amps[1:n_sl+1]
    else:
        slice_amps = np.linspace(-Gsl, Gsl, n_sl)


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


    def rf_wf(tstart, echo_idx):
        pi2_phase = 1 # x
        pi_phase = 1j # y
        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (echo_duration - rf_pi2_duration)/2, tstart + (echo_duration + rf_pi2_duration)/2,
                             tstart + echo_duration - rf_pi_duration/2]), np.array([pi2_phase*rf_amp, 0, pi_phase*rf_amp])                        
#        elif echo_idx == self.echos_per_tr and ph==n_ph and sl==n_sl-1:
#            # last echo of the full sequence
#            return np.array([tstart + rf_pi_duration/2, tr_duration-echo_duration*echos_per_tr]), np.array([0, 0])
        elif echo_idx == echos_per_tr:
            # last echo on any other echo train
            return np.array([tstart + rf_pi_duration/2]), np.array([0])
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2, tstart + echo_duration - rf_pi_duration/2]), np.array([0, pi_phase*rf_amp])


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


    def tx_gate_wf(tstart, echo_idx, ph, sl):
        tx_gate_pre = 15 # us, time to start the TX gate before each RF pulse begins
        tx_gate_post = 1 # us, time to keep the TX gate on after an RF pulse ends

        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (echo_duration - rf_pi2_duration)/2 - tx_gate_pre,
                             tstart + (echo_duration + rf_pi2_duration)/2 + tx_gate_post,
                             tstart + echo_duration - rf_pi_duration/2 - tx_gate_pre]), \
                             np.array([1, 0, 1])
        elif echo_idx == echos_per_tr and ph==n_ph and sl==n_sl-1:
            return np.array([tstart + rf_pi_duration/2 + tx_gate_post, tstart+tr_duration-echo_duration*echos_per_tr]), np.array([0, 0])
        elif echo_idx == echos_per_tr:
            # finish final RF pulse
            return np.array([tstart + rf_pi_duration/2 + tx_gate_post]), np.array([0])
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2 + tx_gate_post, tstart + echo_duration - rf_pi_duration/2 - tx_gate_pre]), \
                np.array([0, 1])


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


    def readout_wf(tstart, echo_idx):
        if echo_idx == 0:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
#        elif echo_idx==echos_per_tr:
#            return np.array([tstart + (echo_duration - readout_duration)/2, tstart + (echo_duration + readout_duration)/2, tstart+echo_duration+tr_duration-echos_per_tr*echo_duration ]), np.array([1, 0, 0])
        else:
            return np.array([tstart + (echo_duration - readout_duration)/2, tstart + (echo_duration + readout_duration)/2 ]), np.array([1, 0])


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


    def readout_grad_wf(tstart, echo_idx):
        if echo_idx == 0:
#            return trap_cent(tstart+echo_duration/2+rf_pi2_duration/2+trap_ramp_duration+readout_duration/2-grad_readout_delay, Grd/2*rd_preemph_factor, readout_duration+trap_ramp_duration,
#                             trap_ramp_duration, trap_ramp_pts)
            return rect_cent(tstart+echo_duration/2+rf_pi2_duration/2+trap_ramp_duration+readout_duration/2-grad_readout_delay, Grd/2.0*rd_preemph_factor,  readout_duration+2*trap_ramp_duration)
        else:
#            return trap_cent(tstart + echo_duration/2-grad_readout_delay, Grd, readout_duration+trap_ramp_duration,
#                             trap_ramp_duration, trap_ramp_pts)
            return rect_cent(tstart+echo_duration/2-grad_readout_delay, Grd, readout_duration+2*trap_ramp_duration)


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


    def phase_grad_wf(tstart, echo_idx, ph):
#        t1, a1 = trap_cent(tstart + (rf_pi_duration+phase_grad_duration-trap_ramp_duration)/2+trap_ramp_duration-grad_phase_delay,
#                            phase_amps[ph-1], phase_grad_duration, trap_ramp_duration, trap_ramp_pts)
#        t2, a2 = trap_cent(tstart + echo_duration/2 + readout_duration/2+phase_grad_duration/2+trap_ramp_duration/2-grad_phase_delay,
#                            -phase_amps[ph-1], phase_grad_duration, trap_ramp_duration, trap_ramp_pts)
        t1, a1 = rect_cent(tstart+rf_pi_duration/2+phase_grad_duration/2+trap_ramp_duration-grad_phase_delay, phase_amps[ph-1], 
                            phase_grad_duration+2*trap_ramp_duration)
        t2, a2 = rect_cent(tstart+echo_duration/2+readout_duration/2+trap_ramp_duration+phase_grad_duration/2-grad_phase_delay, -phase_amps[ph-1], 
                            phase_grad_duration+2*trap_ramp_duration)
        if echo_idx == 0:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
        elif echo_idx == echos_per_tr: # last echo, don't need 2nd trapezoids
            return t1, a1
        else: # otherwise do both trapezoids
            return np.hstack([t1, t2]), np.hstack([a1, a2])


#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


    def slice_grad_wf(tstart, echo_idx,  sl):
#        t1, a1 = trap_cent(tstart + (rf_pi_duration+phase_grad_duration-trap_ramp_duration)/2+trap_ramp_duration-grad_phase_delay, slice_amps[sl], phase_grad_duration,
#                           trap_ramp_duration, trap_ramp_pts)
#        t2, a2 = trap_cent(tstart +echo_duration/2 + readout_duration/2+phase_grad_duration/2+trap_ramp_duration/2-grad_slice_delay, -slice_amps[sl], phase_grad_duration,
#                           trap_ramp_duration, trap_ramp_pts)
        t1, a1 = rect_cent(tstart+rf_pi_duration/2+trap_ramp_duration+phase_grad_duration/2-grad_slice_delay,
                            slice_amps[sl], phase_grad_duration+2*trap_ramp_duration)
        t2, a2 = rect_cent(tstart+echo_duration/2+readout_duration/2+trap_ramp_duration+phase_grad_duration/2-grad_slice_delay,
                            -slice_amps[sl], phase_grad_duration+2*trap_ramp_duration)
        if echo_idx == 0:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
        elif echo_idx == echos_per_tr: # last echo, don't need 2nd trapezoids
            return t1, a1
        else: # otherwise do both trapezoids
            return np.hstack([t1, t2]), np.hstack([a1, a2])
            

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************


#    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    # gpa_fhdo_offset_time in microseconds; offset between channels to
    # avoid parallel updates (default update rate is 0.2 Msps, so
    # 1/0.2 = 5us, 5 / 3.1 gives the offset between channels; extra
    # 0.1 for a safety margin))

    global_t = 20 # start the first TR at 20us
#    for nS in range(nScans):
    if par_acq_factor==0:
        n_sl_par = n_sl
    else:
        n_sl_par = np.int32(n_sl/2)+par_acq_factor
    for sl in range(n_sl_par):
        for ph_block in range(np.int32(n_ph/echos_per_tr)):
            for echo_idx in range(1+echos_per_tr):
                tx_t, tx_a = rf_wf(global_t, echo_idx)
                tx_gate_t, tx_gate_a = tx_gate_wf(global_t, echo_idx, ph_block*echos_per_tr+echo_idx, sl)
                readout_t, readout_a = readout_wf(global_t, echo_idx)
                rx_gate_t, rx_gate_a = readout_wf(global_t, echo_idx)
                readout_grad_t, readout_grad_a = readout_grad_wf(global_t, echo_idx)
                phase_grad_t, phase_grad_a = phase_grad_wf(global_t, echo_idx,  ph_block*echos_per_tr+echo_idx)
                slice_grad_t, slice_grad_a = slice_grad_wf(global_t, echo_idx,  sl)

                expt.add_flodict({
                    'tx0': (tx_t, tx_a),
                    'grad_vx': (readout_grad_t, readout_grad_a/(Gx_factor/1000)/10+shim_x),
                    'grad_vy': (phase_grad_t, phase_grad_a/(Gy_factor/1000)/10+shim_y),
                    'grad_vz': (slice_grad_t, slice_grad_a/(Gz_factor/1000)/10+shim_z), 
                    'rx0_en': (readout_t, readout_a),
                    'tx_gate': (tx_gate_t, tx_gate_a),
                    'rx_gate': (rx_gate_t, rx_gate_a),
                })
                global_t += echo_duration
            
            global_t += tr_duration-echo_duration*echos_per_tr
                
                
#    expt.plot_sequence()
#    plt.show()

    for nS in range(nScans):
        rxd, msgs = expt.run()
        rxd['rx0'] = rxd['rx0']*13.788   # Here I normalize to get the result in mV
        if nS ==0:
            n_rxd = rxd['rx0']
        else:
            n_rxd = np.concatenate((n_rxd, rxd['rx0']), axis=0)
    if par_acq_factor==0 and nScans>1:
        n_rxd = np.reshape(n_rxd, (nScans, n_sl*n_ph*n_rd))
        data_avg = np.average(n_rxd, axis=0) 
    elif par_acq_factor>0 and nScans>1:
        n_rxd = np.reshape(n_rxd, (nScans, n_sl_par*n_ph*n_rd))
        data_avg = np.average(n_rxd, axis=0) 
    else:
        data_avg = n_rxd
    
    expt.__del__()
        
    # Reorganize the data matrix according to the sweep mode
    rxd_temp1 = np.reshape(data_avg, (n_sl_par, n_ph, n_rd))
    rxd_temp2 = rxd_temp1*0
    for ii in range(n_ph):
        ind_temp = ind[ii]
        rxd_temp2[:, ind_temp, :] = rxd_temp1[:, ii, :]
    
    rxd_temp3:complex = np.zeros((n_sl, n_ph, n_rd))+1j*np.zeros((n_sl, n_ph, n_rd))
    rxd_temp3[0:n_sl_par, :,  :] = rxd_temp2
    data_avg = np.reshape(rxd_temp3, -1)    # -1 means reshape to 1D array

    plt.plot(data_avg)
    plt.show()

if __name__ == "__main__":
    
    turbo_spin_echo()
