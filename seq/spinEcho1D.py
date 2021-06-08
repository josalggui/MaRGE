import sys
sys.path.append('../marcos_client')
import numpy as np
import experiment as ex
#from local_config import fpga_clk_freq_MHz
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

def spin_echo1D(self, plotSeq):
#                    plot_rx=True, init_gpa=False,
#                    dbg_sc=0.5, # set to 0 to avoid RF debugging pulses in each RX window, otherwise amp between 0 or 1
#                    lo_freq=0.2, # MHz
#                    rf_amp=1, # 1 = full-scale
#                    trs=1, 
#                    rf_pi2_duration=50, # us, rf pi/2 pulse length
#                    rf_pi_duration=None, # us, rf pi pulse length  - if None then automatically gets set to 2 * rf_pi2_duration
#
#                    # spin-echo properties
#                    echo_duration=2000, # us, time from the centre of one echo to centre of the next
#                    readout_duration=500, # us, time in the centre of an echo when the readout occurs
#                    rx_period=10/3, # us, 3.333us, 300 kHz rate
#                    # (must at least be longer than readout_duration + trap_ramp_duration)
#                    ):
    init_gpa=True                   
    dbg_sc=self.dbg_sc
    lo_freq=self.lo_freq
    rf_amp=self.rf_amp
    trs=self.trs
    rf_pi_duration=None
    rf_pi2_duration=self.rf_pi2_duration
    echo_duration=self.echo_duration
    readout_duration=self.readout_duration
    BW=self.BW
    shim_x: float = self.shim[0]
#    shim_y: float = self.shim[1]
#    shim_z: float = self.shim[2]
    readout_amp=self.readout_amp
    readout_grad_duration=self.readout_grad_duration
    trap_ramp_duration=self.trap_ramp_duration   

    trap_ramp_pts=20
    phase_start_amp=0.6
    slice_start_amp=0.3
    rx_period=1/BW
    phase_grad_duration=150
    grad_readout_delay=8.83    # readout amplifier delay
    grad_phase_delay=8.83
    grad_slice_delay=8.83
   
    """
    readout gradient: x
    phase gradient: y
    slice/partition gradient: z
    """
    
    tr_pause_duration=2000
    echos_per_tr=1 # number of spin echoes (180 pulses followed by readouts) to do
                    
    if rf_pi_duration is None:
        rf_pi_duration = 2 * rf_pi2_duration
    
    phase_amps = np.linspace(phase_start_amp, -phase_start_amp, echos_per_tr)
    slice_amps = np.linspace(slice_start_amp, -slice_start_amp, self.trs)

    # create appropriate waveforms for each echo, based on start time, echo index and TR index
    # note: echo index is 0 for the first interval (90 pulse until first 180 pulse) thereafter 1, 2 etc between each 180 pulse
   
    def rf_wf(tstart, echo_idx):
        rx_tstart=tstart + (echo_duration - readout_duration)/2+ tr_pause_duration
        rx_tend=tstart + (echo_duration + readout_duration)/2+ tr_pause_duration
        rx_tcentre = (rx_tstart + rx_tend) / 2
        pi2_phase = 1 # x
        pi_phase = 1j # y
        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (echo_duration - rf_pi2_duration)/2, tstart + (echo_duration + rf_pi2_duration)/2,
                             tstart + echo_duration - rf_pi_duration/2, rx_tcentre - 20, rx_tcentre + 20]), np.array([pi2_phase*rf_amp, 0, pi_phase*rf_amp, dbg_sc*(1 + 0.5j), 0])                        
        elif echo_idx == echos_per_tr:
            # finish final RF pulse
            return np.array([tstart + rf_pi_duration/2]), np.array([0])
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2, tstart + echo_duration - rf_pi_duration/2]), np.array([0, pi_phase]) * rf_amp

    def tx_gate_wf(tstart, echo_idx):
        tx_gate_pre = 2 # us, time to start the TX gate before each RF pulse begins
        tx_gate_post = 1 # us, time to keep the TX gate on after an RF pulse ends

        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (echo_duration - rf_pi2_duration)/2 - tx_gate_pre,
                             tstart + (echo_duration + rf_pi2_duration)/2 + tx_gate_post,
                             tstart + echo_duration - rf_pi_duration/2 - tx_gate_pre]), \
                             np.array([1, 0, 1])
        elif echo_idx == echos_per_tr:
            # finish final RF pulse
            return np.array([tstart + rf_pi_duration/2 + tx_gate_post]), np.array([0])
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2 + tx_gate_post, tstart + echo_duration - rf_pi_duration/2 - tx_gate_pre]), \
                np.array([0, 1])

    def readout_wf(tstart, echo_idx):
        if echo_idx != 0:
            return np.array([tstart + (echo_duration - readout_duration)/2, tstart + (echo_duration + readout_duration)/2 ]), np.array([1, 0])
        else:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
            
            
    def readout_grad_wf(tstart, echo_idx):
        if echo_idx == 0:
#            return trap_cent(tstart + self.echo_duration*3/4, readout_amp, readout_grad_duration/2,
#                             trap_ramp_duration, trap_ramp_pts)
            return trap_cent(tstart + echo_duration/2 + rf_pi2_duration/2+trap_ramp_duration/2+readout_grad_duration/4, readout_amp, readout_grad_duration/2,
                             trap_ramp_duration, trap_ramp_pts)
        else:
            return trap_cent(tstart + self.echo_duration/2-grad_readout_delay, readout_amp, readout_grad_duration,
                             trap_ramp_duration, trap_ramp_pts)

    def phase_grad_wf(tstart, echo_idx):
#        t1, a1 = trap_cent(tstart + (self.echo_duration - phase_grad_interval)/2, phase_amps[echo_idx-1], phase_grad_duration,
#                           trap_ramp_duration, trap_ramp_pts)
#        t2, a2 = trap_cent(tstart + (self.echo_duration + phase_grad_interval)/2, -phase_amps[echo_idx-1], phase_grad_duration,
#                           trap_ramp_duration, trap_ramp_pts)
        t1, a1 = trap_cent(tstart + (rf_pi_duration+phase_grad_duration-trap_ramp_duration)/2+trap_ramp_duration-grad_phase_delay, phase_amps[echo_idx-1], phase_grad_duration,
                           trap_ramp_duration, trap_ramp_pts)
        t2, a2 = trap_cent(tstart + (echo_duration + readout_duration+trap_ramp_duration)/2+trap_ramp_duration-grad_phase_delay, -phase_amps[echo_idx-1], phase_grad_duration,
                           trap_ramp_duration, trap_ramp_pts)   
        if echo_idx == 0:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
        elif echo_idx == echos_per_tr: # last echo, don't need 2nd trapezoids
            return t1, a1
        else: # otherwise do both trapezoids
            return np.hstack([t1, t2]), np.hstack([a1, a2])

    def slice_grad_wf(tstart, echo_idx, tr_idx):
#        t1, a1 = trap_cent(tstart + (self.echo_duration - phase_grad_interval)/2, slice_amps[tr_idx], phase_grad_duration,
#                           trap_ramp_duration, trap_ramp_pts)
#        t2, a2 = trap_cent(tstart + (self.echo_duration + phase_grad_interval)/2, -slice_amps[tr_idx], phase_grad_duration,
#                           trap_ramp_duration, trap_ramp_pts)
        t1, a1 = trap_cent(tstart + (rf_pi_duration+phase_grad_duration-trap_ramp_duration)/2+trap_ramp_duration-grad_phase_delay, slice_amps[tr_idx], phase_grad_duration,
                           trap_ramp_duration, trap_ramp_pts)
        t2, a2 = trap_cent(tstart + (echo_duration + readout_duration+trap_ramp_duration)/2+trap_ramp_duration-grad_slice_delay, -slice_amps[tr_idx], phase_grad_duration,
                           trap_ramp_duration, trap_ramp_pts)   
        if echo_idx == 0:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
        elif echo_idx == echos_per_tr: # last echo, don't need 2nd trapezoids
            return t1, a1
        else: # otherwise do both trapezoids
            return np.hstack([t1, t2]), np.hstack([a1, a2])


    expt = ex.Experiment(lo_freq=lo_freq, rx_t=rx_period, init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
    # gpa_fhdo_offset_time in microseconds; offset between channels to
    # avoid parallel updates (default update rate is 0.2 Msps, so
    # 1/0.2 = 5us, 5 / 3.1 gives the offset between channels; extra
    # 0.1 for a safety margin))

    global_t = 20 # start the first TR at 20us
    for tr in range(trs):
        for echo in range(echos_per_tr + 1):
            tx_t, tx_a = rf_wf(global_t, echo)
            tx_gate_t, tx_gate_a = tx_gate_wf(global_t, echo)
            readout_t, readout_a = readout_wf(global_t, echo)
            rx_gate_t, rx_gate_a = readout_wf(global_t, echo)
            readout_grad_t, readout_grad_a = readout_grad_wf(global_t, echo)
            phase_grad_t, phase_grad_a = phase_grad_wf(global_t, echo)
            slice_grad_t, slice_grad_a = slice_grad_wf(global_t, echo, tr)

            expt.add_flodict({
                'tx0': (tx_t, tx_a),
                'grad_vx': (readout_grad_t, readout_grad_a+shim_x),
                'rx0_en': (readout_t, readout_a),
                'tx_gate': (tx_gate_t, tx_gate_a),
                'rx_gate': (rx_gate_t, rx_gate_a),
            })
            
            global_t += echo_duration
            
        global_t += tr_pause_duration
        
    if plotSeq==1:
        expt.plot_sequence()
        plt.show()
        expt.__del__()
    elif plotSeq==0:
        rxd, msgs = expt.run()
        expt.__del__()
        return rxd['rx0'], msgs


