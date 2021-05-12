import numpy as np
import experiment as ex
import sys
sys.path.append('../marcos_client')
from local_config import ip_address, fpga_clk_freq_MHz, grad_board
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

def turbo_spin_echo(self):
    trap_ramp_pts=5
    rf_pi_duration=None
    grad_board = "ocra1"

#                   plot_rx=False, init_gpa=False,
#                    dbg_sc=0.5, # set to 0 to avoid RF debugging pulses in each RX window, otherwise amp between 0 or 1
#                    lo_freq=0.2, # MHz
#                    rf_amp=1, # 1 = full-scale
#
#                    rf_pi2_duration=50, # us, rf pi/2 pulse length

#
#                    # trapezoid properties - shared between all gradients for now
#                    trap_ramp_duration=50, # us, ramp-up/down time
#                    trap_ramp_pts=5, # how many points to subdivide ramp into
#
#                    # spin-echo properties
#                    echos_per_tr=5, # number of spin echoes (180 pulses followed by readouts) to do
#                    echo_duration=2000, # us, time from the centre of one echo to centre of the next
#
#                    readout_amp=0.8, # 1 = gradient full-scale
#                    readout_duration=500, # us, time in the centre of an echo when the readout occurs
#                    rx_period=10/3, # us, 3.333us, 300 kHz rate
#                    readout_grad_duration=700, # us, readout trapezoid lengths (mid-ramp-up to mid-ramp-down)
#                    # (must at least be longer than readout_duration + trap_ramp_duration)
#
#                    phase_start_amp=0.6, # 1 = gradient full-scale, starting amplitude (by default ramps from +ve to -ve in each echo)
#                    phase_grad_duration=150, # us, phase trapezoid lengths (mid-ramp-up to mid-ramp-down)
#                    phase_grad_interval=1200, # us, interval between first phase trapezoid and its negative-sign counterpart within a single echo
#
#                    # slice trapezoid timing is the same as phase timing
#                    slice_start_amp=0.3, # 1 = gradient full-scale, starting amplitude (by default ramps from +ve to -ve in each TR)
#
#                    tr_pause_duration=3000, # us, length of time to pause from the end of final echo's RX pulse to start of next TR
#                    trs=5 # number of TRs
#                    
    """
    readout gradient: x
    phase gradient: y
    slice/partition gradient: z
    """

    if rf_pi_duration is None:
        rf_pi_duration = 2 * self.rf_pi2_duration

    phase_amps = np.linspace(self.phase_start_amp, -self.phase_start_amp, self.echos_per_tr)
    slice_amps = np.linspace(self.slice_start_amp, -self.slice_start_amp, self.trs)

    # create appropriate waveforms for each echo, based on start time, echo index and TR index
    # note: echo index is 0 for the first interval (90 pulse until first 180 pulse) thereafter 1, 2 etc between each 180 pulse
    def rf_wf(tstart, echo_idx):
        pi2_phase = 1 # x
        pi_phase = 1j # y
        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (self.echo_duration - self.rf_pi2_duration)/2, tstart + (self.echo_duration + self.rf_pi2_duration)/2,
                             tstart + self.echo_duration - rf_pi_duration/2]), np.array([pi2_phase, 0, pi_phase]) * self.rf_amp
        elif echo_idx == self.echos_per_tr:
            # finish final RF pulse
            return np.array([tstart + rf_pi_duration/2]), np.array([0])
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2, tstart + self.echo_duration - rf_pi_duration/2]), np.array([0, pi_phase]) * self.rf_amp

    def tx_gate_wf(tstart, echo_idx):
        tx_gate_pre = 2 # us, time to start the TX gate before each RF pulse begins
        tx_gate_post = 1 # us, time to keep the TX gate on after an RF pulse ends

        if echo_idx == 0:
            # do pi/2 pulse, then start first pi pulse
            return np.array([tstart + (self.echo_duration - self.rf_pi2_duration)/2 - tx_gate_pre,
                             tstart + (self.echo_duration + self.rf_pi2_duration)/2 + tx_gate_post,
                             tstart + self.echo_duration - rf_pi_duration/2 - tx_gate_pre]), \
                             np.array([1, 0, 1])
        elif echo_idx == self.echos_per_tr:
            # finish final RF pulse
            return np.array([tstart + rf_pi_duration/2 + tx_gate_post]), np.array([0])
        else:
            # finish last pi pulse, start next pi pulse
            return np.array([tstart + rf_pi_duration/2 + tx_gate_post, tstart + self.echo_duration - rf_pi_duration/2 - tx_gate_pre]), \
                np.array([0, 1])

    def readout_grad_wf(tstart, echo_idx):
        if echo_idx == 0:
            return trap_cent(tstart + self.echo_duration*3/4, self.readout_amp, self.readout_grad_duration/2,
                             self.trap_ramp_duration, trap_ramp_pts)
        else:
            return trap_cent(tstart + self.echo_duration/2, self.readout_amp, self.readout_grad_duration,
                             self.trap_ramp_duration, trap_ramp_pts)

    def readout_wf(tstart, echo_idx):
        if echo_idx != 0:
            return np.array([tstart + (self.echo_duration - self.readout_duration)/2, tstart + (self.echo_duration + self.readout_duration)/2 ]), np.array([1, 0])
        else:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise

    def phase_grad_wf(tstart, echo_idx):
        t1, a1 = trap_cent(tstart + (self.echo_duration - self.phase_grad_interval)/2, phase_amps[echo_idx-1], self.phase_grad_duration,
                           self.trap_ramp_duration, trap_ramp_pts)
        t2, a2 = trap_cent(tstart + (self.echo_duration + self.phase_grad_interval)/2, -phase_amps[echo_idx-1], self.phase_grad_duration,
                           self.trap_ramp_duration, trap_ramp_pts)
        if echo_idx == 0:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
        elif echo_idx == self.echos_per_tr: # last echo, don't need 2nd trapezoids
            return t1, a1
        else: # otherwise do both trapezoids
            return np.hstack([t1, t2]), np.hstack([a1, a2])

    def slice_grad_wf(tstart, echo_idx, tr_idx):
        t1, a1 = trap_cent(tstart + (self.echo_duration - self.phase_grad_interval)/2, slice_amps[tr_idx], self.phase_grad_duration,
                           self.trap_ramp_duration, trap_ramp_pts)
        t2, a2 = trap_cent(tstart + (self.echo_duration + self.phase_grad_interval)/2, -slice_amps[tr_idx], self.phase_grad_duration,
                           self.trap_ramp_duration, trap_ramp_pts)
        if echo_idx == 0:
            return np.array([tstart]), np.array([0]) # keep on zero otherwise
        elif echo_idx == self.echos_per_tr: # last echo, don't need 2nd trapezoids
            return t1, a1
        else: # otherwise do both trapezoids
            return np.hstack([t1, t2]), np.hstack([a1, a2])

    #tr_total_time = self.echo_duration * (self.echos_per_tr + 1) + self.tr_pause_duration

    expt = ex.Experiment(lo_freq=self.lo_freq, rx_t=self.rx_period, init_gpa=self.init_gpa)

    global_t = 20 # start the first TR at 20us

    for tr in range(self.trs):
        for echo in range(self.echos_per_tr + 1):
            tx_t, tx_a = rf_wf(global_t, echo)
            tx_gate_t, tx_gate_a = tx_gate_wf(global_t, echo)
            readout_t, readout_a = readout_wf(global_t, echo)
            rx_gate_t, rx_gate_a = readout_wf(global_t, echo)

            readout_grad_t, readout_grad_a = readout_grad_wf(global_t, echo)
            phase_grad_t, phase_grad_a = phase_grad_wf(global_t, echo)
            slice_grad_t, slice_grad_a = slice_grad_wf(global_t, echo, tr)

            global_t += self.echo_duration

            if grad_board == "gpa-fhdo":
                gpa_fhdo_offset = (1 / 0.2 / 3.1) # microseconds; offset between channels to avoid parallel updates (default update rate is 0.2 Msps, so 1/0.2 = 5us, 5 / 3.1 gives the offset between channels; extra 0.1 for a safety margin)
                phase_grad_t = phase_grad_t + gpa_fhdo_offset # can't use += because of casting rules
                slice_grad_t = slice_grad_t + 2*gpa_fhdo_offset

            expt.add_flodict({
                'tx0': (tx_t, tx_a),
                'tx1': (tx_t, tx_a),
                'grad_vx': (readout_grad_t, readout_grad_a),
                'grad_vy': (phase_grad_t, phase_grad_a),
                'grad_vz': (slice_grad_t, slice_grad_a),
                'rx0_en': (readout_t, readout_a),
                'rx1_en': (readout_t, readout_a),
                'tx_gate': (tx_gate_t, tx_gate_a),
                'rx_gate': (rx_gate_t, rx_gate_a),
            })

        global_t += self.tr_pause_duration
        
    expt.__del__()

        # RF
    idict = expt._seq
    tx0_i_t, tx0_i_a = idict['tx0_i']
    tx0_q_t, tx0_q_a = idict['tx0_q']
    tx0_t = tx0_i_t / fpga_clk_freq_MHz
    tx0_y = (tx0_i_a + 1j * tx0_q_a)/32767

    # Grad x
    grad_x_t,  grad_x_a = idict['ocra1_vx']
    grad_x_t_float = grad_x_t / fpga_clk_freq_MHz
    grad_x_a_float = (grad_x_a - 32768) / 32768
    
    # Grad y
    grad_y_t,  grad_y_a = idict['ocra1_vy']
    grad_y_t_float = grad_y_t / fpga_clk_freq_MHz
    grad_y_a_float = (grad_y_a - 32768) / 32768
    
    # Grad z
    grad_z_t,  grad_z_a = idict['ocra1_vz']
    grad_z_t_float = grad_z_t / fpga_clk_freq_MHz
    grad_z_a_float = (grad_z_a - 32768) / 32768
    

    
    return tx0_t, tx0_y,  grad_x_t_float, grad_x_a_float, grad_y_t_float, grad_y_a_float, grad_z_t_float, grad_z_a_float


