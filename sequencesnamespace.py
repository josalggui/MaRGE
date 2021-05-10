"""
Operations Namespace

@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   1.0
@change:    06/28/2020

@summary:   Namespace for operations

"""


class Namespace:
    dbg_sc = "Debug"
    systemproperties = "System Properties"
    lo_freq = "Frequency (MHz)"
    rf_amp = "RF Amplitude"
    G = "Gradients amplification"
    trs = "Number of TRs"
    grad_tstart = "Gradients start time"
    tr_total_time = "TR"
    rf_tstart = "RF start time"
    rf_tend = "RF end time"
    rx_tstart = "RX start time"
    rx_tend = "RX end time"
    rx_period = "RX period"
    sampletime = "Sample Time"
    samples = "RX Samples"
    x_shim = "X shim gradient"
    y_shim = "Y shim gradient"
    z_shim = "Z shim gradient"
    z2_shim = "Z² shim gradient"
    seq = "sequence"
    sqncproperties = "Sequence Properties"
    slice_start_amp = "Slice start amplitude"
    phase_amp = "Phase amplitude"
    phase_start_amp = "Phase start amplitude"
    readout_amp = "Readout amplitude"
    trap_ramp_duration = "Gradients ramp duration"
    phase_delay = "Phase Delay"
    echos_per_tr = "Echos per TR"
    echo_duration = "Echos Duration"
    rf_duration = "RF Duration"
    phase_duration = "Phase Duration"
    readout_duration = "Readout Duration"
    readout_grad_duration = "Readout Gradient Duration"
    phase_grad_interval = "Phase interval"
    phase_grad_duration = "Phase Gradient Duration"
    tr_pause_duration = "TR pause Duration"
    plot_rx = "plot rx"
    init_gpa = "GPA initialisation"
    slice_amp = "Slice amplitude"
    rf_pi2_duration = "RF pi2 duration"
    rf_pi_duration = "RF pi duration"
    R = "Radial"
    GE = "Gradient Echo"
    TSE = "Turbo Spin Echo"
    gradientshims = "Gradients shim values"
    shim = "Shims (x,y,z,z²)"
   
class Tooltip_label:
    rx_period = "Sampling time"
    rf_amp = "Full scale = 1"
    
class Tooltip_inValue:
    lo_freq = "Value between 0 and 1"
    rf_amp = "Value between 0 and 1"
    dbg_sc = "Value between 0 and 1"
    slice_amp = "Value between 0 and 1"
    phase_amp = "Value between 0 and 1"
    readout_amp = "Value between 0 and 1"
    G = "Value between 0 and 1"
    phase_start_amp = "Value between 0 and 1"
    slice_start_amp = "Value between 0 and 1"

class Reconstruction:
    spectrum = "1D FFT"
    kspace = "2D FFT"
