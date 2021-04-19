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
    rf_amp = "Amplifier"
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
    shim = "Gradient Shim Values"
    x_grad = "X Gradient"
    y_grad = "Y Gradient"
    z_grad = "Z Gradient"
    z2_grad = "Z² Gradient"
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
    
   

class Reconstruction:
    spectrum = "1D FFT"
    kspace = "2D FFT"
