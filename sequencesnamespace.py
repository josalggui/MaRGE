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
    RFproperties = "RF and System Properties"
    Gproperties = "Gradients Properties"
    lo_freq = "Frequency (MHz)"
    rf_amp = "RF Amplitude"
    G = "Gradients amplification"
    trs = "Number of TRs"
    grad_tstart = "Gradients start time"
    tr_total_time = "TR"
    rf_tstart = "RF start time (us)"
    rf_tend = "RF end time (us)"
    rx_tstart = "RX start time"
    rx_tend = "RX end time"
    rx_period = "RX period"
    rx_wait="Waiting time before readout (ms)"
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
    trap_ramp_duration = "Gradients ramp duration (us)"
    phase_delay = "Phase Delay"
    echos_per_tr = "Echos per TR"
    echo_duration = "TE (ms)"
    phase_duration = "Phase Duration (us)"
    readout_duration = "Readout Duration (ms)"
    readout_grad_duration = "Readout Gradient Duration"
    phase_grad_interval = "Phase interval"
    phase_grad_duration = "Phase and Slice Duration (us)"
    tr_pause_duration = "TR pause Duration"
    plot_rx = "plot rx"
    init_gpa = "GPA initialisation"
    slice_amp = "Slice amplitude"
    rf_pi2_duration = "RF excitation duration (us)"
    rf_pi_duration = "RF refocusing duration (us)"
    R = "Radial"
    GE = "Gradient Echo"
    TSE = "Turbo Spin Echo"
    SE1D = "Spin Echo 1D"
    gradientshims = "Gradients shim values"
    shim = "Shims (x,y,z)"
    BW= "BandWidth (KHz)"
    n="Number of points (Nx,Ny,Nz)"
    fov="FOV [rd,ph,sl] (mm)"# TSE(x,y,z); RARE(rd,ph,sl)"
    tr_duration="TR (ms)"
    nScans="nScans"
    rawdata ="rawdata"
    average = "average"
    fft = "fft"
    preemph_factor = "Pre emphasis factor"
    sweep_mode = "Sweep Mode"
    par_acq_factor = "Partial acquisition factor"
    axes = "Axes (rd=0,ph=1,sl=2)"
    oversampling_factor = "oversampling factor"
    larmorFreq="Larmor Frequency (MHz)"
    rfExAmp="RF Excitation Amplitude (a.u.)"
    rfReAmp="RF Refocusing Amplitude (a.u.)"
    rfExTime="RF Excitation Time (us)"
    rfReTime="RF Refocusing Time (us)"
    echoSpacing="Echo Spacing (ms)"
    nPoints="Number of Points"
    etl="ETL"
    acqTime="Acquisition Time (ms)"
    CPMG="CPMG"
    RARE= "RARE"
    repetitionTime = "Repetition Time (ms)"
    inversionTime="Inversion Time (ms)" 
    dfov="Displacement of fOV (mm)"
    axesEnable="Axes Enable (on=1, off=0)"
    sweepMode= "Sweep Mode (0=T2w, 1=T1w, 2=Rhow)" 
    phaseGradTime="Phase Grad. Time (us)"
    rdPreemphasis="Rd Preemphasis Factor (a.u.)"
    drfPhase = "Phase of 90º pulse (º)" 
    dummyPulses = "Number of Dummy Pulses" 
    axis = "Axes (rd=0,ph=1,sl=2)"
    n_rd = 'N rd'
    n_ph = 'N ph'
    n_sl = 'N sl'
    ns = 'ns'
    x = 'x'
    y = 'y'
    z = 'z'
    fov_rd = 'fov rd'
    fov_ph = 'fov ph'
    fov_sl = 'fov sl'
    shimming = 'Shimming (x, y, z)'
    parAcqLines = 'Partial adquisition (Off=0; 1,2...=nº lines)'
    age = "Age"
    sex = "Sex (M/F)"
    weight="Weight (gr)"
    demographics = "Demographical info"
    name_code = "ID"
    descrip = 'Description'
    
class Tooltip_label:
    rx_period = "Sampling time"
    rf_amp = "Full scale = 1"
    
class Tooltip_inValue:
    rf_amp = "Value between 0 and 1"
    dbg_sc = "Value between 0 and 1"
    slice_amp = "Value between 0 and 1"
    phase_amp = "Value between 0 and 1"
#    readout_amp = "Value between 0 and 1"
    G = "Value between 0 and 1"
    phase_start_amp = "Value between 0 and 1"
    slice_start_amp = "Value between 0 and 1"

class Reconstruction:
    spectrum = "1D FFT"
    kspace = "2D FFT"
