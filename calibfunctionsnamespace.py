"""
Operations Namespace

@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   1.0
@change:    06/28/2020

@summary:   Namespace for operations

"""


class Namespace:
    cfn = "Calibration Function"
    systemproperties = "System Properties"
    RFproperties = "RF Properties"
    gradientshims = "Gradients shim values"
    sqncproperties = "Calibration function's properties"
    lo_freq = "Frequency (MHz)"
    rf_amp = "RF Amplitude"
    rf_tstart = "RF start time"
    rf_duration = "RF duration (us)"
    rx_tstart = "RX start time"
    echo_duration = "TE (ms)"
    fcn = "calibfunction"
    shim = "Shims (x,y,z)"
    rf_wait = "Tx - rx time"
    BW= "BandWidth (KHz)"
    tr_duration="TR (ms)"
    axes = "Axes (rd=1,ph=2,sl=3)"
    N = "Number of TRs"
    step = "Step"
    rf_pi2_duration="RF excitation duration (us)"
    readout_duration = "Readout duration (ms)"
    rx_wait="Waiting time before readout (us)"
    shim_initial = "Initial value for shimming"
    shim_final = "Final value for shimming"

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
