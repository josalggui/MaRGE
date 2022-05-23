"""
Operations Namespace

@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   1.0
@change:    06/28/2020

@summary:   Namespace for operations

"""


class Namespace:
    seq = "sequence"
    dbg_sc = "Debug"
#    systemproperties = "System Properties"
    systemproperties = ""
    RFproperties = "RF and System Properties"
    Gproperties = "Gradients Properties"
    age = "Age"
    sex = "Sex (M/F)"
    weight="Weight (gr)"
    demographics = "Demographical info"
    name_code = "ID"
    descrip = "Description"
    n_rd = 'N rd'
    n_ph = 'N ph'
    n_sl = 'N sl'
    ns = 'ns'
    
    RARE= "RARE"
    nScans = "nScans"
    larmorFreq="Larmor Frequency (MHz)"
    rfExAmp="RF Excitation Amplitude (a.u.)"
    rfReAmp="RF Refocusing Amplitude (a.u.)"
    rfExTime="RF Excitation Time (us)"
    rfReTime="RF Refocusing Time (us)"
    echoSpacing="Echo Spacing (ms)"
    preExTime="Time from preexcitation pulse to inversion pulse (us)"
    inversionTime="Inversion Time (ms)" 
    repetitionTime = "Repetition Time (ms)"
    fov="FOV [rd,ph,sl] (mm)"
    dfov="Displacement of fOV (mm)"
    nPoints="Number of Points"
    etl="ETL"
    acqTime="Acquisition Time (ms)"
    axes = "Axes (rd=0,ph=1,sl=2)"
    axesEnable="Axes Enable (on=1, off=0)"
    sweepMode= "Sweep Mode (0=T2w, 1=T1w, 2=Rhow)" 
    rdGradTime="Readout grad. Time (us)"
    rdDephTime="Readout dephasing grad. Time (ms)"
    phGradTime="Phase and Slice grad. Time (ms)"
    rdPreemphasis="Rd Preemphasis Factor (a.u.)"
    drfPhase = "Phase of 90º pulse (º)" 
    dummyPulses = "Number of Dummy Pulses" 
    shimming = 'Shimming (x, y, z)'
    parFourierFraction= 'Fraction of acquired k-space along phase direction'





    
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
