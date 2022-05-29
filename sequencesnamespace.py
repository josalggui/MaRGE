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
    HASTE = "HASTE"
    GRE3D = "GRE3D"
    
    acqTime="Acquisition Time (ms)"
    axes = "Axes (rd=0,ph=1,sl=2)"
    axesEnable="Axes Enable (on=1, off=0)"
    crusherDelay='Crusher Delay (ms)'
    drfPhase = "Phase of 90º pulse (º)" 
    dummyPulses = "Number of Dummy Pulses" 
    dfov="Displacement of fOV (mm)"
    dephGradTime="Dephasing Gradient Time (ms)"
    echoSpacing="Echo Spacing (ms)"
    echoTime = "Echo Time (ms)"
    etl="ETL"
    fov="FOV [rd,ph,sl] (mm)"
    inversionTime="Inversion Time (ms)" 
    larmorFreq="Larmor Frequency (MHz)"
    nScans = "nScans"
    nPoints="Number of Points"
    preExTime="Time from preexcitation pulse to inversion pulse (us)"
    parFourierFractionSl= 'Fraction of acquired k-space (Sl)'
    parFourierFractionPh='Fraction of acquired k-space (Ph)'
    phGradTime="Phase and Slice grad. Time (ms)"
    repetitionTime = "Repetition Time (ms)"
    rdGradTime="Readout grad. Time (us)"
    rdDephTime="Readout dephasing grad. Time (ms)"
    rdPreemphasis="Rd Preemphasis Factor"
    rfExAmp="RF Excitation Amplitude (a.u.)"
    rfReAmp="RF Refocusing Amplitude (a.u.)"
    rfExTime="RF Excitation Time (us)"
    rfReTime="RF Refocusing Time (us)"
    rfEnvelope='Envelope of RF pulse (Rec or Sinc)'
    shimming = 'Shimming (x, y, z)'
    ssPreemphasis='Sl Preemphasis Factor'
    sweepMode= "Sweep Mode (0=T2w, 1=T1w, 2=Rhow)" 
    spoiler = "Spoiler (0 or 1)"
    




    
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
