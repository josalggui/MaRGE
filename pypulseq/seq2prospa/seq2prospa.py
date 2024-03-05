from pypulseq.Sequence.sequence import Sequence
from pypulseq.seq2prospa import convert_seq
from pypulseq.seq2prospa import make_se, make_gre


def main(seq: Sequence):
    prospa = convert_seq.main(seq)
    return prospa


if __name__ == '__main__':
    # seq = make_gre.main()
    seq = make_se.main()
    output = main(seq)

    pre = """
    ########################################################
    # 
    # Gradient-echo imaging
    #
    ########################################################
    
    procedure(pulse_program,dir,mode)
    
    
    # Interface description (name, label, x, y, control_type, variable_type)
      interface = ["b1Freq",       "B1 Frequency (MHz)",    "0","0", "tbw", "freq",
                   "repTime",      "Repetition time (ms)",  "0","1", "tbw", "reptime",
                   "rampTime",      "Grad ramp time (us)",  "0","2", "tbw", "float,[150,1e3]",
                   "maxPercent",   "% of k-space to collect",  "1","0", "tbw", "float,[1,100]",
                   "FOV",          "Field of view (mm)",    "1","1", "tb",  "float",
                   "plane",        "Imaging plane",         "1","2", "tm",  "[\\"xy\\",\\"yx\\",\\"xz\\",\\"zx\\",\\"yz\\",\\"zy\\"]",
                   "90Amplitude",  "Pulse amplitude (dB)",  "2","0", "tb","pulseamp",
                   "pulseLength",  "Pulse length (us)",     "2","1", "tb","pulselength",
                   "echoTime",     "Echotime (us)",         "2","2", "tb",  "sdelay"]
    
    
    
    # Relationships to determine remaining variable values
       relationships = ["filterCorr = 6*acqTime*1000/nrPnts + 8.5",
                        "readGrad = 2*pi*nrPnts/(acqTime*1e-3*gamma*FOV*1e-3)",
                        "phaseGrad = readGrad",
                        "bandWidth = nrPnts/(acqTime*1e-3)",
                        "(n1,n2,n3,n4,n5,n6,n8,n9,n10) = geImaging:setImagingPlane(plane,readGrad,phaseGrad,xshim,yshim,zshim,xcal,ycal,zcal)",
                        "n12 = 75",           #Number of steps in the ramp
                        "d12 = rampTime/n12", #Has to be bigger than 2 us
                        "d13 = filterCorr",
                        "d14 = 43",            #Gradient amp delay
                        "d15 = (bandWidth/10000 - 0.5)*acqTime*1000/nrPnts",#Linear eddy current compensation
                        "d1 = pulseLength",
                        "d2 = acqTime*500 - 2*rampTime + d14 + d15",
                        "d3 = echoTime - (acqTime*500 + 5*rampTime + d1/2 + d2) + rxLat",
                        "d4 = d14+d15",
                        "d11 = 250", #delay to settle shim gradients
                        "n7 = nrPnts",
                        "a1 = 90Amplitude",
                        "totPnts = nrPnts",
                        "totTime = acqTime"]
    
    
    # Define the tabs and their order
       tabs = ["Pulse_sequence","Acquisition","Processing_Display_Std","File_Settings"]
    
    # These parameters will be changed between experiments
       variables = ["n4"]
    
    # dx,dy
       dim = [170,26]
    """
    post = """
       lst = endpp() # Return parameter list

    # Phase cycle list
    phaseList = [0,1,2,3; # 90 phase
                0,1,2,3] # Acquire
                
    endproc(lst,tabs,interface,relationships,variables,dim,phaseList)
    """

    output = pre + output + post

    print(output)

"""
n7 = 128
gradRamp = 250 us
acqTime = 128 * 50 us
pulseLength = 100 us
pulseAmplitude = -18
"""
