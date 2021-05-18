"""
Rabi map

@author:    Yolanda Vives

@summary: increase the pulse width and plot the peak value of the signal received 
@status: under development
@todo:

"""

import matplotlib as plt
from manager.datamanager import DataManager
from seq_standalone.fid_standalone import fid
from seq_standalone.spinEcho_standalone import spin_echo

def rabi_flops(fid_SE=1,  # 0=FID, 1=SE
                lo_freq=0.5, 
                pulse_duration=50):  # Initial pulse duration

    if fid_SE==0:
        rxd, msgs=fid(lo_freq)  # use fid or spin_echo
    else:
        rxd, msgs=spin_echo(lo_freq)            

    dataobject:DataManager=DataManager(rxd, lo_freq, len(rxd))
    f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency = dataobject.get_peakparameters()
        
    return(f_signalValue)


if __name__ == "__main__":
    
    peakVals = [] #define the array 
    for k in range(20):
        print(k)
        peak_value=rabi_flops(fid_SE=1, lo_freq=3, pulse_duration=50+k)
        list.append(peakVals)
    
    plt.plot(peakVals)
    plt.show()
