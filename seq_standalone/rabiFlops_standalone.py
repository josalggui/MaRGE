"""
Rabi map

@author:    Yolanda Vives

@summary: increase the pulse width and plot the peak value of the signal received 
@status: under development
@todo:

"""

import matplotlib as plt
from manager.datamanager import DataManager
from spinEcho_standalone import spin_echo
import time

def rabi_flops(lo_freq, pulse_duration):  # Initial pulse duration
        
    rxd, msgs=spin_echo(lo_freq, pulse_duration)            

    dataobject:DataManager=DataManager(rxd, lo_freq, len(rxd))
    f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency = dataobject.get_peakparameters()
            
    return f_signalValue


if __name__ == "__main__":

    peakVals = [] #define the array 
    for k in range(2):
        print(k)
        peak_value=rabi_flops(lo_freq=3.069, pulse_duration=50+k)
        print(peak_value)
        peakVals.append(peak_value)
        time.sleep(1)
    
    plt.plot(peakVals)
    plt.show()
