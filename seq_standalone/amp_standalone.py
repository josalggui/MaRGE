"""
Frequency fit

@author:    Yolanda Vives

@summary: look for larmor frequency
@status: under development
@todo:

"""
import numpy as np
#from configs.hw_config import fo
from seq.fid import fid
from manager.datamanager import DataManager

def amplitude(lo_freq=0.5, 
                rf_amp=1, 
                step=0.01,  #
                bw2=0.2):

    freqPeak=lo_freq
    peak_f = 0
#    peakIdx=0
    
    while(1):
        n = (2*bw2/step)+1
        f = np.linspace(freqPeak-bw2, freqPeak+bw2, n) 
        for ff in f:
            
            self.rxd, self.msgs=fid(lo_freq=ff)
            
            dataobject:DataManager=DataManager(self.rxd, ff, len(self.rxd))
            f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency = dataobject.get_peakparameters(self)
            
            if (f_signalValue > peak_f):
                peak_f=f_signalValue
#                peak_t=t_signalValue
#                peakIdx=f_signalIdx
                freqPeak=f_signalFrequency
        
        if step > 1e-6:
            bw2=bw2/2
            step=step/2
        else:
            break

    return(freqPeak, peak_f)

if __name__ == "__main__":
    
#        for k in range(20):
#            print(k)
    lo_freq=3.041
    amplitude(lo_freq=lo_freq, step=0.1, bw2=0.2)
