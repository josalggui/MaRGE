"""
Frequency fit

@author:    Yolanda Vives

@summary: look for larmor frequency
@status: under development
@todo:

"""
import numpy as np
from configs.hw_config import fo
from seq_standalone.fid_standalone import fid
from seq_standalone.spinEcho_standalone import spin_echo
from manager.datamanager import DataManager

def larmor(self, fid_SE=1,  # 0=FID, 1=SE
                lo_freq=0.5, 
                step=0.1,  # MHz
                bw2=0.2):  # BW/2 of search (MHz)

    freqPeak=lo_freq
    peak_f = 0
#    peakIdx=0

    n = (2*bw2/step)+1
    f = np.linspace(freqPeak-bw2, freqPeak+bw2, n) 
    for ff in f:
        
        if fid_SE==0:
            self.rxd, self.msgs=fid(lo_freq=ff)  # use fid or spin_echo
        else:
            self.rxd, self.msgs=spin_echo(lo_freq=ff)
        
        dataobject:DataManager=DataManager(self.rxd, ff, len(self.rxd))
        f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency = dataobject.get_peakparameters(self)
        
        if (f_signalValue > peak_f):
            peak_f=f_signalValue
#                peak_t=t_signalValue
#                peakIdx=f_signalIdx
            freqPeak=f_signalFrequency
        

    return(freqPeak, peak_f)

if __name__ == "__main__":
    
    f_larmor, v_larmor=larmor(fid_SE=1, lo_freq=fo, step=0.1, bw2=0.2)
