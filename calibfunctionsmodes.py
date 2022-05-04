"""
Calibfunctions Modes

@author:    
@contact:   
@version:   2.0 (Beta)
@change:    

@summary:   TBD

@status:    
@todo:      
"""

from calibfunctionsnamespace import Namespace as cfnmspc

class LarmorFreq:
    
    def __init__(self, 
                 cfn:str, 
                 lo_freq: float=None, 
                 rf_amp: float=None, 
                 nScans:int=None, 
                 N: float=None, 
                 step_larmor:int=None, 
                 rx_wait:int=None, 
                 rf_pi2_duration: int=None, 
                 BW:float=None, 
                 echo_duration:int=None, 
                 readout_duration:int=None, 
                 resolution_larmor:float=None
                 ):
                     
             
        self.cfn: str=cfn
        self.lo_freq: float=lo_freq
        self.rf_amp: float=rf_amp
        self.nScans:int=nScans
        self.N: int=N
        self.step_larmor: int=step_larmor
        self.rf_pi2_duration: int=rf_pi2_duration
        self.echo_duration:int=echo_duration
        self.BW: float=BW
        self.readout_duration:int=readout_duration
        self.rx_wait:int=rx_wait
        self.resolution_larmor:float=resolution_larmor
    
    @property
    def sqncproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            cfnmspc.nScans:[int(self.nScans)], 
            cfnmspc.N:[int(self.N)], 
            cfnmspc.step_larmor:[int(self.step_larmor)], 
            cfnmspc.lo_freq: [float(self.lo_freq)],
            cfnmspc.BW:[int(self.BW)], 
            cfnmspc.echo_duration:[int(self.echo_duration)], 
            cfnmspc.readout_duration:[int(self.readout_duration)], 
            cfnmspc.rx_wait:[int(self.rx_wait)], 
            cfnmspc.resolution_larmor:[float(self.resolution_larmor)]
        }

    @property
    def RFproperties(self) -> dict:
        return{
            cfnmspc.rf_amp: [float(self.rf_amp)],
            cfnmspc.rf_pi2_duration:[int(self.rf_pi2_duration)], 
        }    
 

class FlipAngle:
    
    def __init__(self,
                 cfn: str,
                 lo_freq: float=None,
                 rf_amp: float=None,
                 N: int=None, 
                 nScans: int=None, 
                 step:float=None, 
                 rf_pi2_duration:int=None, 
                 tr_duration: int=None, 
                 echo_duration:int=None, 
                 BW: float=None, 
                 readout_duration:int=None, 
                 rx_wait:int=None
                 ):
                   
        self.cfn: str=cfn
        self.lo_freq: float=lo_freq
        self.rf_amp: float=rf_amp
        self.N: int=N
        self.nScans: int=nScans
        self.step: float=step
        self.rf_pi2_duration: int=rf_pi2_duration
        self.tr_duration: int=tr_duration
        self.echo_duration:int=echo_duration
        self.BW: float=BW
        self.readout_duration:int=readout_duration
        self.rx_wait:int=rx_wait

    @property
    def sqncproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            cfnmspc.lo_freq: [float(self.lo_freq)],
            cfnmspc.BW:[int(self.BW)], 
            cfnmspc.tr_duration:[int(self.tr_duration)],             
            cfnmspc.echo_duration:[int(self.echo_duration)], 
            cfnmspc.readout_duration:[int(self.readout_duration)], 
            cfnmspc.rx_wait:[int(self.rx_wait)], 
        }

    @property
    def RFproperties(self) -> dict:
        return{
            cfnmspc.rf_amp: [float(self.rf_amp)],
            cfnmspc.N:[int(self.N)], 
            cfnmspc.nScans:[int(self.nScans)], 
            cfnmspc.step:[float(self.step)], 
            cfnmspc.rf_pi2_duration:[int(self.rf_pi2_duration)], 
        }    
        
class RabiFlops:
    """

    """
    def __init__(self,
                 cfn: str,
                 lo_freq: float=None,
                 rf_amp: float=None,
                 N: int=None, 
                 nScans:int=None, 
                 rf_pi2_duration0:int=None, 
                 rf_pi2_durationEnd:int=None, 
                 tr_duration: int=None, 
                 echo_duration:int=None, 
                 BW: float=None, 
                 readout_duration:int=None, 
                 rx_wait:int=None
                 ):
                   
        self.cfn: str=cfn
        self.lo_freq: float=lo_freq
        self.rf_amp: float=rf_amp
        self.N: int=N
        self.nScans:int=nScans
        self.rf_pi2_duration0: int=rf_pi2_duration0
        self.rf_pi2_durationEnd: int=rf_pi2_durationEnd
        self.tr_duration: int=tr_duration
        self.echo_duration:int=echo_duration
        self.BW: float=BW
        self.readout_duration:int=readout_duration
        self.rx_wait:int=rx_wait

    @property
    def RFproperties(self) -> dict:
        return{
            cfnmspc.rf_amp: [float(self.rf_amp)],
            cfnmspc.rf_pi2_duration0:[int(self.rf_pi2_duration0)], 
            cfnmspc.rf_pi2_durationEnd:[int(self.rf_pi2_durationEnd)], 
            cfnmspc.N:[int(self.N)], 
            cfnmspc.nScans:[int(self.nScans)], 
            cfnmspc.lo_freq: [float(self.lo_freq)],
            cfnmspc.BW:[int(self.BW)], 
            cfnmspc.tr_duration:[int(self.tr_duration)],             
            cfnmspc.echo_duration:[int(self.echo_duration)], 
            cfnmspc.readout_duration:[int(self.readout_duration)], 
            cfnmspc.rx_wait:[int(self.rx_wait)], 
        }    

#    @property
#    def gradientshims(self) -> dict:
#        return{
#            cfnmspc.shim:[list(self.shim)]
#        }  

class InvRecov:
    """

    """
    def __init__(self,
                 cfn: str,
                 lo_freq: float=None,
                 rf_amp: float=None,
                 N: int=None, 
                 step:int=None, 
                 rf_duration:int=None, 
                 tr_duration: int=None, 
                 echo_duration:int=None, 
                 BW: float=None, 
                 readout_duration:int=None, 
                 rx_wait:int=None
                 ):
                   
        self.cfn: str=cfn
        self.lo_freq: float=lo_freq
        self.rf_amp: float=rf_amp
        self.N: int=N
        self.step: int=step
        self.rf_duration: int=rf_duration
        self.tr_duration: int=tr_duration
        self.echo_duration:int=echo_duration
        self.BW: float=BW
        self.readout_duration:int=readout_duration
        self.rx_wait:int=rx_wait

    @property
    def sqncproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            cfnmspc.N:[int(self.N)], 
            cfnmspc.step:[int(self.step)], 
            cfnmspc.lo_freq: [float(self.lo_freq)],
            cfnmspc.BW:[int(self.BW)], 
            cfnmspc.tr_duration:[int(self.tr_duration)],             
            cfnmspc.echo_duration:[int(self.echo_duration)], 
            cfnmspc.readout_duration:[int(self.readout_duration)], 
            cfnmspc.rx_wait:[int(self.rx_wait)], 
        }

    @property
    def RFproperties(self) -> dict:
        return{
            cfnmspc.rf_amp: [float(self.rf_amp)],
            cfnmspc.rf_duration:[int(self.rf_duration)], 
        }    

#    @property
#    def gradientshims(self) -> dict:
#        return{
#            cfnmspc.shim:[list(self.shim)]
#        }  
           

class GradShim:
    """
    Spectrum Operation Class
    """
    def __init__(self,
         cfn: str,
         lo_freq: float=None,
         BW: float=None,
         tr_duration: int=None,          
         rf_amp: float=None,
         rf_duration:int=None, 
         rf_tstart:int=None, 
         N: int=None, 
         shim_initial:float=None, 
         shim_final:float=None, 
         readout_duration:int=None, 
         rx_wait:int=None
         ):

                     
        """
        Initialization of gradient echo sequence class
        @param frequency:       Frequency value for operation
        @param amplification:     Attenuation value for operation
        @param shim:            Shim values for operation
        @return:                None
        """
        self.cfn: str=cfn
        self.lo_freq: float=lo_freq
        self.BW: float=BW
        self.tr_duration: int=tr_duration     
        self.rf_amp: float=rf_amp
        self.rf_duration:int=rf_duration
        self.rf_tstart:int=rf_tstart
        self.N: int=N
        self.shim_initial:float=shim_initial
        self.shim_final:float=shim_final
        self.readout_duration:int=readout_duration
        self.rx_wait:int=rx_wait
        

    @property
    def systemproperties(self) -> dict:
        return{
            cfnmspc.lo_freq: [float(self.lo_freq)],
            cfnmspc.BW: [int(self.BW)],
            cfnmspc.tr_duration:[int(self.tr_duration)], 
            cfnmspc.N:[int(self.N)], 
            cfnmspc.shim_initial:[float(self.shim_initial)], 
            cfnmspc.shim_final:[float(self.shim_final)], 
        }
    
    @property    
    def RFproperties(self) -> dict:
        return{
            cfnmspc.rf_amp: [float(self.rf_amp)],
            cfnmspc.rf_duration:[int(self.rf_duration)], 
            cfnmspc.rf_tstart:[int(self.rf_tstart)], 
        }    

    @property
    def sqncproperties(self) -> dict:
        return{
            cfnmspc.readout_duration:[int(self.readout_duration)], 
            cfnmspc.rx_wait:[int(self.rx_wait)], 
        }
           

"""
Definition of default calibfunctions
"""
defaultcalibfunctions={

    #RabiFlops(lo_freq,rf_amp,N,nScans,rf_pi2_duration0,rf_pi2_durationEnd,tr_duration,echo_duration,BW,readout_duration,rx_wait)
    'Rabi Flops': RabiFlops('Rabi Flops', 3.03, 0.6, 80, 3, 30, 200,  500, 100, 80, 3, 50), 
    #InvRecov(lo_freq,rf_amp,N,step,rf_pi2_duration,tr_duration,echo_duration,BW,readout_duration,rx_wait)
    'Inversion Recovery': InvRecov('Inversion Recovery', 3.041, 0.3, 20, 5, 50, 20, 15, 31, 5, 100), 
    #larmor(lo_freq,rf_amp,nScans,N,step,rf_pi2_duration,echo_duration,BW,readout_duration,rx_wait)
    'Larmor Frequency': LarmorFreq('Larmor Frequency', 3.03, 0.6, 3, 20, 2,  100, 190, 80, 100, 3, 1), 
    #Shimming(lo_freq,BW,tr_duration,rf_amp,rf_duration,rf_tstart,N_shim,shim_initial,shim_final,readout_duration,rx_wait)
    'Shimming': GradShim('Shimming', 3.041, 31, 500, 0.2, 50, 100, 10, -0.01, 0.01, 50, 100), 
    #FlipAngle(lo_freq,rf_amp,N,nScans,step,rf_pi2_duration,tr_duration,echo_duration,BW,readout_duration,rx_wait)
    'FlipAngle': FlipAngle('Flip Angle', 3.03, 0.3, 10,3,  0.05, 190, 500, 100, 80, 3, 100), 
}
