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

#class LarmorFreq:
#    
#    def __init__(self, 
#                 seq:str, 
#                 dbg_sc: float=None, 
#                 lo_freq: float=None, 
#                 rf_amp: float=None, 
#                 trs: int=None, 
#                 rf_pi2_duration: int=None, 
#                 echo_duration:int=None, 
#                 readout_duration:int=None, 
#                 rx_period:float=None, 
#                 ):
#    
#        self.seq:str=seq
#        self.dbg_sc:float=dbg_sc
#        self.lo_freq:float=lo_freq
#        self.rf_amp: float=rf_amp
#        self.trs:int=trs
#        self.rf_pi2_duration:int=rf_pi2_duration
#        self.echo_duration:int=echo_duration
#        self.readout_duration:int=readout_duration
#        self.rx_period:float=rx_period
#    
#    @property
#    def systemproperties(self) -> dict:
#        # TODO: add server cmd's as third entry in list
#        return {
#            nmspc.lo_freq: [float(self.lo_freq)],
#            nmspc.rf_amp: [float(self.rf_amp)],
#            nmspc.dbg_sc:[float(self.dbg_sc)], 
#            nmspc.trs:[int(self. trs)]
#        }
#
#    @property
#    def sqncproperties(self) -> dict:
#        return{
#            nmspc.rf_pi2_duration:[int(self.rf_pi2_duration)],            
#            nmspc.echo_duration:[int(self.echo_duration)], 
#            nmspc.readout_duration:[int(self.readout_duration)], 
#            nmspc.rx_period:[float(self.rx_period)]
#        }    

#class Amplitude:
#    
#    def __init__(self, 
#                 seq:str, 
#                 dbg_sc: float=None, 
#                 lo_freq: float=None, 
#                 rf_amp: float=None, 
#                 rf_duration: int=None, 
#                 rf_tstart:int=None, 
#                 rf_wait:int=None, 
#                 rx_period:float=None, 
#                 readout_duration:int=None
#                 ):
#        
#        self.seq: str=seq
#        self.dbg_sc: float= dbg_sc
#        self.lo_freq: float=lo_freq
#        self.rf_amp: float=rf_amp
#        self.rf_duration: int=rf_duration
#        self.rf_tstart:int=rf_tstart
#        self.rf_wait:int=rf_wait
#        self.rx_period:float=rx_period
#        self.readout_duration:int=readout_duration
#
#    @property
#    def systemproperties(self) -> dict:
#        # TODO: add server cmd's as third entry in list
#        return {
#            nmspc.lo_freq: [float(self.lo_freq)],
#            nmspc.rf_amp: [float(self.rf_amp)],
#            nmspc.dbg_sc:[float(self.dbg_sc)]
#        }
#
#    @property
#    def sqncproperties(self) -> dict:
#        return{
#            nmspc.rf_duration:[int(self.rf_duration)],            
#            nmspc.rf_tstart:[int(self.rf_tstart)], 
#            nmspc.rf_wait:[int(self.rf_wait)], 
#            nmspc.rx_period:[float(self.rx_period)], 
#            nmspc.readout_duration:[int(self.readout_duration)]
#        }    

    
        
class RabiFlops:
    """

    """
    def __init__(self,
                 cfn: str,
                 lo_freq: float=None,
                 rf_amp: float=None,
                 N: int=None, 
                 step:int=None, 
                 rf_pi2_duration:int=None, 
                 tr_duration: int=None, 
                 echo_duration:int=None, 
                 BW: int=None, 
                 readout_duration:int=None, 
                 shim: list=None
                 ):
                   
        self.cfn: str=cfn
        self.lo_freq: float=lo_freq
        self.rf_amp: float=rf_amp
        self.N: int=N
        self.step: int=step
        self.rf_pi2_duration: int=rf_pi2_duration
        self.tr_duration: int=tr_duration
        self.echo_duration:int=echo_duration
        self.BW: int=BW
        self.readout_duration:int=readout_duration
        self.shim:list=shim

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
        }

    @property
    def RFproperties(self) -> dict:
        return{
            cfnmspc.rf_amp: [float(self.rf_amp)],
            cfnmspc.rf_pi2_duration:[int(self.rf_pi2_duration)], 
        }    

    @property
    def gradientshims(self) -> dict:
        return{
            cfnmspc.shim:[list(self.shim)]
        }  
           

#class GradShim:
#    """
#    Spectrum Operation Class
#    """
#    def __init__(self,
#                 seq: str,
#                 dbg_sc: float=None, 
#                 lo_freq: float=None,
#                rf_amp: float=None,
#                 trs: int=None,
#                 rx_period: int=None, 
#                 rf_tstart: int=None, 
#                 slice_amp: float=None, 
#                 phase_amp: float=None, 
#                 readout_amp: float=None, 
#                 rf_duration: int=None, 
#                 trap_ramp_duration: int=None, 
#                 phase_delay: int=None, 
#                 phase_duration: int=None, 
#                 shim: list=None
#                 ):
#
#                     
#        """
#        Initialization of gradient echo sequence class
#        @param frequency:       Frequency value for operation
#        @param amplification:     Attenuation value for operation
#        @param shim:            Shim values for operation
#        @return:                None
#        """
#        self.seq: str=seq
#        self.dbg_sc: float=dbg_sc
#        self.lo_freq: float=lo_freq
#        self.rf_amp: float=rf_amp
#        self.trs:int=trs
#        self.rx_period: float=rx_period
#        self.rf_tstart: int=rf_tstart
#        self.slice_amp: float=slice_amp
#        self.phase_amp: float=phase_amp
#        self.readout_amp: float=readout_amp
#        self.rf_duration: int=rf_duration
#        self.trap_ramp_duration: int=trap_ramp_duration
#        self.phase_delay: int=phase_delay
#        self.phase_duration: int=phase_duration
#        self.shim:list=shim
#        
#
#    @property
#    def systemproperties(self) -> dict:
#        # TODO: add server cmd's as third entry in list
#        return {
#            cfnmspc.lo_freq: [float(self.lo_freq)],
#            cfnmspc.rf_amp: [self.rf_amp],
#            cfnmspc.trs:[int(self. trs)], 
#            cfnmspc.dbg_sc:[float(self.dbg_sc)]
#        }
#
#    @property
#    def sqncproperties(self) -> dict:
#        return{
#            cfnmspc.rx_period:[float(self.rx_period)], 
#            cfnmspc.rf_tstart:[int(self.rf_tstart)],         
#            cfnmspc.slice_amp:[float(self.slice_amp)], 
#            cfnmspc.phase_amp:[float(self.phase_amp)], 
#            cfnmspc.readout_amp:[float(self.readout_amp)], 
#            cfnmspc.rf_duration:[int(self.rf_duration)], 
#            cfnmspc.trap_ramp_duration:[int(self.trap_ramp_duration)], 
#            cfnmspc.phase_delay:[int(self.phase_delay)], 
#           cfnmspc.phase_duration:[int(self.phase_duration)]
#        }
#           
#
#    @property
#    def gradientshims(self) -> dict:
#        return{
#            cfnmspc.shim:[list(self.shim)]
#        } 

"""
Definition of default calibfunctions
"""
defaultcalibfunctions={

    #RabiFlops(lo_freq,rf_amp,N,step,rf_pi2_duration,tr_duration,echo_duration,BW,readout_duration,shimming)
    'Rabi Flops': RabiFlops('Rabi Flops', 3.041, 0.3, 20, 5, 100, 1000, 5, 31, 5000,(0, 0, 0))

}
           
