"""
Calibration functions Modes

@author:    Yolanda Vives
@contact:   
@version:   2.0 (Beta)
@change:    

@summary:   TBD

@status:    

"""

from sequencesnamespace import Namespace as nmspc

#class SpinEchoSeq:
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
#                 BW:float=None, 
#                 shim: list=None
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
#        self.BW:float=BW
#        self.shim:list=shim
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
#            nmspc.BW:[float(self.BW)]
#        }    
#        
#    @property
#    def gradientshims(self) -> dict:
#        return{
#            nmspc.shim:[list(self.shim)]
#        }
#        
class SpinEcho1DSeq:
    
    def __init__(self, 
                 seq:str, 
                 dbg_sc: float=None, 
                 lo_freq: float=None, 
                 rf_amp: float=None, 
                 trs: int=None, 
                 rf_pi2_duration: int=None, 
                 echo_duration:int=None, 
                 readout_duration:int=None, 
                 BW:float=None, 
                 shim: list=None, 
                 readout_amp:float=None, 
                 readout_grad_duration:int=None, 
                 trap_ramp_duration:int=None
                 ):
    
        self.seq:str=seq
        self.dbg_sc:float=dbg_sc
        self.lo_freq:float=lo_freq
        self.rf_amp: float=rf_amp
        self.trs:int=trs
        self.rf_pi2_duration:int=rf_pi2_duration
        self.echo_duration:int=echo_duration
        self.readout_duration:int=readout_duration
        self.BW:float=BW
        self.shim:list=shim
        self.readout_amp:float=readout_amp
        self.readout_grad_duration:int=readout_grad_duration
        self.trap_ramp_duration:int=trap_ramp_duration
    
    @property
    def systemproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.lo_freq: [float(self.lo_freq)],
            nmspc.rf_amp: [float(self.rf_amp)],
            nmspc.dbg_sc:[float(self.dbg_sc)], 
            nmspc.trs:[int(self. trs)]
        }

    @property
    def sqncproperties(self) -> dict:
        return{
            nmspc.rf_pi2_duration:[int(self.rf_pi2_duration)],            
            nmspc.echo_duration:[int(self.echo_duration)], 
            nmspc.readout_duration:[int(self.readout_duration)], 
            nmspc.BW:[float(self.BW)], 
            nmspc.readout_amp:[float(self.readout_amp)], 
            nmspc.readout_grad_duration:[int(self.readout_grad_duration)], 
            nmspc.trap_ramp_duration:[int(self.trap_ramp_duration)]
        }    
        
    @property
    def gradientshims(self) -> dict:
        return{
            nmspc.shim:[list(self.shim)]
        }
#        
#class SpinEcho2DSeq:
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
#                 BW:float=None, 
#                 shim: list=None, 
#                 readout_amp:float=None, 
#                 readout_grad_duration:int=None, 
#                 trap_ramp_duration:int=None, 
#                 phase_start_amp:float=None, 
#                 phase_grad_duration:int=None, 
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
#        self.BW:float=BW
#        self.shim:list=shim
#        self.readout_amp:float=readout_amp
#        self.readout_grad_duration:int=readout_grad_duration
#        self.trap_ramp_duration:int=trap_ramp_duration
#        self.phase_start_amp:float=phase_start_amp
#        self.phase_grad_duration:int=phase_grad_duration
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
#            nmspc.BW:[float(self.BW)], 
#            nmspc.readout_amp:[float(self.readout_amp)], 
#            nmspc.readout_grad_duration:[int(self.readout_grad_duration)], 
#            nmspc.trap_ramp_duration:[int(self.trap_ramp_duration)], 
#            nmspc.phase_start_amp:[float(self.phase_start_amp)], 
#            nmspc.phase_grad_duration:[int(self.phase_grad_duration)]
#        }    
#        
#    @property
#    def gradientshims(self) -> dict:
#        return{
#            nmspc.shim:[list(self.shim)]
#        }
#        
class SpinEchoSeq:
    
    def __init__(self, 
                 seq:str, 
                 lo_freq: float=None, 
                 rf_amp: float=None, 
                 rf_pi2_duration: int=None, 
                 echo_duration:int=None, 
                 tr:int=None, 
                 BW:float=None, 
                 nScans:int=None, 
                 shim: list=None,             
                 readout_amp:float=None, 
                 trap_ramp_duration:int=None, 
                 phase_start_amp:float=None, 
                 phase_grad_duration:int=None, 
                 slice_start_amp:float=None, 
                 n:list=None, 
                 fov:list=None
                 ):
    
        self.seq:str=seq
        self.lo_freq:float=lo_freq
        self.rf_amp: float=rf_amp
        self.rf_pi2_duration:int=rf_pi2_duration
        self.echo_duration:int=echo_duration
        self.tr:int=tr
        self.BW:float=BW
        self.nScans:int=nScans
        self.shim:list=shim
        self.readout_amp:float=readout_amp
        self.trap_ramp_duration:int=trap_ramp_duration
        self.phase_start_amp:float=phase_start_amp
        self.phase_grad_duration:int=phase_grad_duration
        self.slice_start_amp:float=slice_start_amp
        self.n:list=n
        self.fov:list=fov
    
    @property
    def RFproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.lo_freq: [float(self.lo_freq)], 
            nmspc.BW:[float(self.BW)], 
            nmspc.nScans:[int(self.nScans)], 
            nmspc.n:[list(self.n)], 
            nmspc.fov:[list(self.fov)], 
            nmspc.tr:[int(self.tr)], 
            nmspc.echo_duration:[int(self.echo_duration)],
            nmspc.rf_amp: [float(self.rf_amp)],
            nmspc.rf_pi2_duration:[int(self.rf_pi2_duration)], 
        }

    @property
    def Gproperties(self) -> dict:
        return{
            nmspc.trap_ramp_duration:[int(self.trap_ramp_duration)], 
            nmspc.readout_amp:[float(self.readout_amp)], 
            nmspc.phase_start_amp:[float(self.phase_start_amp)], 
            nmspc.phase_grad_duration:[int(self.phase_grad_duration)], 
            nmspc.slice_start_amp:[float(self.slice_start_amp)], 

        }    
        
    @property
    def gradientshims(self) -> dict:
        return{
            nmspc.shim:[list(self.shim)]
        }

class FIDSeq:
    
    def __init__(self, 
                 seq:str, 
                 dbg_sc: float=None, 
                 lo_freq: float=None, 
                 rf_amp: float=None, 
                 rf_duration: int=None, 
                 rf_tstart:int=None, 
                 rf_wait:int=None, 
                 rx_period:float=None, 
                 readout_duration:int=None
                 ):
        
        self.seq: str=seq
        self.dbg_sc: float= dbg_sc
        self.lo_freq: float=lo_freq
        self.rf_amp: float=rf_amp
        self.rf_duration: int=rf_duration
        self.rf_tstart:int=rf_tstart
        self.rf_wait:int=rf_wait
        self.rx_period:float=rx_period
        self.readout_duration:int=readout_duration

    @property
    def systemproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.lo_freq: [float(self.lo_freq)],
            nmspc.rf_amp: [float(self.rf_amp)],
            nmspc.dbg_sc:[float(self.dbg_sc)]
        }

    @property
    def sqncproperties(self) -> dict:
        return{
            nmspc.rf_duration:[int(self.rf_duration)],            
            nmspc.rf_tstart:[int(self.rf_tstart)], 
            nmspc.rf_wait:[int(self.rf_wait)], 
            nmspc.rx_period:[float(self.rx_period)], 
            nmspc.readout_duration:[int(self.readout_duration)]
        }    

    
        
class RadialSeq:
    """
    Spectrum Operation Class
    """
    def __init__(self,
                 seq: str,
                 dbg_sc: float=None, 
                 lo_freq: float=None,
                 rf_amp: float=None,
                 trs: int=None,
                 gradients: float=None, 
                 grad_tstart:int=None, 
                 tr_total_time: int=None, 
                 rf_tstart: int=None, 
                 rf_tend: int=None, 
                 rx_tstart: int=None, 
                 rx_tend: int=None, 
                 rx_period: int=None, 
                 shim: list=None
                 ):
        """
        Initialization of spectrum operation class
        @param frequency:       Frequency value for operation
        @return:                None
        """
#        if shim is None:
#            shim=[0, 0, 0, 0]
#        else:
#            while len(shim) < 4:
#                shim += [0]
        self.seq: str=seq
        self.dbg_sc: float= dbg_sc
        self.lo_freq: float=lo_freq
        self.rf_amp: float=rf_amp
        self.trs:int=trs
        self.G: float=gradients
        self.grad_tstart: int=grad_tstart
        self.tr_total_time: int=tr_total_time
        self.rf_tstart: int=rf_tstart
        self.rf_tend: int=rf_tend
        self.rx_tstart: int=rx_tstart
        self.rx_tend: int=rx_tend
        self.rx_period: float=rx_period
        self.shim:list=shim
#        self.x_shim: float=shim[0]
#        self.y_shim: float=shim[1]
#        self.z_shim: float=shim[2]
#        self.z2_shim: float = shim[3]

    @property
    def systemproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.lo_freq: [float(self.lo_freq)],
            nmspc.rf_amp: [self.rf_amp],
            nmspc.trs:[int(self. trs)], 
            nmspc.dbg_sc:[float(self.dbg_sc)]
        }

    @property
    def sqncproperties(self) -> dict:
        return{
            nmspc.G:[float(self.G)], 
            nmspc.grad_tstart:[int(self.grad_tstart)], 
            nmspc.tr_total_time:[int(self.tr_total_time)], 
            nmspc.rf_tstart:[int(self.rf_tstart)], 
            nmspc.rf_tend:[int(self.rf_tend)], 
            nmspc.rx_tstart:[int(self.rx_tstart)], 
            nmspc.rx_tend:[int(self.rx_tend)], 
            nmspc.rx_period:[float(self.rx_period)]
            
        }    

    @property
    def gradientshims(self) -> dict:
        return{
#           nmspc.x_shim:[float(self.x_shim)], 
#           nmspc.y_shim:[float(self.y_shim)], 
#           nmspc.z_shim:[float(self.z_shim)], 
#           nmspc.z2_shim:[float(self.z2_shim)]
            nmspc.shim:[list(self.shim)]
        }  
           

class GradEchoSeq:
    """
    Spectrum Operation Class
    """
    def __init__(self,
                 seq: str,
                 dbg_sc: float=None, 
                 lo_freq: float=None,
                rf_amp: float=None,
                 trs: int=None,
                 rx_period: int=None, 
                 rf_tstart: int=None, 
                 slice_amp: float=None, 
                 phase_amp: float=None, 
                 readout_amp: float=None, 
                 rf_duration: int=None, 
                 trap_ramp_duration: int=None, 
                 phase_delay: int=None, 
                 phase_duration: int=None, 
                 shim: list=None
                 ):

                     
        """
        Initialization of gradient echo sequence class
        @param frequency:       Frequency value for operation
        @param amplification:     Attenuation value for operation
        @param shim:            Shim values for operation
        @return:                None
        """
        self.seq: str=seq
        self.dbg_sc: float=dbg_sc
        self.lo_freq: float=lo_freq
        self.rf_amp: float=rf_amp
        self.trs:int=trs
        self.rx_period: float=rx_period
        self.rf_tstart: int=rf_tstart
        self.slice_amp: float=slice_amp
        self.phase_amp: float=phase_amp
        self.readout_amp: float=readout_amp
        self.rf_duration: int=rf_duration
        self.trap_ramp_duration: int=trap_ramp_duration
        self.phase_delay: int=phase_delay
        self.phase_duration: int=phase_duration
        self.shim:list=shim
        

    @property
    def systemproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.lo_freq: [float(self.lo_freq)],
            nmspc.rf_amp: [self.rf_amp],
            nmspc.trs:[int(self. trs)], 
            nmspc.dbg_sc:[float(self.dbg_sc)]
        }

    @property
    def sqncproperties(self) -> dict:
        return{
            nmspc.rx_period:[float(self.rx_period)], 
            nmspc.rf_tstart:[int(self.rf_tstart)],         
            nmspc.slice_amp:[float(self.slice_amp)], 
            nmspc.phase_amp:[float(self.phase_amp)], 
            nmspc.readout_amp:[float(self.readout_amp)], 
            nmspc.rf_duration:[int(self.rf_duration)], 
            nmspc.trap_ramp_duration:[int(self.trap_ramp_duration)], 
            nmspc.phase_delay:[int(self.phase_delay)], 
           nmspc.phase_duration:[int(self.phase_duration)]
        }
           

    @property
    def gradientshims(self) -> dict:
        return{
            nmspc.shim:[list(self.shim)]
        } 

class TSE_Seq:
    """
    Spectrum Operation Class
    """
    def __init__(self,
                 seq: str,
                 dbg_sc: float=None, 
                 lo_freq: float=None,
                 rf_amp: float=None,
                 trs: int=None,
                 rx_period: int=None, 
                 trap_ramp_duration: int=None, 
                 echos_per_tr: int=None, 
                 echo_duration: int=None, 
                 slice_start_amp: float=None, 
                 phase_start_amp: float=None, 
                 readout_amp: float=None, 
                 rf_pi2_duration: int=None, 
                 phase_grad_duration: int=None, 
                 readout_duration: int=None, 
                 readout_grad_duration: int=None, 
                 phase_grad_interval: int=None, 
                 tr_pause_duration: int=None, 
                 shim: list=None
                 ):

 
        """
        Initialization of gradient echo sequence class
        @param frequency:       Frequency value for operation
        @param amplification:     Attenuation value for operation
        @param shim:            Shim values for operation
        @return:                None
        """
        
        self.seq: str=seq
        self.dbg_sc: float=dbg_sc
        self.lo_freq: float=lo_freq
        self.rf_amp: float=rf_amp
        self.trs:int=trs
        self.rx_period: float=rx_period
        self.trap_ramp_duration: int=trap_ramp_duration        
        self.echos_per_tr: int=echos_per_tr
        self.echo_duration: int=echo_duration
        self.slice_start_amp: float=slice_start_amp        
        self.phase_start_amp: float=phase_start_amp  
        self.readout_amp: float=readout_amp     
        self.rf_pi2_duration: int=rf_pi2_duration   
        self.phase_grad_duration: int=phase_grad_duration    
        self.readout_duration: int=readout_duration
        self.readout_grad_duration: int=readout_grad_duration
        self.phase_grad_interval: int=phase_grad_interval
        self.tr_pause_duration: int=tr_pause_duration
        self.shim:list=shim

    @property
    def systemproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.lo_freq: [float(self.lo_freq)],
            nmspc.rf_amp: [self.rf_amp],
            nmspc.trs:[int(self. trs)], 
            nmspc.dbg_sc:[float(self.dbg_sc)]
        }
        
    @property
    def sqncproperties(self) -> dict:
        return{
            nmspc.rx_period:[float(self.rx_period)], 
            nmspc.trap_ramp_duration:[int(self.trap_ramp_duration)], 
            nmspc.echos_per_tr:[int(self.echos_per_tr)],         
            nmspc.echo_duration:[int(self.echo_duration)],  
            nmspc.slice_start_amp:[float(self.slice_start_amp)], 
            nmspc.phase_start_amp:[float(self.phase_start_amp)], 
            nmspc.readout_amp:[float(self.readout_amp)], 
            nmspc.rf_pi2_duration:[int(self.rf_pi2_duration)], 
            nmspc.phase_grad_duration:[int(self.phase_grad_duration)], 
            nmspc.readout_duration:[int(self.readout_duration)], 
            nmspc.readout_grad_duration:[int(self.readout_grad_duration)], 
            nmspc.phase_grad_interval:[int(self.phase_grad_interval)], 
            nmspc.tr_pause_duration:[int(self.tr_pause_duration)]
        }    
        
    @property
    def gradientshims(self) -> dict:
        return{
            nmspc.shim:[list(self.shim)]
        } 

"""
Definition of default sequences
"""
defaultsequences={

    #SpinEchoSeq(dbg_sc,lo_freq,rf_amp,trs,rf_pi2_duration,echo_duration,readout_duration,BW,shimming(rd,ph,sl))
#    'Spin Echo': SpinEchoSeq('SE', 0, 3.069, 0.62, 1, 70, 2000, 1000, 0.05, (0,  0,  0)), 
    #SpinEcho1DSeq(dbg_sc,lo_freq,rf_amp,trs,rf_pi2_duration,echo_duration,readout_duration,BW,shimming(rd,ph,sl))
    'Spin Echo 1D': SpinEcho1DSeq('SE1D', 0, 3.069, 0.62, 1, 70, 2000, 1000, 0.05, (0,  0,  0), 0.2, 700, 250), 
    #SpinEcho2DSeq(dbg_sc,lo_freq,rf_amp,trs,rf_pi2_duration,echo_duration,readout_duration,BW,shimming(rd,ph,sl),phase_start_amp,phase_grad_duration,n(sr,sph,sl))
#    'Spin Echo 2D': SpinEcho2DSeq('SE2D', 0, 3.069, 0.8, 1, 50, 2000, 500, 0.3, (0,  0,  0), 0.8, 700, 100, 0.6, 150), 
    #SpinEchoSeq(lo_freq,rf_amp,rf_pi2_duration,TE,TR,BW,nScans,shimming(rd,ph,sl),readout_amp, trap_ramp_duration,phase_start_amp,phase_grad_duration,slice_start_amp,n(x,y,z),fov(rd,ph,sl))
    'Spin Echo': SpinEchoSeq('SE', 3.069, 0.2, 50, 2000, 5000, 0.3, 10, (0,  0,  0), 0.8, 100, 0.6, 150, 0.6, (10, 10, 10), (100, 100, 100)), 
    #FID(dbg_sc,lo_freq,rf_amp,rf_duration,rf_tstart,rf_wait,rx_period,readout_duration)
    'Free Induction Decay': FIDSeq('FID', 0, 3, 0.6, 50, 100, 100, 3.333, 500), 
    #RadialSeq(dbg_sc,lo_freq,rf_amp,trs,G,grad_tstart,TR,rf_tstart,rf_tend,rx_tstart,rx_tend,rx_period,shimming(rd,ph,sl))
    'Radial': RadialSeq('R', 0, 3, 0.2, 3, 0.5, 0, 220, 5, 50, 70, 180, 3.333, (0.01,  0.01,  0.01)),
    #GradEchoSeq(dbg_sc,lo_freq,rf_amp,trs,rx_period,rf_tstart,sliceAmp,phAmp,rdAmp,rfDur,trapRampDur,phDelay,phDur,shimming(rd,ph,sl))
    'Gradient Echo': GradEchoSeq('GE',0,  3, 0.1, 2, 3.333, 100, 0.4, 0.3, 0.8, 50, 100, 100, 200, (0.01, 0.01, 0.01)), 
    #TurboSpinEcho(dbg_sc,lo_freq,rf_amp,trs,rx_period,trapRampDur,echosTR,echosDur,sliceAmp,phAmp,rdAmp,rfDur,phDur,rdDur,rdGradDur,phGint,TRPauseDur,shimming(rd,ph,sl))
    'Turbo Spin Echo': TSE_Seq('TSE',  0, 3, 1, 5, 3.333, 100, 5, 2000, 0.3, 0.6,0.8, 50, 150, 500, 700, 1200, 3000, (0.01, 0.01, 0.01))
}

 
