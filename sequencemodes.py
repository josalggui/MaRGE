"""
Operation Modes

@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   2.0 (Beta)
@change:    13/06/2020

@summary:   TBD

@status:    Under development, simple 1D spectrum operation implemented
@todo:      Add gradient waveform, add more properties, add more operation types (create directory for operations)
"""

from sequencesnamespace import Namespace as nmspc

class RadialSeq:
    """
    Spectrum Operation Class
    """
    def __init__(self,
                 seq: str,
                 dbg_sc: float = None, 
                 lo_freq: float = None,
                 rf_amp: float = None,
                 trs: int = None,
                 gradients: float = None, 
                 grad_tstart:int = None, 
                 tr_total_time: int = None, 
                 rf_tstart: int = None, 
                 rf_tend: int = None, 
                 rx_tstart: int = None, 
                 rx_tend: int = None, 
                 rx_period: int = None, 
#                 shim: list = None, 
                 ):
        """
        Initialization of spectrum operation class
        @param frequency:       Frequency value for operation
        @param amplification:     Attenuation value for operation
        @param shim:            Shim values for operation
        @return:                None
        """
#        if shim is None:
#            shim = [0, 0, 0, 0]
#        else:
#            while len(shim) < 4:
#                shim += [0]
        self.seq: str = seq
        self.dbg_sc: float= dbg_sc
        self.lo_freq: float = lo_freq
        self.rf_amp: float = rf_amp
        self.trs:int = trs
        self.G: float = gradients
        self.grad_tstart: int = grad_tstart
        self.tr_total_time: int = tr_total_time
        self.rf_tstart: int = rf_tstart
        self.rf_tend: int = rf_tend
        self.rx_tstart: int = rx_tstart
        self.rx_tend: int = rx_tend
        self.rx_period: float = rx_period

#        self._shim_x: int = shim[0]
#        self._shim_y: int = shim[1]
#        self._shim_z: int = shim[2]
#        self._shim_z2: int = shim[3]
        #self._sequence = sequence # sqncs.FID
        #self._sequencebytestream = Assembler().assemble(self._sequence.path)

    @property
    def systemproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.lo_freq: [float(self.lo_freq)],
            nmspc.rf_amp: [self.rf_amp],
            nmspc.trs:[int(self. trs)], 
            nmspc.dbg_sc:[float(self.dbg_sc)]
        }

#    @property
#    def gradientshims(self):
#        return {
#            nmspc.x_grad: [self._shim_x],
#            nmspc.y_grad: [self._shim_y],
#            nmspc.z_grad: [self._shim_z]
#            
#        }

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
           

class GradEchoSeq:
    """
    Spectrum Operation Class
    """
    def __init__(self,
                 seq: str,
                 dbg_sc: float = None, 
                 lo_freq: float = None,
                rf_amp: float = None,
                 trs: int = None,
                 rx_period: int = None, 
                 rf_tstart: int = None, 
                 slice_amp: float = None, 
                 phase_amp: float = None, 
                 readout_amp: float = None, 
                 rf_duration: int = None, 
                 trap_ramp_duration: int = None, 
                 phase_delay: int = None, 
                 phase_duration: int = None
#                 shim: list = None, 
                 ):

                     
        """
        Initialization of gradient echo sequence class
        @param frequency:       Frequency value for operation
        @param amplification:     Attenuation value for operation
        @param shim:            Shim values for operation
        @return:                None
        """
        self.seq: str = seq
        self.dbg_sc: float = dbg_sc
        self.lo_freq: float = lo_freq
        self.rf_amp: float = rf_amp
        self.trs:int = trs
        self.rx_period: float = rx_period
        self.rf_tstart: int = rf_tstart
        self.slice_amp: float = slice_amp
        self.phase_amp: float = phase_amp
        self.readout_amp: float = readout_amp
        self.rf_duration: int = rf_duration
        self.trap_ramp_duration: int = trap_ramp_duration
        self.phase_delay: int = phase_delay
        self.phase_duration: int = phase_duration
        

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

class TSE_Seq:
    """
    Spectrum Operation Class
    """
    def __init__(self,
                 seq: str,
                 dbg_sc: float = None, 
                 lo_freq: float = None,
                 rf_amp: float = None,
                 trs: int = None,
                 rx_period: int = None, 
                 trap_ramp_duration: int = None, 
                 echos_per_tr: int = None, 
                 echo_duration: int = None, 
                 slice_start_amp: float = None, 
                 phase_start_amp: float = None, 
                 readout_amp: float = None, 
                 rf_pi2_duration: int = None, 
                 phase_grad_duration: int = None, 
                 readout_duration: int = None, 
                 readout_grad_duration: int = None, 
                 phase_grad_interval: int = None, 
                 tr_pause_duration: int = None
                 ):

 
        """
        Initialization of gradient echo sequence class
        @param frequency:       Frequency value for operation
        @param amplification:     Attenuation value for operation
        @param shim:            Shim values for operation
        @return:                None
        """
        
        self.seq: str = seq
        self.dbg_sc: float = dbg_sc
        self.lo_freq: float = lo_freq
        self.rf_amp: float = rf_amp
        self.trs:int = trs
        self.rx_period: float = rx_period
        self.trap_ramp_duration: int = trap_ramp_duration        
        self.echos_per_tr: int = echos_per_tr
        self.echo_duration: int = echo_duration
        self.slice_start_amp: float = slice_start_amp        
        self.phase_start_amp: float = phase_start_amp  
        self.readout_amp: float = readout_amp     
        self.rf_pi2_duration: int = rf_pi2_duration   
        self.phase_grad_duration: int = phase_grad_duration    
        self.readout_duration: int = readout_duration
        self.readout_grad_duration: int = readout_grad_duration
        self.phase_grad_interval: int = phase_grad_interval
        self.tr_pause_duration: int = tr_pause_duration

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

"""
Definition of default sequences
"""
defaultsequences = {
    #
    #FID(dbg_sc,lo_freq,rf_amp,trs,rx_period,trapRampDur,rfDur)
    #'FID': FID('FID', 0.2, 1, 5, 3.333, 50, )
    #RadialSeq(dbg_sc,lo_freq,rf_amp,trs,G,grad_tstart,TR,rf_tstart,rf_tend,rx_tstart,rx_tend,rx_period)
    'Radial': RadialSeq('R', 0.5, 0.2, 0.2, 3, 0.5, 0, 220, 5, 50, 70, 180, 3.333),
    #GradEchoSeq(dbg_sc,lo_freq,rf_amp,trs,rx_period,rf_tstart,sliceAmp,phAmp,rdAmp,rfDur,trapRampDur,phDelay,phDur)
    'Gradient Echo': GradEchoSeq('GE',0.5,  0.1, 0.1, 2, 3.333, 100, 0.4, 0.3, 0.8, 50, 50, 100, 200), 
    #TurboSpinEcho(dbg_sc,lo_freq,rf_amp,trs,rx_period,trapRampDur,echosTR,echosDur,sliceAmp,phAmp,rdAmp,rfDur,phDur,rdDur,rdGradDur,phGint,TRPauseDur)
    'Turbo Spin Echo': TSE_Seq('TSE',  0.5, 0.2, 1, 5, 3.333, 50, 5, 2000, 0.3, 0.6,0.8, 50, 150, 500, 700, 1200, 3000 )

}

