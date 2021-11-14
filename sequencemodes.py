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


class SpinEchoSeq:
    
    def __init__(self, 
                 seq:str, 
                 lo_freq: float=None, 
                 rf_amp: float=None, 
                 rf_pi2_duration: int=None, 
                 echo_duration:int=None, 
                 tr_duration:int=None, 
                 BW:int=None, 
                 nScans:int=None, 
                 shim: list=None,             
                 trap_ramp_duration:int=None, 
                 phase_grad_duration:int=None, 
                 n:list=None, 
                 fov:list=None, 
                 preemph_factor:float=None
                 ):
    
        self.seq:str=seq
        self.lo_freq:float=lo_freq
        self.rf_amp: float=rf_amp
        self.rf_pi2_duration:int=rf_pi2_duration
        self.echo_duration:int=echo_duration
        self.tr_duration:int=tr_duration
        self.BW:int=BW
        self.nScans:int=nScans
        self.shim:list=shim
        self.trap_ramp_duration:int=trap_ramp_duration
        self.phase_grad_duration:int=phase_grad_duration
        self.n:list=n
        self.fov:list=fov
        self.preemph_factor:float=preemph_factor
    
    @property
    def RFproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.lo_freq: [float(self.lo_freq)], 
            nmspc.BW:[int(self.BW)], 
            nmspc.nScans:[int(self.nScans)], 
            nmspc.n:[list(self.n)], 
            nmspc.fov:[list(self.fov)], 
            nmspc.tr_duration:[int(self.tr_duration)], 
            nmspc.echo_duration:[int(self.echo_duration)],
            nmspc.rf_amp: [float(self.rf_amp)],
            nmspc.rf_pi2_duration:[int(self.rf_pi2_duration)], 
        }

    @property
    def Gproperties(self) -> dict:
        return{
            nmspc.trap_ramp_duration:[int(self.trap_ramp_duration)], 
            nmspc.phase_grad_duration:[int(self.phase_grad_duration)], 
            nmspc.preemph_factor:[float(self.preemph_factor)]
        }    
        
    @property
    def gradientshims(self) -> dict:
        return{
            nmspc.shim:[list(self.shim)]
        }
        
class TurboSpinEchoSeq:
    
    def __init__(self, 
                 seq:str, 
                 lo_freq: float=None, 
                 rf_amp: float=None, 
                 rf_pi2_duration: int=None, 
                 echo_duration:int=None, 
                 tr_duration:int=None, 
                 BW:int=None, 
                 nScans:int=None, 
                 shim: list=None,             
                 trap_ramp_duration:int=None, 
                 phase_grad_duration:int=None, 
                 axes:list=None, 
                 n:list=None, 
                 fov:list=None, 
                 preemph_factor:float=None, 
                 echos_per_tr:int=None, 
                 sweep_mode:int=None, 
                 par_acq_factor:int=None
                 ):
    
        self.seq:str=seq
        self.lo_freq:float=lo_freq
        self.rf_amp: float=rf_amp
        self.rf_pi2_duration:int=rf_pi2_duration
        self.echo_duration:int=echo_duration
        self.tr_duration:int=tr_duration
        self.BW:int=BW
        self.nScans:int=nScans
        self.shim:list=shim
        self.trap_ramp_duration:int=trap_ramp_duration
        self.phase_grad_duration:int=phase_grad_duration
        self.n:list=n
        self.fov:list=fov
        self.preemph_factor:float=preemph_factor
        self.echos_per_tr:int=echos_per_tr
        self.sweep_mode:int=sweep_mode
        self.par_acq_factor:int=par_acq_factor
        self.axes:list=axes
    
    @property
    def RFproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.lo_freq: [float(self.lo_freq)], 
            nmspc.BW:[int(self.BW)], 
            nmspc.nScans:[int(self.nScans)], 
            nmspc.axes:[list(self.axes)], 
            nmspc.n:[list(self.n)], 
            nmspc.fov:[list(self.fov)], 
            nmspc.tr_duration:[int(self.tr_duration)], 
            nmspc.echo_duration:[int(self.echo_duration)],
            nmspc.rf_amp: [float(self.rf_amp)],
            nmspc.rf_pi2_duration:[int(self.rf_pi2_duration)], 
            nmspc.echos_per_tr:[int(self.echos_per_tr)], 
            nmspc.par_acq_factor:[int(self.par_acq_factor)]
        }

    @property
    def Gproperties(self) -> dict:
        return{
            nmspc.trap_ramp_duration:[int(self.trap_ramp_duration)], 
            nmspc.phase_grad_duration:[int(self.phase_grad_duration)], 
            nmspc.preemph_factor:[float(self.preemph_factor)], 
            nmspc.sweep_mode:[int(self.sweep_mode)]
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

class CPMGSeq:
    
   
    def __init__(self, 
                 seq:str, 
                 larmorFreq: float=None, 
                 rfExAmp: float=None, 
                 rfReAmp: float=None, 
                 rfExTime: int=None, 
                 rfReTime:int=None, 
                 echoSpacing:int=None, 
                 nPoints:int=None, 
                 etl:int=None, 
                 acqTime: int=None,             
                 ):
    
        self.seq:str=seq
        self.larmorFreq:float=larmorFreq
        self.rfExAmp: float=rfExAmp
        self.rfReAmp:float=rfReAmp
        self.rfExTime:float=rfExTime
        self.rfReTime:float=rfReTime
        self.echoSpacing:float=echoSpacing
        self.nPoints:int=nPoints
        self.etl:int=etl
        self.acqTime:float=acqTime
       
    @property
    def  systemproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.larmorFreq: [float(self.larmorFreq)], 
            nmspc.rfExAmp:[float(self.rfExAmp)], 
            nmspc.rfReAmp:[float(self.rfReAmp)], 
            nmspc.rfExTime:[float(self.rfExTime)], 
            nmspc.rfReTime:[float(self.rfReTime)], 
            nmspc.echoSpacing:[float(self.echoSpacing)], 
            nmspc.nPoints:[int(self.nPoints)], 
            nmspc.etl:[int(self.etl)],
            nmspc.acqTime: [float(self.acqTime)],
        }

class RARE:
    def __init__(self, 
                 seq:str, 
                 nScans:int=None, 
                 
                 larmorFreq: float=None, 
                 rfExAmp: float=None, 
                 rfReAmp: float=None, 
                 rfExTime:int=None, 
                 rfReTime:int=None, 
                 echoSpacing:int=None, 
                 acqTime:int=None, 
                 shimming:list=None, 
                 
                 repetitionTime:int = None, 
                 inversionTime:int=None, 
                 fov:list=None, 
                 dfov:list=None,
                 
                 nPoints:list=None, 
                 etl:int=None, 
                 
                 axes:list=None, 
                 axesEnable:list=None, 
                 sweepMode:int=None, 
                 phaseGradTime:int=None,  
                 rdPreemphasis:float = None,
                 drfPhase:int = None, 
                 dummyPulses:int = None, 
                 
                 parAcqLines:int = None, 
                 ):
    
        self.seq:str=seq
        self.larmorFreq:float=larmorFreq
        self.rfExAmp: float=rfExAmp
        self.rfReAmp:float=rfReAmp
        self.rfExTime:int=rfExTime
        self.rfReTime:int=rfReTime
        self.echoSpacing:int=echoSpacing
        self.nPoints:int=nPoints
        self.etl:int=etl
        self.acqTime:int=acqTime
        self.shimming:list=shimming
        self.nScans:int=nScans
        self.repetitionTime:int= repetitionTime
        self.inversionTime:int=inversionTime
        self.fov:list=fov
        self.dfov:list=dfov
        self.axes:list=axes
        self.axesEnable:list=axesEnable
        self.sweepMode:int=sweepMode
        self.phaseGradTime:int=phaseGradTime 
        self.rdPreemphasis:float = rdPreemphasis
        self.drfPhase:int = drfPhase 
        self.dummyPulses:int = dummyPulses 
        self.parAcqLines:int = parAcqLines
    @property
    def  systemproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.nScans: [int(self.nScans)], 
            nmspc.larmorFreq: [float(self.larmorFreq)], 
            nmspc.rfExAmp:[float(self.rfExAmp)], 
            nmspc.rfReAmp:[float(self.rfReAmp)], 
            nmspc.rfExTime:[int(self.rfExTime)], 
            nmspc.rfReTime:[int(self.rfReTime)], 
            nmspc.echoSpacing:[int(self.echoSpacing)], 
            nmspc.acqTime: [int(self.acqTime)],
            nmspc.shimming: [list(self.shimming)],
            
            nmspc. repetitionTime:[int(self. repetitionTime)], 
            nmspc.inversionTime:[int(self.inversionTime)], 
            nmspc.axesEnable: [list(self.axesEnable)], 
            nmspc.axes: [list(self.axes)], 
            nmspc.fov:[list(self.fov)], 
            nmspc.dfov: [list(self.dfov)],
            nmspc.nPoints:[list(self.nPoints)], 
            nmspc.etl:[int(self.etl)],
    
            nmspc.sweepMode:[int(self.sweepMode)], 
            nmspc.phaseGradTime:[int(self.phaseGradTime)], 
            nmspc.rdPreemphasis:[float(self.rdPreemphasis)], 
            nmspc.drfPhase:[int(self.drfPhase)], 
            nmspc.dummyPulses :[int(self.dummyPulses)], 
            nmspc.parAcqLines :[int(self.parAcqLines)], 

        }

"""
Definition of default sequences
"""
defaultsequences={

    #SpinEchoSeq(lo_freq,rf_amp,rf_pi2_duration,TE,TR,BW,nScans,shimming(rd,ph,sl), trap_ramp_duration,phase_grad_duration,n(x,y,z),fov(rd,ph,sl),preemph_factor)
    'Spin Echo': SpinEchoSeq('SE', 3.03, 0.6, 65, 10, 500, 31, 1, (0,  0,  0), 1000, 100, (40, 1, 1), (20, 20, 15), 1.05), 
    #SpinEchoSeq(lo_freq,rf_amp,rf_pi2_duration,TE,TR,BW,nScans,shimming(rd,ph,sl), trap_ramp_duration,phase_grad_duration,n(x,y,z),fov(rd,ph,sl),preemph_factor,echos_per_tr,sweep_mode,par_acq_factor)
    'Turbo Spin Echo': TurboSpinEchoSeq('TSE', 3.0807, 0.3,30, 20, 1000, 30, 1, (0,  0,  0), 100, 500, (1, 2, 3), (60, 1, 1), (15, 10, 10), 1.0, 1, 1, 0),
    #FID(dbg_sc,lo_freq,rf_amp,rf_duration,rf_tstart,rf_wait,rx_period,readout_duration)
    'Free Induction Decay': FIDSeq('FID', 0, 3, 0.6, 50, 100, 100, 3.333, 500), 
    #RadialSeq(dbg_sc,lo_freq,rf_amp,trs,G,grad_tstart,TR,rf_tstart,rf_tend,rx_tstart,rx_tend,rx_period,shimming(rd,ph,sl))
    'Radial': RadialSeq('R', 0, 3, 0.2, 3, 0.5, 0, 220, 5, 50, 70, 180, 3.333, (0.01,  0.01,  0.01)),
    #GradEchoSeq(dbg_sc,lo_freq,rf_amp,trs,rx_period,rf_tstart,sliceAmp,phAmp,rdAmp,rfDur,trapRampDur,phDelay,phDur,shimming(rd,ph,sl))
    'Gradient Echo': GradEchoSeq('GE',0,  3, 0.1, 2, 3.333, 100, 0.4, 0.3, 0.8, 50, 100, 100, 200, (0.01, 0.01, 0.01)), 
    #TurboSpinEcho(dbg_sc,lo_freq,rf_amp,trs,rx_period,trapRampDur,echosTR,echosDur,sliceAmp,phAmp,rdAmp,rfDur,phDur,rdDur,rdGradDur,phGint,TRPauseDur,shimming(rd,ph,sl))
#    'Turbo Spin Echo': TSE_Seq('TSE',  0, 3, 1, 5, 3.333, 100, 5, 2000, 0.3, 0.6,0.8, 50, 150, 500, 700, 1200, 3000, (0.01, 0.01, 0.01))
    'CPMG': CPMGSeq('CPMG', 3.08e6, 0.3, 0.3, 35e-6, 70e-6, 10e-3, 500, 100, 2e-3), 
    'RARE': RARE('RARE', 1, 3.08, 0.3, 0, 35, 0, 20, 4,[-70, -90, 10],  500, 0, (120, 120,120),  (0, 0, 0), (60, 1, 1), 15, (0, 1, 2), (0, 0, 0), 1, 1000, 1, 0, 1, 0) }
