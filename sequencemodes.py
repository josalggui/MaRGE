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

class RARE:
    def __init__(self, 
                 seq:str='RARE', 
                 nScans:int=1, 
                 larmorFreq: float=3.08, 
                 rfExAmp: float=0.3, 
                 rfReAmp: float=0.3, 
                 rfExTime:float=30.0, 
                 rfReTime:float=60.0, 
                 echoSpacing:float=20.0, 
                 preExTime:float=0.0, 
                 inversionTime:float=0.0, 
                 repetitionTime:float=500.0, 
                 fov:list=[120.0, 120.0, 120.0], 
                 dfov:list=[0.0, 0.0, 0.0],
                 nPoints:list=[60, 1, 1], 
                 etl:int=15, 
                 acqTime:float=4.0, 
                 axes:list=[0, 1, 2], 
                 axesEnable:list=[0, 0, 0], 
                 sweepMode:int=1, 
                 rdGradTime:float=5.0, 
                 rdDephTime:float=1.0,
                 phGradTime:float=1.0,
                 rdPreemphasis:float = 1.0,
                 drfPhase:float = 0.0, 
                 dummyPulses:int = 1, 
                 shimming:list=[-70.0, -90.0, 10.0], 
                 parFourierFraction:float = 1.0, 
                 ):

        self.seq:str=seq 
        self.nScans:int=nScans
        self.larmorFreq: float=larmorFreq
        self.rfExAmp: float=rfExAmp
        self.rfReAmp: float=rfReAmp
        self.rfExTime:float=rfExTime
        self.rfReTime:float=rfReTime
        self.echoSpacing:float=echoSpacing
        self.preExTime:float=preExTime
        self.inversionTime:float=inversionTime
        self.repetitionTime:float =repetitionTime
        self.fov:list=fov
        self.dfov:list=dfov
        self.nPoints:list=nPoints
        self.etl:int=etl
        self.acqTime:float=acqTime
        self.axes:list=axes
        self.axesEnable:list=axesEnable
        self.sweepMode:int=sweepMode
        self.rdGradTime:float= rdGradTime
        self.rdDephTime:float=rdDephTime
        self.phGradTime:float=phGradTime
        self.rdPreemphasis:float = rdPreemphasis
        self.drfPhase:float = drfPhase
        self.dummyPulses:int = dummyPulses
        self.shimming:list=shimming
        self.parFourierFraction:float = parFourierFraction

    @property
    def RFproperties(self) -> dict:
        return{
            nmspc.larmorFreq:[float(self.larmorFreq)],
            nmspc.rfExAmp:[float(self.rfExAmp)], 
            nmspc.rfReAmp:[float(self.rfReAmp)], 
            nmspc.rfExTime:[float(self.rfExTime)], 
            nmspc.rfReTime:[float(self.rfReTime)], 
            nmspc.drfPhase:[float(self.drfPhase)], 
            }
    
    @property
    def IMproperties(self) -> dict:
        return{
            nmspc.nScans:[int(self.nScans)],
            nmspc.nPoints:[list(self.nPoints)],  
            nmspc.axes:[list(self.axes)], 
            nmspc.axesEnable:[list(self.axesEnable)], 
            nmspc.fov:[list(self.fov)], 
            nmspc.dfov:[list(self.dfov)], 
            }
    
    @property
    def SEQproperties(self) -> dict:
        return{
            nmspc.etl:[int(self.etl)],
            nmspc.echoSpacing:[float(self.echoSpacing)], 
            nmspc.repetitionTime:[float(self.repetitionTime)], 
            nmspc.acqTime:[float(self.acqTime)],
            nmspc.preExTime:[float(self.preExTime)], 
            nmspc.inversionTime:[float(self.inversionTime)], 
            nmspc.sweepMode:[int(self.sweepMode)],
            nmspc.dummyPulses:[int(self.dummyPulses)], 
            nmspc.parFourierFraction:[float(self.parFourierFraction)],
            }
        
    @property
    def OTHproperties(self) -> dict:
        return{
            nmspc.shimming:[list(self.shimming)],
            nmspc.rdGradTime:[float(self.rdGradTime)], 
            nmspc.rdDephTime:[float(self.rdDephTime)], 
            nmspc.phGradTime:[float(self.phGradTime)],
            nmspc.rdPreemphasis:[float(self.rdPreemphasis)],
            }

class HASTE:
    def __init__(self, 
                 seq:str='HASTE', 
                 nScans:int=1, 
                 larmorFreq: float=3.08,            # MHz 
                 rfExAmp: float=0.058,              # a.u. 
                 rfReAmp: float=0.116,            # a.u. 
                 rfExTime:int=170.0,                  # us 
                 rfReTime:int=170.0,                  # us
                 rfEnvelope:str='Rec',              # 'Rec' or 'Sinc'
                 echoSpacing:float=10.0,              # ms 
                 preExTime:float=0.0,                 # ms 
                 inversionTime:float=0.0,             # ms
                 repetitionTime:int = 1000.0,          # ms
                 fov:list=[120., 120., 20.],         # mm 
                 dfov:list=[0., 0., 0.],                 # mm
                 nPoints:list=[60, 60, 1],          # points 
                 acqTime:int=4.,                     # ms 
                 axes:list=[0, 1, 2],               # [rd, ph, sl], 0->x, 1->y, 2->z 
                 axesEnable:list=[1, 1, 1],         # [rd, ph, sl], 0->Off, 1->On 
                 sweepMode:int=1,                   # 0->k2k, 1->02k, 2->k20 
                 rdGradTime:int=5.,                  # ms
                 rdDephTime:int=1.,                  # ms
                 phGradTime:int=1.,                  # ms
                 rdPreemphasis:float=1.,             # readout dephasing grad is multiplied by this number
                 ssPreemphasis:float=1.,             # slice rephasing grad is multiplied by this number
                 crusherDelay:float=0.,              # us
                 drfPhase:float = 0.,                # degrees, excitation pulse phase 
                 dummyPulses:int = 1,               # pulses 
                 shimming:list=[-70., -90., 10.],           # a.u.*1e4, shimming along the X,Y and Z axes
                 parFourierFraction:float=1.0,      # fraction of acquired k-space along phase direction 
                 ):
        self.seq:str=seq 
        self.nScans:int=nScans
        self.larmorFreq: float=larmorFreq
        self.rfExAmp: float=rfExAmp
        self.rfReAmp: float=rfReAmp
        self.rfExTime:int=rfExTime
        self.rfReTime:int=rfReTime
        self.rfEnvelope:str=rfEnvelope
        self.echoSpacing:float=echoSpacing
        self.preExTime:float=preExTime
        self.inversionTime:float=inversionTime
        self.repetitionTime:int=repetitionTime
        self.fov:list=fov
        self.dfov:list=dfov
        self.nPoints:list=nPoints
        self.acqTime:int=acqTime
        self.axes:list=axes
        self.axesEnable:list=axesEnable
        self.sweepMode:int=sweepMode
        self.rdGradTime:int= rdGradTime
        self.rdDephTime:int=rdDephTime
        self.phGradTime:int=phGradTime
        self.rdPreemphasis:float = rdPreemphasis
        self.ssPreemphasis:float = ssPreemphasis
        self.crusherDelay:float = crusherDelay
        self.drfPhase:int = drfPhase
        self.dummyPulses:int = dummyPulses
        self.shimming:list=shimming
        self.parFourierFraction:float = parFourierFraction
        
    @property 
    def  RFproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.larmorFreq:[float(self.larmorFreq)], 
            nmspc.rfExAmp:[float(self.rfExAmp)], 
            nmspc.rfReAmp:[float(self.rfReAmp)], 
            nmspc.rfExTime:[int(self.rfExTime)], 
            nmspc.rfReTime:[int(self.rfReTime)],
            nmspc.rfEnvelope:[str(self.rfEnvelope)],
            nmspc.drfPhase:[int(self.drfPhase)], 
            }
    
    @property
    def IMproperties(self) -> dict:
        return{
            nmspc.nScans:[int(self.nScans)],
            nmspc.nPoints:[list(self.nPoints)],  
            nmspc.axes:[list(self.axes)], 
            nmspc.axesEnable:[list(self.axesEnable)], 
            nmspc.fov:[list(self.fov)], 
            nmspc.dfov:[list(self.dfov)], 
            }
    
    @property
    def SEQproperties(self) -> dict:
        return{
            nmspc.echoSpacing:[float(self.echoSpacing)], 
            nmspc.repetitionTime:[int(self.repetitionTime)], 
            nmspc.acqTime:[int(self.acqTime)],
            nmspc.preExTime:[float(self.preExTime)], 
            nmspc.inversionTime:[float(self.inversionTime)], 
            nmspc.sweepMode:[int(self.sweepMode)],
            nmspc.dummyPulses:[int(self.dummyPulses)], 
            nmspc.parFourierFraction:[float(self.parFourierFraction)],
            }
        
    @property
    def OTHproperties(self) -> dict:
        return{
            nmspc.shimming:[list(self.shimming)],
            nmspc.rdGradTime:[int(self.rdGradTime)], 
            nmspc.rdDephTime:[int(self.rdDephTime)], 
            nmspc.phGradTime:[int(self.phGradTime)],
            nmspc.rdPreemphasis:[float(self.rdPreemphasis)],
            nmspc.ssPreemphasis:[float(self.ssPreemphasis)],
            nmspc.crusherDelay:[float(self.crusherDelay)],
            }


"""
Definition of default sequences
"""
defaultsequences={
    'RARE': RARE(),
    'HASTE': HASTE()
    }



