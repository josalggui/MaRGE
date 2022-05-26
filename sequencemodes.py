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
                 rfExTime:int=30, 
                 rfReTime:int=60, 
                 echoSpacing:float=20, 
                 preExTime:float=0, 
                 inversionTime:float=0, 
                 repetitionTime:int=500, 
                 fov:list=[120, 120, 120], 
                 dfov:list=[0, 0, 0],
                 nPoints:list=[60, 1, 1], 
                 etl:int=1, 
                 acqTime:int=4, 
                 axes:list=[0, 1, 2], 
                 axesEnable:list=[0, 0, 0], 
                 sweepMode:int=1, 
                 rdGradTime:int=5, 
                 rdDephTime:int=1,
                 phGradTime:int=1,
                 rdPreemphasis:float = 1,
                 drfPhase:int = 0, 
                 dummyPulses:int = 1, 
                 shimming:list=[-70, -90, 10], 
                 parFourierFraction:float = 1, 
                 ):

        self.seq:str=seq 
        self.nScans:int=nScans
        self.larmorFreq: float=larmorFreq
        self.rfExAmp: float=rfExAmp
        self.rfReAmp: float=rfReAmp
        self.rfExTime:int=rfExTime
        self.rfReTime:int=rfReTime
        self.echoSpacing:float=echoSpacing
        self.preExTime:float=preExTime
        self.inversionTime:float=inversionTime
        self.repetitionTime:int =repetitionTime
        self.fov:list=fov
        self.dfov:list=dfov
        self.nPoints:list=nPoints
        self.etl:int=etl
        self.acqTime:int=acqTime
        self.axes:list=axes
        self.axesEnable:list=axesEnable
        self.sweepMode:int=sweepMode
        self.rdGradTime:int= rdGradTime
        self.rdDephTime:int=rdDephTime
        self.phGradTime:int=phGradTime
        self.rdPreemphasis:float = rdPreemphasis
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
            nmspc.echoSpacing:[float(self.echoSpacing)], 
            nmspc.repetitionTime:[int(self.repetitionTime)], 
            nmspc.acqTime:[int(self.acqTime)], 
            nmspc.axes:[list(self.axes)], 
            nmspc.axesEnable:[list(self.axesEnable)], 
            nmspc.preExTime:[float(self.preExTime)], 
            nmspc.inversionTime:[float(self.inversionTime)], 
            nmspc.fov:[list(self.fov)], 
            nmspc.dfov:[list(self.dfov)], 
            nmspc.sweepMode:[int(self.sweepMode)], 
            nmspc.rdGradTime:[int(self.rdGradTime)], 
            nmspc.rdDephTime:[int(self.rdDephTime)], 
            nmspc.phGradTime:[int(self.phGradTime)], 
            nmspc.rdPreemphasis:[float(self.rdPreemphasis)], 
            nmspc.dummyPulses:[int(self.dummyPulses)], 
<<<<<<< HEAD
            nmspc.shimming:[list(self.shimming)], 
            nmspc.parFourierFraction:[float(self.parFourierFraction)], 
=======
            nmspc.parAcqLines:[int(self.parAcqLines)], 
>>>>>>> GUI_changes
            nmspc.drfPhase:[int(self.drfPhase)], 
        }
    @property
    def IMproperties(self) -> dict:
        return{
            nmspc.nScans:[int(self.nScans)],
            nmspc.nPoints:[list(self.nPoints)],             
        }
    
    @property
    def SEQproperties(self) -> dict:
        return{
            nmspc.etl:[int(self.etl)]
        }
        
    @property
    def OTHproperties(self) -> dict:
        return{
            nmspc.shimming:[list(self.shimming)],
        }

class HASTE:
    def __init__(self, 
                 seq:str='HASTE', 
                 nScans:int=1, 
                 larmorFreq: float=3.08,            # MHz 
                 rfExAmp: float=0.058,              # a.u. 
                 rfReAmp: float=0.116,            # a.u. 
                 rfExTime:int=170,                  # us 
                 rfReTime:int=170,                  # us
                 rfEnvelope:str='Rec',              # 'Rec' or 'Sinc'
                 echoSpacing:float=10,              # ms 
                 preExTime:float=0,                 # ms 
                 inversionTime:float=0,             # ms
                 repetitionTime:int = 500,          # ms
                 fov:list=[120., 120., 20.],         # mm 
                 dfov:list=[0., 0., 0.],                 # mm
                 nPoints:list=[60, 60, 1],          # points 
                 acqTime:int=4,                     # ms 
                 axes:list=[0, 1, 2],               # [rd, ph, sl], 0->x, 1->y, 2->z 
                 axesEnable:list=[1, 1, 1],         # [rd, ph, sl], 0->Off, 1->On 
                 sweepMode:int=1,                   # 0->k2k, 1->02k, 2->k20 
                 rdGradTime:int=5,                  # ms
                 rdDephTime:int=1,                  # ms
                 phGradTime:int=1,                  # ms
                 rdPreemphasis:float=1,             # readout dephasing grad is multiplied by this number
                 ssPreemphasis:float=1,             # slice rephasing grad is multiplied by this number
                 crusherDelay:float=0,              # us
                 drfPhase:float = 0,                # degrees, excitation pulse phase 
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
    def  systemproperties(self) -> dict:
        # TODO: add server cmd's as third entry in list
        return {
            nmspc.nScans:[int(self.nScans)], 
            nmspc.larmorFreq:[float(self.larmorFreq)], 
            nmspc.rfExAmp:[float(self.rfExAmp)], 
            nmspc.rfReAmp:[float(self.rfReAmp)], 
            nmspc.rfExTime:[int(self.rfExTime)], 
            nmspc.rfReTime:[int(self.rfReTime)],
            nmspc.rfEnvelope:[str(self.rfEnvelope)],
            nmspc.echoSpacing:[float(self.echoSpacing)], 
            nmspc.repetitionTime:[int(self.repetitionTime)], 
            nmspc.nPoints:[list(self.nPoints)], 
            nmspc.acqTime:[int(self.acqTime)], 
            nmspc.axes:[list(self.axes)], 
            nmspc.axesEnable:[list(self.axesEnable)], 
            nmspc.preExTime:[float(self.preExTime)], 
            nmspc.inversionTime:[float(self.inversionTime)], 
            nmspc.fov:[list(self.fov)], 
            nmspc.dfov:[list(self.dfov)], 
            nmspc.sweepMode:[int(self.sweepMode)], 
            nmspc.rdGradTime:[int(self.rdGradTime)], 
            nmspc.rdDephTime:[int(self.rdDephTime)], 
            nmspc.phGradTime:[int(self.phGradTime)], 
            nmspc.rdPreemphasis:[float(self.rdPreemphasis)], 
            nmspc.ssPreemphasis:[float(self.ssPreemphasis)],
            nmspc.crusherDelay:[float(self.crusherDelay)],
            nmspc.dummyPulses:[int(self.dummyPulses)], 
            nmspc.shimming:[list(self.shimming)], 
            nmspc.parFourierFraction:[float(self.parFourierFraction)], 
            nmspc.drfPhase:[int(self.drfPhase)], 
        }



"""
Definition of default sequences
"""
defaultsequences={
    'RARE': RARE(),
    'HASTE': HASTE()
    }



