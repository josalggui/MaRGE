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
                 seq:str, 
                 nScans:int=None, 
                 
                 larmorFreq: float=None, 
                 rfExAmp: float=None, 
                 rfReAmp: float=None, 
                 rfExTime:int=None, 
                 rfReTime:int=None, 
                 
                 echoSpacing:float=None, 
                 preExTime:float=None, 
                 inversionTime:float=None, 
                 repetitionTime:int = None, 
                 fov:list=None, 
                 dfov:list=None,
                 nPoints:list=None, 
                 etl:int=None, 
                 acqTime:int=None, 
                 
                 axes:list=None, 
                 axesEnable:list=None, 
                 sweepMode:int=None, 
                
                 rdGradTime:int=None, 
                 rdDephTime:int=None,
                 phGradTime:int=None,
                 rdPreemphasis:float = None,
                 drfPhase:int = None, 
                 dummyPulses:int = None, 
                 
                 shimming:list=None, 
                 parAcqLines:int = None, 
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
        self.parAcqLines:int = parAcqLines

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
            nmspc.echoSpacing:[float(self.echoSpacing)], 
            nmspc.etl:[int(self.etl)], 
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
            nmspc.dummyPulses:[int(self.dummyPulses)], 
            nmspc.shimming:[list(self.shimming)], 
            nmspc.parAcqLines:[int(self.parAcqLines)], 
            nmspc.drfPhase:[int(self.drfPhase)], 
        }

"""
Definition of default sequences
"""
defaultsequences={
#    'RARE': RARE(seq,nScans, larmorFreq, rfExAmp, rfReAmp, rfExTime, rfReTime, echoSpacing, preExTime, inversionTime, repetitionTime, fov, dfov, nPoints, etl, ...
#                            acqTime, axes, axesEnable, sweepMode, rdGradTime, rdDephTime, phGradTime, rdPreemphasis, drfPhase, dummyPulses,shimming, parAcqLines)
    'RARE': RARE('RARE', 1, 3.076, 0.3, 0, 35, 0, 10., 0., 0.,500., (120, 120, 120),  (0, 0, 0), (60, 1, 1), 30, 4, (0, 1, 2), (0, 0, 0), 1, 6, 1, 1, 1.005, 0, 1, (-70, -90, 10), 0) 
    }
    
#class CPMGSeq:
#    
#   
#    def __init__(self, 
#                 seq:str, 
#                 larmorFreq: float=None, 
#                 rfExAmp: float=None, 
#                 rfReAmp: float=None, 
#                 rfExTime: int=None, 
#                 rfReTime:int=None, 
#                 echoSpacing:int=None, 
#                 nPoints:int=None, 
#                 etl:int=None, 
#                 acqTime: int=None,             
#                 ):
#    
#        self.seq:str=seq
#        self.larmorFreq:float=larmorFreq
#        self.rfExAmp: float=rfExAmp
#        self.rfReAmp:float=rfReAmp
#        self.rfExTime:float=rfExTime
#        self.rfReTime:float=rfReTime
#        self.echoSpacing:float=echoSpacing
#        self.nPoints:int=nPoints
#        self.etl:int=etl
#        self.acqTime:float=acqTime
#       
#    @property
#    def  systemproperties(self) -> dict:
#        # TODO: add server cmd's as third entry in list
#        return {
#            nmspc.larmorFreq: [float(self.larmorFreq)], 
#            nmspc.rfExAmp:[float(self.rfExAmp)], 
#            nmspc.rfReAmp:[float(self.rfReAmp)], 
#            nmspc.rfExTime:[float(self.rfExTime)], 
#            nmspc.rfReTime:[float(self.rfReTime)], 
#            nmspc.echoSpacing:[float(self.echoSpacing)], 
#            nmspc.nPoints:[int(self.nPoints)], 
#            nmspc.etl:[int(self.etl)],
#            nmspc.acqTime: [float(self.acqTime)],
#        }
#




