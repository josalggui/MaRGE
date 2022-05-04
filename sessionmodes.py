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

      
class Subject:
    def __init__(self, 
                 name_code:str,
                 age:str=None, 
                 sex:str=None, 
                 weight:str=None,
                 ):
    
        self.name_code:str=name_code
        self.age:str=age
        self.sex: str=sex  
        self.weight:str=weight

    @property
    def demographics(self) -> dict:
        return{
            nmspc.name_code:[str(self.name_code)], 
            nmspc.age: [str(self.age)],
            nmspc.sex:[str(self.sex)] , 
            nmspc.weight:[str(self.weight)]
        }
        
class Fruit:
    def __init__(self, 
                 name_code:str,
                 weight:str=None 
                 ):
    
        self.name_code:str=name_code
        self.weight:str=weight     

    @property
    def demographics(self) -> dict:
        return{
            nmspc.name_code:[str(self.name_code)], 
            nmspc.weight: [str(self.weight)]
        }

class Test:
    def __init__(self, 
                 name_code:str,
                 descrip:str
                 ):
    
        self.name_code:str=name_code
        self.descrip:str=descrip

    @property
    def demographics(self) -> dict:
        return{
            nmspc.name_code:[str(self.name_code)], 
            nmspc.descrip:[str(self.descrip)]
        }        
        
        
"""
Definition of default sequences
"""
defaultsessions={

    #SpinEchoSeq(lo_freq,rf_amp,rf_pi2_duration,TE,TR,BW,nScans,shimming(rd,ph,sl), trap_ramp_duration,phase_grad_duration,n(x,y,z),fov(rd,ph,sl),preemph_factor)
#    'Subject':Subject('Subj ID',42, 'F', 60),
    'Subject':Subject('Subj_ID','Age', 'Sex', 'Weight'),  
    'Fruit':Fruit('Fruit\'s name', 'Weigth'), 
    'Test':Test('Test_ID', '')
}
