"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: All sequences on the GUI must be here
"""

import seq.rare as rare
import seq.haste as haste
import seq.gre3d as gre

class RARE(rare.RARE):
    def __init__(self): super(RARE, self).__init__()

class GRE3D(gre.GRE3D):
    def __init__(self): super(GRE3D, self).__init__()

class HASTE(haste.HASTE):
    def __init__(self): super(HASTE, self).__init__()

"""
Definition of default sequences
"""
defaultsequences={
    'RARE': RARE(),
    'GRE3D': GRE3D(),
    'HASTE': HASTE(),
}