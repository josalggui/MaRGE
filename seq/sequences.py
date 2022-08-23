"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: All sequences on the GUI must be here
"""

import seq.rare as rare
import seq.haste as haste
import seq.gre3d as gre
import seq.petra as petra
import seq.fid as fid
import seq.rabiFlops as rabiFlops
import seq.cpmg as cpmg
import seq.larmor as larmor
import seq.inversionRecovery as inversionRecovery
import seq.noise as noise
import seq.shimmingSweep as shimming
import seq.sliceSelection as sliceSelection
import seq.fov as fov
import seq.sweepImage as sweep

class RARE(rare.RARE):
    def __init__(self): super(RARE, self).__init__()

class GRE3D(gre.GRE3D):
    def __init__(self): super(GRE3D, self).__init__()

class PETRA(petra.PETRA):
    def __init__(self): super(PETRA, self).__init__()

class HASTE(haste.HASTE):
    def __init__(self): super(HASTE, self).__init__()

class FID(fid.FID):
    def __init__(self): super(FID, self).__init__()

class RabiFlops(rabiFlops.RabiFlops):
    def __init__(self): super(RabiFlops, self).__init__()

class Larmor(larmor.Larmor):
    def __init__(self): super(Larmor, self).__init__()

class Noise(noise.Noise):
    def __init__(self): super(Noise, self).__init__()

class CPMG(cpmg.CPMG):
    def __init__(self): super(CPMG, self).__init__()

class IR(inversionRecovery.InversionRecovery):
    def __init__(self): super(IR, self).__init__()

class Shimming(shimming.ShimmingSweep):
    def __init__(self): super(Shimming, self).__init__()

class SliceSelection(sliceSelection.SliceSelection):
    def __init__(self): super(SliceSelection, self).__init__()

class FOV(fov.FOV):
    def __init__(self): super(FOV, self).__init__()

class SWEEP(sweep.SweepImage):
    def __init__(self): super(SWEEP, self).__init__()

"""
Definition of default sequences
"""
defaultsequences = {
    'RARE': RARE(),
    'GRE3D': GRE3D(),
    'PETRA': PETRA(),
    'HASTE': HASTE(),
    'FID': FID(),
    'Larmor': Larmor(),
    'Noise': Noise(),
    'RabiFlops': RabiFlops(),
    'CPMG': CPMG(),
    'InversionRecovery': IR(),
    'Shimming': Shimming(),
    'SliceSelection': SliceSelection(),
    'FOV': FOV(),
    'SWEEP': SWEEP(),
}