import seq.rabiFlops as rabiFlops
import seq.noise as noise
import seq.inversionRecovery as inversionRecovery
import seq.cpmg as cpmg
import seq.larmor as larmor
import seq.shimmingSweep as shimmingSweep
import seq.fov as fov
import seq.sliceSelection as sliceSelection


class RabiFlops(rabiFlops.RabiFlops):
    def __init__(self): super(RabiFlops, self).__init__()

class Noise(noise.Noise):
    def __init__(self): super(Noise, self).__init__()

class InversionRecovery(inversionRecovery.InversionRecovery):
    def __init__(self): super(InversionRecovery, self).__init__()

class CPMG(cpmg.CPMG):
    def __init__(self): super(CPMG, self).__init__()

class Larmor(larmor.Larmor):
    def __init__(self): super(Larmor, self).__init__()

class ShimmingSweep(shimmingSweep.ShimmingSweep):
    def __init__(self): super(ShimmingSweep, self).__init__()

class FOV(fov.FOV):
    def __init__(self): super(FOV, self).__init__()

class SliceSelection(sliceSelection.SliceSelection):
    def __init__(self): super(SliceSelection, self).__init__()


"""
Definition of default sequences
"""
defaultCalibFunctions = {
    'RabiFlops': RabiFlops(),
    'Noise': Noise(),
    'InversionRecovery': InversionRecovery(),
    'CPMG': CPMG(),
    'Larmor': Larmor(),
    'ShimmingSweep': ShimmingSweep(),
    'fov': FOV(),
    'SliceSelection': SliceSelection(),
}