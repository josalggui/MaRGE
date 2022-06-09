import seq.rabiFlops as rabiFlops
import seq.noise as noise
import seq.inversionRecovery as inversionRecovery
import seq.cpmg as cpmg


class RabiFlops(rabiFlops.RabiFlops):
    def __init__(self): super(RabiFlops, self).__init__()

class Noise(noise.Noise):
    def __init__(self): super(Noise, self).__init__()

class InversionRecovery(inversionRecovery.InversionRecovery):
    def __init__(self): super(InversionRecovery, self).__init__()

class CPMG(cpmg.CPMG):
    def __init__(self): super(CPMG, self).__init__()

"""
Definition of default sequences
"""
defaultCalibFunctions = {
    'RabiFlops': RabiFlops(),
    'Noise': Noise(),
    'InversionRecovery': InversionRecovery(),
    'CPMG': CPMG(),
}