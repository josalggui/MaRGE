import seq.rabiFlops as rabiFlops
import seq.noise as noise
import seq.inversionRecovery as inversionRecovery


class RabiFlops(rabiFlops.RabiFlops):
    def __init__(self): super(RabiFlops, self).__init__()

class Noise(noise.Noise):
    def __init__(self): super(Noise, self).__init__()

class InversionRecovery(inversionRecovery.InversionRecovery):
    def __init__(self): super(InversionRecovery, self).__init__()

"""
Definition of default sequences
"""
defaultCalibFunctions={
    'RabiFlops': RabiFlops(),
    'Noise': Noise(),
    'InversionRecovery': InversionRecovery(),
}