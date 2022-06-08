import seq.rabiFlopsNew as rabiFlops
import seq.noise as noise


class RabiFlops(rabiFlops.RabiFlops):
    def __init__(self): super(RabiFlops, self).__init__()

class Noise(noise.Noise):
    def __init__(self): super(Noise, self).__init__()
#
# class HASTE(haste.HASTE):
#     def __init__(self): super(HASTE, self).__init__()

"""
Definition of default sequences
"""
defaultCalibFunctions={
    'RabiFlops': RabiFlops(),
    'Noise': Noise(),
    # 'HASTE': HASTE(),
}