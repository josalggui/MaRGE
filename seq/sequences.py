import seq.rare as rare
import seq.haste as haste
import seq.gre as gre

class RARE(rare.RARE):
    def __init__(self): x = 1

class HASTE(haste.HASTE):
    def __init__(self): x = 1

class GRE(gre.GRE):
    def __init__(self): x = 1

"""
Definition of default sequences
"""
defaultsequences={
    'RARE': RARE(),
    'HASTE': HASTE(),
    'GRE3D': GRE3D(),
    }