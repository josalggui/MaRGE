"""
Global Variables and Objects

@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   1.0
@change:    02/05/2020

@summary:   Global variables

"""

class SqncObject:
    """
    Sequence object class
    """
    def __init__(self, name, path):
        self.str = name
        self.path = path


class Sequences:
    """
    Class with predefined sequences as sequence objects
    """
    SE = SqncObject('Spin Echo', 'seq/spinEcho.py')
    SE1D=SqncObject('Spin Echo 1D', 'seq/spinEcho1D.py')
    FID = SqncObject('Free Induction Decay', 'seq/fid.py')
    GE = SqncObject('Gradient Echo', 'seq/gradEcho.py')
    R = SqncObject('Radial', 'seq/radial.py')
    TSE = SqncObject('Turbo Spin Echo', 'seq/turboSpinEcho.py')


class Gradients:
    X = 0
    Y = 1
    Z = 2
    Z2 = 3


class Relaxations:
    T1 = 'T1'
    T2 = 'T2'


class ProjectionAxes:
    x = 0
    y = 1
    z = 2


class StyleSheets:
    breezeDark = "view/stylesheets/breeze-dark.qss"
    breezeLight = "view/stylesheets/breeze-light.qss"


# Instances
sqncs = Sequences()
grads = Gradients()
rlxs = Relaxations()
pax = ProjectionAxes()
