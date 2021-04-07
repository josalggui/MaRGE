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
    FID = SqncObject('Free Induction Decay', 'sequence/FID.txt')
    SE = SqncObject('Spin Echo', 'sequence/SE_te.txt')
    IR = SqncObject('Inversion Recovery', 'sequence/IR_ti.txt')
    SIR = SqncObject('Saturation Inversion Recovery', 'sequence/SIR_ti.txt')
    imgSE = SqncObject('Spin Echo for Imaging', 'sequence/img/2DSE.txt')


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
    breezeDark = "stylesheets/breeze-dark.qss"
    breezeLight = "stylesheets/breeze-light.qss"


# Instances
sqncs = Sequences()
grads = Gradients()
rlxs = Relaxations()
pax = ProjectionAxes()
