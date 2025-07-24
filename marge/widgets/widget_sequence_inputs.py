"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QTabWidget, QSizePolicy


class SequenceInputsWidget(QTabWidget):
    def __init__(self, parent, *args, **kwargs):
        super(SequenceInputsWidget, self).__init__(*args, **kwargs)
        self.main = parent
        self.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        self.setMaximumWidth(400)
