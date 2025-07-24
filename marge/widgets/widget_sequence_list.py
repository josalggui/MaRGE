"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QComboBox, QSizePolicy
from marge.seq.sequences import defaultsequences


class SequenceListWidget(QComboBox):
    def __init__(self, parent, *args, **kwargs):
        super(SequenceListWidget, self).__init__(*args, **kwargs)
        self.main = parent

        # Add sequences to sequences list
        self.addItems(sorted(list(defaultsequences.keys())))

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setMaximumWidth(400)
