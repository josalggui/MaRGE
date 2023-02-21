"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QComboBox, QSizePolicy
from seq.sequences import defaultsequences


class SequenceListWidget(QComboBox):
    def __init__(self, parent, *args, **kwargs):
        super(SequenceListWidget, self).__init__(*args, **kwargs)
        self.main = parent

        # Add sequences to sequences list
        self.addItems(list(defaultsequences.keys()))

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setMaximumWidth(400)


        # if hasattr(parent, 'onSequenceUpdate'):
        #     parent.onSequenceUpdate.connect(self.sequenceUpdate)
        #
        # if hasattr(parent, 'onSequenceChanged'):
        #     parent.onSequenceChanged.connect(self.triggeredSequenceChanged)
        #     # Make parent reachable from outside __init__
        #     self.parent = parent
        #     self._currentSequence = "RARE"
        #     self.setParametersUI(self._currentSequence)
        # else:
        #     self._currentSequence=parent.sequence
