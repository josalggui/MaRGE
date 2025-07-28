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

        toLittle = self.main.session.get("little_version", False)

        # Add sequences to the combo box depending on the mode
        if toLittle:
            # Add only sequences with toLittle == True
            for name, sequence in defaultsequences.items():
                if sequence.getParameter('toLittle') == True:
                    self.addItem(name)
        else:
            # Add all sequences
            self.addItems(sorted(list(defaultsequences.keys())))

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setMaximumWidth(400)
