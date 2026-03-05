"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QComboBox, QSizePolicy
from marge.seq.sequences import defaultsequences, sequence_display_names


class SequenceListWidget(QComboBox):
    def __init__(self, parent, *args, **kwargs):
        super(SequenceListWidget, self).__init__(*args, **kwargs)
        self.main = parent

        # Show folder-prefixed labels in the UI, keep raw seq key as item data.
        keys = sorted(defaultsequences.keys(), key=lambda key: sequence_display_names.get(key, key).lower())
        for key in keys:
            display_name = sequence_display_names.get(key, key)
            self.addItem(display_name, key)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setMaximumWidth(400)
