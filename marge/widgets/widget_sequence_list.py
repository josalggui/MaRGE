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

        # Keep root sequences first and append folder-prefixed ones afterwards.
        ordered_keys = sorted(
            defaultsequences.keys(),
            key=lambda key: ("/" in sequence_display_names.get(key, key), sequence_display_names.get(key, key).lower())
        )
        for key in ordered_keys:
            self.addItem(sequence_display_names.get(key, key), key)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setMaximumWidth(400)
