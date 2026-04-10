"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QListWidget, QSizePolicy


class OutputListWidget(QListWidget):
    """Scrollable list widget that displays the output fields of the last run sequence."""
    def __init__(self, parent, *args, **kwargs):
        super(OutputListWidget, self).__init__(*args, **kwargs)
        self.main = parent
        self.setMaximumHeight(200)
