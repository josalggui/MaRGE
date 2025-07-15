"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QListWidget, QSizePolicy


class OutputListWidget(QListWidget):
    def __init__(self, parent, *args, **kwargs):
        super(OutputListWidget, self).__init__(*args, **kwargs)
        self.main = parent
        self.setMaximumHeight(200)
