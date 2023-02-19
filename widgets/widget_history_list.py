"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QListWidget


class HistoryListWidget(QListWidget):
    def __init__(self, parent, *args, **kwargs):
        super(HistoryListWidget, self).__init__(*args, **kwargs)
        self.main = parent
        self.setMaximumHeight(200)
