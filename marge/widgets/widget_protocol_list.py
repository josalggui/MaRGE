"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QComboBox


class ProtocolListWidget(QComboBox):
    def __init__(self, main, *args, **kwargs):
        super(ProtocolListWidget, self).__init__(*args, **kwargs)
        self.main = main
