"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import os

from PyQt5.QtWidgets import QListWidget


class ProtocolInputsWidget(QListWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main

