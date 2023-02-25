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

        # Get predefined sequences for each protocol
        self.sequences = {}
        for protocol in self.main.protocol_list.protocols:
            prov = []
            for path in os.listdir(os.path.join("protocols", protocol)):
                if os.path.isfile(os.path.join("protocols", path)):
                    prov.append(path)
            self.sequences[protocol] = prov.copy()
