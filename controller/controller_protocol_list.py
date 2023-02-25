"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import os

from widgets.widget_protocol_list import ProtocolListWidget


class ProtocolListController(ProtocolListWidget):
    def __init__(self, *args, **kwargs):
        super(ProtocolListController, self).__init__(*args, **kwargs)
        self.protocol = None
        self.protocols = None

        if not os.path.exists('protocols'):
            os.makedirs('protocols')

        self.updateProtocolList()

        self.currentTextChanged.connect(self.updateProtocolInputs)

    def getCurrentProtocol(self):
        return self.currentText()

    def updateProtocolInputs(self):
        # Get the name of the selected sequence
        self.protocol = self.getCurrentProtocol()

        # Delete sequences from current protocol
        self.main.protocol_inputs.clear()

        # Add items corresponding to selected protocol
        self.main.protocol_inputs.addItems(self.main.protocol_inputs.sequences[self.protocol])

    def updateProtocolList(self):
        self.blockSignals(True)
        self.clear()
        self.blockSignals(False)

        # Get the protocols
        self.protocols = []
        for path in os.listdir("protocols"):
            if len(path.split('.')) == 1:
                self.protocols.append(path)

        # Add protocols to list
        self.addItems(self.protocols)

        self.protocol = self.getCurrentProtocol()
