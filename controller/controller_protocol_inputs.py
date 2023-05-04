"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidgetItem, QMenu, QAction

from seq.sequences import defaultsequences
from widgets.widget_protocol_inputs import ProtocolInputsWidget

import configs.hw_config as hw


class ProtocolInputsController(ProtocolInputsWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clicked_item = None
        self.updateProtocolInputs()

        # Connect items to runToList method
        self.itemDoubleClicked.connect(self.sequenceDoubleClicked)
        self.itemClicked.connect(self.showSequenceInputs)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)

    def showContextMenu(self, point):
        self.clicked_item = self.itemAt(point)
        if self.clicked_item is not None:
            menu = QMenu(self)
            action1 = QAction("Delete", self)
            action1.triggered.connect(self.deleteSequence)
            menu.addAction(action1)

            menu.exec_(self.mapToGlobal(point))

    def deleteSequence(self):
        protocol = self.main.protocol_list.getCurrentProtocol()
        file = "%s.csv" % self.clicked_item.text()
        path = "protocols/%s/%s" % (protocol, file)
        os.remove(path)
        self.updateProtocolInputs()
        print("\nProtocol removed")

    def sequenceDoubleClicked(self, item):
        protocol = self.main.protocol_list.getCurrentProtocol()
        sequence = item.text()
        file = sequence + ".csv"
        seq_name = sequence.split('_')[0]

        # Load parameters
        defaultsequences[seq_name].loadParams("protocols/"+protocol, file)

        # Set larmor frequency, fov and dfov to the value into the hw_config file
        defaultsequences[seq_name].mapVals['larmorFreq'] = hw.larmorFreq
        defaultsequences[seq_name].mapVals['fov'] = hw.fov
        defaultsequences[seq_name].mapVals['dfov'] = hw.dfov

        # Run the sequence
        self.main.toolbar_sequences.runToList(seq_name=seq_name, item_name=sequence)

    def updateProtocolInputs(self):
        self.clear()

        # Get predefined sequences for each protocol
        self.sequences = {}
        for protocol in self.main.protocol_list.protocols:
            prov = []
            for path in os.listdir(os.path.join("protocols", protocol)):
                if path.split('.')[-1] == 'csv':
                    prov.append(path.split('.')[0])
            self.sequences[protocol] = prov.copy()

        # Add the predefined sequences of the first protocol to the protocol_input list
        protocol = self.main.protocol_list.getCurrentProtocol()
        if len(self.sequences) > 0:
            self.addItems(self.sequences[protocol])

    def showSequenceInputs(self, item):
        # Get file name
        file_name = "%s.csv" % item.text()
        seq_name = file_name.split('_')[0]

        # Extract items from the input_data
        input_info = defaultsequences[seq_name].mapNmspc.values()
        input_vals = defaultsequences[seq_name].mapVals.values()

        # Set number of rows
        self.main.input_table.setColumnCount(1)
        self.main.input_table.setRowCount(len(input_info))

        # Input items into the table
        self.main.input_table.setVerticalHeaderLabels(input_info)
        self.main.input_table.setHorizontalHeaderLabels(['Values'])
        for m, item in enumerate(input_vals):
            new_item = QTableWidgetItem(str(item))
            self.main.input_table.setItem(m, 0, new_item)
