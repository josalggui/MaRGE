"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import os
import csv
import copy

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidgetItem, QMenu, QAction

from marge.seq.sequences import defaultsequences
from marge.widgets.widget_protocol_inputs import ProtocolInputsWidget

import marge.configs.hw_config as hw

import numpy as np


class ProtocolInputsController(ProtocolInputsWidget):
    """
    Controller class for managing protocol inputs.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Inherits:
        ProtocolInputsWidget: Base class for protocol inputs widget.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the ProtocolInputsController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.clicked_item = None
        self.updateProtocolInputs()

        # Connect items to runToList method
        self.itemDoubleClicked.connect(self.sequenceDoubleClicked)
        self.itemClicked.connect(self.showSequenceInputs)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)

    def showContextMenu(self, point):
        """
        Displays the context menu when right-clicked on an item.

        Args:
            point (QPoint): The position where the context menu was requested.
        """
        self.clicked_item = self.itemAt(point)
        if self.clicked_item is not None:
            menu = QMenu(self)

            # Show input action button
            action2 = QAction("Show inputs", self)
            action2.triggered.connect(self.showProtocolInputs)
            menu.addAction(action2)

            # Load inputs action button
            action3 = QAction("Load inputs", self)
            action3.triggered.connect(self.loadProtocolInputs)
            menu.addAction(action3)

            # Delete action button
            action1 = QAction("Delete", self)
            action1.triggered.connect(self.deleteSequence)
            menu.addAction(action1)

            menu.exec_(self.mapToGlobal(point))

    def deleteSequence(self):
        """
        Deletes a sequence from the protocol inputs.
        """
        protocol = self.main.protocol_list.getCurrentProtocol()
        file = "%s.csv" % self.clicked_item.text()
        path = "protocols/%s/%s" % (protocol, file)
        os.remove(path)
        self.updateProtocolInputs()
        print("Protocol removed")

    def sequenceDoubleClicked(self, item):
        """
        Handles double-click event on a sequence item.

        Args:
            item (QListWidgetItem): The item that was double-clicked.
        """
        protocol = self.main.protocol_list.getCurrentProtocol()
        protocol_item = item.text()
        file = protocol_item + ".csv"
        seq_name = protocol_item.split('_')[0]

        sequence = copy.deepcopy(defaultsequences[seq_name])

        # Pick the shimming from sequence, as it was set up by autocalibration
        try:
            shimming = copy.deepcopy(sequence.mapVals['shimming'])
        except:
            pass

        # Load parameters
        sequence.loadParams("protocols/"+protocol, file)

        # Set larmor frequency, fov and dfov to the value into the hw_config file
        sequence.mapVals['larmorFreq'] = hw.larmorFreq
        sequence.mapVals['fov'] = hw.fov
        sequence.mapVals['dfov'] = hw.dfov
        sequence.mapVals['shimming'] = shimming
        hw.dfov = [0.0, 0.0, 0.0]

        # Run the sequence
        map_nmspc = list(sequence.mapNmspc.values())
        map_vals = list(sequence.mapVals.values())
        self.main.toolbar_sequences.runToList(seq_name=seq_name, item_name=protocol_item, map_nmspc=map_nmspc,
                                              map_vals=map_vals)

    def updateProtocolInputs(self):
        """
        Updates the protocol inputs list.
        """
        self.clear()

        # Get predefined sequences for each protocol
        self.sequences = {}
        for protocol in self.main.protocol_list.protocols:
            # Construct the path
            path = os.path.join("protocols", protocol)

            # List all files in the specified directory
            files = [file for file in os.listdir(path) if file.endswith('.csv') and os.path.isfile(os.path.join(path, file))]

            # Sort the list of files based on their creation date
            files.sort(key=lambda x: os.path.getctime(os.path.join(path, x)))

            # Remove extensions
            files = [file.split('.')[0] for file in files]

            # Add files to the protocol variable
            self.sequences[protocol] = files.copy()

        # Add the predefined sequences of the first protocol to the protocol_input list
        protocol = self.main.protocol_list.getCurrentProtocol()
        if len(self.sequences) > 0:
            self.addItems(self.sequences[protocol])

    def showSequenceInputs(self, item):
        """
        Displays the inputs for a selected sequence.

        Args:
            item (QListWidgetItem): The selected item.
        """
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

    def showProtocolInputs(self):

        # Get path to the file
        protocol = self.main.protocol_list.getCurrentProtocol()
        file_name = "%s.csv" % self.clicked_item.text()
        path_to_file = "protocols/%s/%s" % (protocol, file_name)

        # Get strings for the info
        seq_name = file_name.split('_')[0]
        input_info = defaultsequences[seq_name].mapNmspc.values()

        # Get value for the info
        map_vals = {}
        with open(path_to_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for l in reader:
                map_vals = l
        input_vals = map_vals.values()

        # Set number of rows
        self.main.input_table.setColumnCount(1)
        self.main.input_table.setRowCount(len(input_info))

        # Input items into the table
        self.main.input_table.setVerticalHeaderLabels(input_info)
        self.main.input_table.setHorizontalHeaderLabels(['Values'])
        for m, item in enumerate(input_vals):
            new_item = QTableWidgetItem(str(item))
            self.main.input_table.setItem(m, 0, new_item)

    def loadProtocolInputs(self):
        # Get path to the file
        protocol = self.main.protocol_list.getCurrentProtocol()
        file_name = "%s.csv" % self.clicked_item.text()
        path_to_file = "protocols/%s/%s" % (protocol, file_name)

        # Get strings for the info
        seq_name = file_name.split('_')[0]
        seq = defaultsequences[seq_name]
        map_vals_old = seq.mapVals

        # Get value for the info
        map_vals_new = {}
        with open(path_to_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for l in reader:
                map_vals_new = l

        seq.mapVals = {}

        # Get key for corresponding modified parameter
        for key in seq.mapKeys:
            data_len = seq.mapLen[key]
            val_old = map_vals_old[key]
            try:
                val_new = map_vals_new[key]
            except:
                val_new = str(val_old)*1
            val_new = val_new.replace('[', '')
            val_new = val_new.replace(']', '')
            val_new = val_new.split(',')
            if type(val_old) == str:
                val_old = [val_old]
            elif data_len == 1:
                val_old = [val_old]
            data_type = type(val_old[0])

            inputNum = []
            for ii in range(data_len):
                if data_type == float or data_type == np.float64:
                    try:
                        inputNum.append(float(val_new[ii]))
                    except:
                        inputNum.append(float(val_old[ii]))
                elif data_type == int:
                    try:
                        inputNum.append(int(val_new[ii]))
                    except:
                        inputNum.append(int(val_old[ii]))
                else:
                    try:
                        inputNum.append(str(val_new[0]))
                        break
                    except:
                        inputNum.append(str(val_old[0]))
                        break
            if data_type == str:
                seq.mapVals[key] = inputNum[0]
            else:
                if data_len == 1:  # Save value into mapVals
                    seq.mapVals[key] = inputNum[0]
                else:
                    seq.mapVals[key] = inputNum

        self.main.sequence_list.updateSequence()
        print("Parameters of %s sequence loaded" % file_name)
