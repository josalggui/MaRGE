"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import csv
import os
import platform
import shutil

from PyQt5.QtWidgets import QFileDialog

from marge.seq.sequences import defaultsequences
from marge.widgets.widget_toolbar_protocols import ProtocolsToolBar


class ProtocolsController(ProtocolsToolBar):
    """
    Controller class for managing protocols in the application.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Inherits:
        ProtocolsToolBar: Base class for the protocols toolbar.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the ProtocolsController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.action_new_protocol.triggered.connect(self.newProtocol)
        self.action_del_protocol.triggered.connect(self.delProtocol)
        self.action_new_sequence.triggered.connect(self.newSequence)
        self.action_del_sequence.triggered.connect(self.delSequence)

    def delProtocol(self):
        """
        Deletes a protocol.

        Opens a file dialog to select a protocol directory for removal.
        """
        # Open a file dialog to get the filename to save to
        directory = 'protocols'
        folder_name = QFileDialog.getExistingDirectory(self.main, "Remove protocol", directory)

        if folder_name:
            shutil.rmtree(folder_name)
            print(f"Protocol {folder_name} removed")
            self.main.protocol_list.updateProtocolList()

    def delSequence(self):
        """
        Deletes a sequence from a protocol.

        Opens a file dialog to select a sequence file for removal from the current protocol.
        """
        # Get the current protocol
        protocol = self.main.protocol_list.getCurrentProtocol()

        # Open a file dialog to get the filename to save to
        directory = 'protocols/%s' % protocol
        file_name, _ = QFileDialog.getOpenFileName(None, 'Remove sequence from protocol', directory, options = QFileDialog.Options())

        # Delete protocol
        if file_name:
            os.remove(file_name)
            print(f"Sequence {file_name} removed from protocol {protocol}")
            self.main.protocol_inputs.updateProtocolInputs()

    def newProtocol(self):
        """
        Creates a new protocol.

        Opens a file dialog to specify the name and location for the new protocol.
        """
        # Open a file dialog to get the filename to save to
        file_name, _ = QFileDialog.getSaveFileName(self.main, 'New Protocol', 'protocols', '')

        if file_name:
            # Delete extension
            file_name = file_name.split('.')[0]

            # Check if the folder is the good one
            directory = os.path.dirname(file_name).split('/')[-1]
            protocol = file_name.split('/')[-1]
            if directory != 'protocols':
                print("Error. New protocols should be in 'protocols' folder.")
                return

            if not os.path.exists(file_name):
                os.makedirs(file_name)
                print("New protocol created successfully")
                self.main.protocol_inputs.sequences[protocol] = []
                self.main.protocol_list.updateProtocolList()
                self.main.protocol_inputs.updateProtocolInputs()
            else:
                print("Protocol already exist")

    def newSequence(self):
        """
        Adds a new sequence to a protocol.

        Gets the current protocol and sequence, opens a file dialog to specify the name and location
        for the new sequence file within the current protocol directory, and saves the sequence as a CSV file.
        """
        # Get the current protocol
        protocol = self.main.protocol_list.getCurrentProtocol()

        # Get the current sequence
        seq_name = self.main.sequence_list.getCurrentSequence()
        sequence = defaultsequences[seq_name]

        # Open a file dialog to get the filename to save to
        directory = 'protocols/%s' % protocol
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            None, "Add sequence to protocol", directory, "",
            options=options
        )

        if file_name:
            if platform.system()=='Linux':
                file_name = "%s_%s.csv" % (seq_name, file_name.split('/')[-1])
            else:
                file_name = "%s_%s" % (seq_name, file_name.split('/')[-1])

            # Save csv with input parameters
            with open('%s/%s' % (directory, file_name), 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=sequence.mapKeys)
                writer.writeheader()
                map_vals = {}
                for key in sequence.mapKeys:  # take only the inputs from mapVals
                    map_vals[key] = sequence.mapVals[key]
                writer.writerows([sequence.mapNmspc, map_vals])

            self.main.protocol_inputs.updateProtocolInputs()

            print("%s sequence added to the %s protocol" % (file_name.split('.')[0], protocol))
