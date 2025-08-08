"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import os

from marge.widgets.widget_protocol_list import ProtocolListWidget


class ProtocolListController(ProtocolListWidget):
    """
    Controller class for managing the protocol list.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Inherits:
        ProtocolListWidget: Base class for protocol list widget.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the ProtocolListController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(ProtocolListController, self).__init__(*args, **kwargs)
        self.protocol = None
        self.protocols = None

        if not os.path.exists('protocols'):
            os.makedirs('protocols')

        self.updateProtocolList()

        self.currentTextChanged.connect(self.updateProtocolInputs)

    def getCurrentProtocol(self):
        """
        Returns the currently selected protocol.

        Returns:
            str: The name of the current protocol.
        """
        return self.currentText()

    def updateProtocolInputs(self):
        """
        Updates the protocol inputs based on the selected protocol.
        """
        # Get the name of the selected sequence
        self.protocol = self.getCurrentProtocol()

        # Delete sequences from current protocol
        self.main.protocol_inputs.clear()

        # Add items corresponding to selected protocol
        self.main.protocol_inputs.addItems(self.main.protocol_inputs.sequences[self.protocol])

        return 0

    def updateProtocolList(self):
        """
        Updates the list of protocols.
        """
        self.blockSignals(True)
        self.clear()

        # Get the protocols
        self.protocols = []
        for path in os.listdir("protocols"):
            if len(path.split('.')) == 1:
                self.protocols.append(path)

        # Add protocols to list
        self.addItems(self.protocols)
        self.blockSignals(False)

        self.protocol = self.getCurrentProtocol()