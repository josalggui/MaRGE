"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
from marge.widgets.widget_sequence_list import SequenceListWidget
from marge.seq.sequences import defaultsequences


class SequenceListController(SequenceListWidget):
    """
    Controller class for managing the sequence list.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Inherits:
        SequenceListWidget: Base class for sequence list widget.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the SequenceListController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(SequenceListController, self).__init__(*args, **kwargs)
        self.seq_name = self.getCurrentSequence()
        self.currentTextChanged.connect(self.updateSequence)
        self.currentTextChanged.connect(self.showSequenceInfo)

        # Here the GUI updates the inputs to the last used inputs
        for key in defaultsequences.keys():
            defaultsequences[key].loadParams()

    def getCurrentSequence(self):
        """
        Returns the name of the currently selected sequence.

        Returns:
            str: The name of the current sequence.
        """
        return self.currentText()

    def updateSequence(self):
        """
        Updates the selected sequence and resets the corresponding input parameters.
        """
        # Get the name of the selected sequence
        self.seq_name = self.getCurrentSequence()

        # Reset the tabs with corresponding input parameters
        if hasattr(self.main, "sequence_inputs"):
            self.main.sequence_inputs.removeTabs()
            self.main.sequence_inputs.displayInputParameters()

    def showSequenceInfo(self):
        """
        Displays information about the selected sequence.
        """
        # Get the name of the selected sequence
        self.seq_name = self.getCurrentSequence()

        defaultsequences[self.seq_name].sequenceInfo()
