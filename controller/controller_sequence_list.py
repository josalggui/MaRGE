"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from widgets.widget_combobox_sequence_list import SequenceListWidget
from seq.sequences import defaultsequences


class SequenceListController(SequenceListWidget):
    def __init__(self, *args, **kwargs):
        super(SequenceListController, self).__init__(*args, **kwargs)
        self.seq_name = self.getCurrentSequence()
        self.currentTextChanged.connect(self.updateSequence)
        self.currentTextChanged.connect(self.showSequenceInfo)

    def getCurrentSequence(self):
        return self.currentText()

    def updateSequence(self):
        # Get the name of the selected sequence
        self.seq_name = self.getCurrentSequence()

        # Reset the tabs with corresponding input parameters
        if hasattr(self.main, "sequence_inputs"):
            self.main.sequence_inputs.removeTabs()
            self.main.sequence_inputs.displayInputParameters()

    def showSequenceInfo(self):
        # Get the name of the selected sequence
        self.seq_name = self.getCurrentSequence()

        defaultsequences[self.seq_name].sequenceInfo()
