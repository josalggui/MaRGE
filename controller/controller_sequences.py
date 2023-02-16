"""
session_controller.py
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from toolbars.toolbar_sequences import SequenceToolBar


class SequenceController(SequenceToolBar):
    def __init__(self, *args, **kwargs):
        super(SequenceController, self).__init__(*args, **kwargs)

        # Set the action_iterate button checkable
        self.iterative_run = None
        self.action_iterate.setCheckable(True)

        # Connect the action buttons to the slots
        self.action_autocalibration.triggered.connect(self.autocalibration)
        self.action_acquire.triggered.connect(self.startAcquisition)
        self.action_add_to_list.triggered.connect(self.runToList)
        self.action_view_sequence.triggered.connect(self.startSequencePlot)
        self.action_localizer.triggered.connect(self.startLocalizer)
        self.action_iterate.triggered.connect(self.iterate)

    def autocalibration(self):
        pass

    def startAcquisition(self):
        pass

    def runToList(self):
        pass

    def startSequencePlot(self):
        pass

    def startLocalizer(self):
        pass

    def iterate(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: swtich the iterative mode
        """
        if self.action_iterate.isChecked():
            self.action_iterate.setToolTip('Switch to single run')
            self.action_iterate.setStatusTip("Switch to single run")
            self.iterative_run = True
        else:
            self.action_iterate.setToolTip('Switch to iterative run')
            self.action_iterate.setStatusTip(("Switch to iterative run"))
            self.iterative_run = False
