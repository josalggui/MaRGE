"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QToolBar, QAction
from PyQt5.QtGui import QIcon

class SequenceToolBar(QToolBar):
    def __init__(self, parent, *args, **kwargs):
        super(SequenceToolBar, self).__init__(*args, **kwargs)
        self.main = parent
        self.demo = self.main.demo

        # Autocalibration
        self.action_autocalibration = QAction(QIcon("resources/icons/calibration-light.png"), "Autocalibration", self)
        self.action_autocalibration.setStatusTip("Run autocalibration")
        self.addAction(self.action_autocalibration)

        # Localizer
        self.action_localizer = QAction(QIcon("resources/icons/localizer-light.png"), "Localizer", self)
        self.action_localizer.setStatusTip("Run Localizer")
        self.addAction(self.action_localizer)

        # Add sequence to waiting list
        self.action_add_to_list = QAction(QIcon("resources/icons/clipboard-list-check"), "Sequence to list", self)
        self.action_add_to_list.setStatusTip("Add current sequence to waiting list")
        self.addAction(self.action_add_to_list)

        # Add run action
        self.action_acquire = QAction(QIcon("resources/icons/acquire.png"), "Acquire", self)
        self.action_acquire.setStatusTip("Run current sequence")
        self.addAction(self.action_acquire)

        # Iterative run
        self.action_iterate = QAction(QIcon("resources/icons/iterate.png"), "Iterative run", self)
        self.action_iterate.setStatusTip("Set iterative mode on")
        self.addAction(self.action_iterate)

        # Bender button
        self.action_bender = QAction(QIcon("resources/icons/bender.png"), "Bender button", self)
        self.action_bender.setStatusTip("Bender will do a full protocol with a single click")
        self.addAction(self.action_bender)

        # Plot sequence
        self.action_view_sequence = QAction(QIcon("resources/icons/plotSequence.png"), "Plot sequence", self)
        self.action_view_sequence.setStatusTip("Plot current sequence")
        self.addAction(self.action_view_sequence)