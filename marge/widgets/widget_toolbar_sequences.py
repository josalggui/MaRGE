"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QToolBar, QAction
from PyQt5.QtGui import QIcon
from importlib import resources

class SequenceToolBar(QToolBar):
    def __init__(self, parent, *args, **kwargs):
        super(SequenceToolBar, self).__init__(*args, **kwargs)
        self.main = parent

        # Autocalibration
        with resources.path("marge.resources.icons", "calibration-light.png") as path_autocal:
            self.action_autocalibration = QAction(QIcon(str(path_autocal)), "Autocalibration", self)
        self.action_autocalibration.setStatusTip("Run autocalibration")
        self.addAction(self.action_autocalibration)

        # Localizer
        with resources.path("marge.resources.icons", "localizer-light.png") as path_localizer:
            self.action_localizer = QAction(QIcon(str(path_localizer)), "Localizer", self)
        self.action_localizer.setStatusTip("Run Localizer")
        self.addAction(self.action_localizer)

        # Add sequence to waiting list
        with resources.path("marge.resources.icons", "clipboard-list-check") as path_add_list:
            self.action_add_to_list = QAction(QIcon(str(path_add_list)), "Sequence to list", self)
        self.action_add_to_list.setStatusTip("Add current sequence to waiting list")
        self.addAction(self.action_add_to_list)

        # Add run action
        with resources.path("marge.resources.icons", "acquire.png") as path_acquire:
            self.action_acquire = QAction(QIcon(str(path_acquire)), "Acquire", self)
        self.action_acquire.setStatusTip("Run current sequence")
        self.addAction(self.action_acquire)

        # Iterative run
        with resources.path("marge.resources.icons", "iterate.png") as path_iterate:
            self.action_iterate = QAction(QIcon(str(path_iterate)), "Iterative run", self)
        self.action_iterate.setStatusTip("Set iterative mode on")
        self.addAction(self.action_iterate)

        # Bender button
        with resources.path("marge.resources.icons", "bender.png") as path_bender:
            self.action_bender = QAction(QIcon(str(path_bender)), "Bender button", self)
        self.action_bender.setStatusTip("Bender will do a full protocol with a single click")
        self.addAction(self.action_bender)

        # Plot sequence
        with resources.path("marge.resources.icons", "plotSequence.png") as path_plot:
            self.action_view_sequence = QAction(QIcon(str(path_plot)), "Plot sequence", self)
        self.action_view_sequence.setStatusTip("Plot current sequence")
        self.addAction(self.action_view_sequence)

        # Save sequence parameters
        with resources.path("marge.resources.icons", "saveParameters.png") as path_save:
            self.action_save_parameters = QAction(QIcon(str(path_save)), "Save parameters", self)
        self.action_save_parameters.setStatusTip("Save the current parameters")
        self.addAction(self.action_save_parameters)

        # Load sequence parameters
        with resources.path("marge.resources.icons", "loadParameters.png") as path_load:
            self.action_load_parameters = QAction(QIcon(str(path_load)), "Load parameters", self)
        self.action_load_parameters.setStatusTip("Load parameters and update current sequence to loaded values")
        self.addAction(self.action_load_parameters)

        # Save sequence parameters for calibration
        with resources.path("marge.resources.icons", "favouriteParameters.png") as path_cal:
            self.action_save_parameters_cal = QAction(QIcon(str(path_cal)), "Save for calibration", self)
        self.action_save_parameters_cal.setStatusTip("Save current configuration for calibration")
        self.addAction(self.action_save_parameters_cal)
