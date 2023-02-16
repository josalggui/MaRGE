"""
session_controller.py
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QMainWindow, QToolBar, QAction, QStatusBar, QGridLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize
import qdarkstyle
from controller.controller_marcos import MarcosController


class MainWindow(QMainWindow):
    def __init__(self, session, demo = False, parent=None):
        super(MainWindow, self).__init__(parent)
        self.toolbarMarcos = None
        self.action_view_sequence = None
        self.action_iterate = None
        self.action_acquire = None
        self.action_add_to_list = None
        self.action_localizer = None
        self.action_autocalibration = None
        self.action_copybitstream = None
        self.marcos_action = None
        self.action_gpa_init = None
        self.action_server = None
        self.toolbar = None
        self.session = session
        self.demo = demo
        self.setWindowTitle(session['directory'])
        self.resize(QSize(800, 600))

        # Set stylesheet
        self.styleSheet = qdarkstyle.load_stylesheet_pyqt5()
        self.setStyleSheet(self.styleSheet)

        # Set up the toolbar
        self.setupToolBar()

        # Add vertical layout
        self.main_layout = QGridLayout()

        # Status bar
        self.setStatusBar(QStatusBar(self))
        
    def setupToolBar(self):

        self.toolbarMarcos = MarcosController(self.demo, "MaRCoS toolbar")
        self.addToolBar(self.toolbarMarcos)

        # Add toolbar
        self.toolbar = QToolBar("Sequence toolbar")
        self.addToolBar(self.toolbar)

        # # Setup GPA board
        # self.action_gpa_init = QAction(QIcon("resources/icons/initGPA.png"), "Init GPA board", self)
        # self.action_gpa_init.setStatusTip("Init GPA board")
        # self.toolbar.addAction(self.action_gpa_init)
        #
        # # Setup MaRCoS
        # self.action_copybitstream = QAction(QIcon("resources/icons/M.png"), "MaRCoS init", self)
        # self.action_copybitstream.setStatusTip("Install MaRCoS into Red Pitaya")
        # self.toolbar.addAction(self.action_copybitstream)
        #
        # # Connect to the server
        # self.action_server = QAction(QIcon("resources/icons/server-light.png"), "MaRCoS server", self)
        # self.action_server.setStatusTip("Connect to server")
        # self.toolbar.addAction(self.action_server)

        # Autocalibration
        self.action_autocalibration = QAction(QIcon("resources/icons/calibration-light.png"), "Autocalibration", self)
        self.action_autocalibration.setStatusTip("Run autocalibration")
        self.toolbar.addAction(self.action_autocalibration)
        
        # Localizer
        self.action_localizer = QAction(QIcon("resources/icons/localizer-light.png"), "Localizer", self)
        self.action_localizer.setStatusTip("Run Localizer")
        self.toolbar.addAction(self.action_localizer)

        # Add sequence to waiting list
        self.action_add_to_list = QAction(QIcon("resources/icons/clipboard-list-check"), "Sequence to list", self)
        self.action_add_to_list.setStatusTip("Add current sequence to waiting list")
        self.toolbar.addAction(self.action_add_to_list)

        # Add run action
        self.action_acquire = QAction(QIcon("resources/icons/acquire.png"), "Acquire", self)
        self.action_acquire.setStatusTip("Run current sequence")
        self.toolbar.addAction(self.action_acquire)

        # Iterative run
        self.action_iterate = QAction(QIcon("resources/icons/iterate.png"), "Iterative run", self)
        self.action_iterate.setStatusTip("Set iterative mode on")
        self.toolbar.addAction(self.action_iterate)

        # Plot sequence
        self.action_view_sequence = QAction(QIcon("resources/icons/plotSequence.png"), "Plot sequence", self)
        self.action_view_sequence.setStatusTip("Plot current sequence")
        self.toolbar.addAction(self.action_view_sequence)

