"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import sys

from PyQt5.QtWidgets import QToolBar, QAction
from PyQt5.QtGui import QIcon
from importlib import resources

class MarcosToolBar(QToolBar):
    def __init__(self, main, *args, **kwargs):
        super(MarcosToolBar, self).__init__(*args, **kwargs)
        self.main = main

        # Prepare your SD card
        with resources.path("marge.resources.icons", "redpitaya.png") as path_init_gpa:
            self.action_marcos_install = QAction(QIcon(str(path_init_gpa)), "Set up server and client", self)
        self.action_marcos_install.setStatusTip("Set up server and client")
        self.addAction(self.action_marcos_install)

        # Enable only on Linux
        if sys.platform.startswith("linux"):
            self.action_marcos_install.setEnabled(True)
        else:
            self.action_marcos_install.setEnabled(False)

        # Setup MaRCoS
        with resources.path("marge.resources.icons", "M.png") as path_m:
            self.action_copybitstream = QAction(QIcon(str(path_m)), "MaRCoS init", self)
        self.action_copybitstream.setStatusTip("Install MaRCoS into Red Pitaya")
        self.addAction(self.action_copybitstream)

        # Connect to the server
        with resources.path("marge.resources.icons", "server-light.png") as path_server:
            self.action_server = QAction(QIcon(str(path_server)), "MaRCoS server", self)
        self.action_server.setStatusTip("Connect to server")
        self.addAction(self.action_server)

        # Setup GPA board
        with resources.path("marge.resources.icons", "gpa.png") as path_gpa:
            self.action_gpa_init = QAction(QIcon(str(path_gpa)), "Init power modules", self)
        self.action_gpa_init.setStatusTip("Init GPA board")
        self.addAction(self.action_gpa_init)