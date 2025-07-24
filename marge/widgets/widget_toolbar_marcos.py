"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QToolBar, QAction
from PyQt5.QtGui import QIcon
from importlib import resources

class MarcosToolBar(QToolBar):
    def __init__(self, main, *args, **kwargs):
        super(MarcosToolBar, self).__init__(*args, **kwargs)
        self.main = main

        # Setup all
        with resources.path("marge.resources.icons", "initGPA.png") as path_init_gpa:
            self.action_start = QAction(QIcon(str(path_init_gpa)), "Setup MaRCoS", self)
        self.action_start.setStatusTip("Setup MaRCoS")
        self.addAction(self.action_start)

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