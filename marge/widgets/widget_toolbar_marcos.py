"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QToolBar, QAction
from PyQt5.QtGui import QIcon

class MarcosToolBar(QToolBar):
    def __init__(self, main, *args, **kwargs):
        super(MarcosToolBar, self).__init__(*args, **kwargs)
        self.main = main

        # Setup all
        self.action_start = QAction(QIcon("resources/icons/initGPA.png"), "Setup MaRCoS", self)
        self.action_start.setStatusTip("Setup MaRCoS")
        self.addAction(self.action_start)

        # Setup MaRCoS
        self.action_copybitstream = QAction(QIcon("resources/icons/M.png"), "MaRCoS init", self)
        self.action_copybitstream.setStatusTip("Install MaRCoS into Red Pitaya")
        self.addAction(self.action_copybitstream)

        # Connect to the server
        self.action_server = QAction(QIcon("resources/icons/server-light.png"), "MaRCoS server", self)
        self.action_server.setStatusTip("Connect to server")
        self.addAction(self.action_server)

        # Setup GPA board
        self.action_gpa_init = QAction(QIcon("resources/icons/gpa.png"), "Init power modules", self)
        self.action_gpa_init.setStatusTip("Init GPA board")
        self.addAction(self.action_gpa_init)
        