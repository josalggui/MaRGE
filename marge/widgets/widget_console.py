"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QMainWindow, QTextEdit, QSizePolicy

class ConsoleWidget(QMainWindow):
    """Read-only QTextEdit console embedded in a narrow main window for log output."""
    def __init__(self):
        super().__init__()

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.setCentralWidget(self.console)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setMaximumWidth(400)


