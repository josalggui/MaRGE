from PyQt5.QtCore import QEvent

from marge.ui.window_postprocessing import MainWindow

class ProcessingWindowController(MainWindow):
    def __init__(self, *args, **kwargs):
        super(ProcessingWindowController, self).__init__(*args, **kwargs)

    def set_console(self):
        self.left_layout.addWidget(self.main.console)

    def set_session(self, session):
        self.session = session
        self.setWindowTitle(session['directory'])

    def changeEvent(self, event):
        if event.type() == QEvent.ActivationChange:  # Event type 99
            if self.isActiveWindow():
                self.set_console()
        super().changeEvent(event)
