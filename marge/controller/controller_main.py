"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import sys
import threading

from PyQt5.QtCore import QEvent

from marge.seq.sequences import defaultsequences
from marge.ui.window_main import MainWindow
import marge.autotuning.autotuning as autotuning
import marge.configs.hw_config as hw


class MainController(MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainController, self).__init__(*args, **kwargs)

        self.set_session(self.session)

        self.initializeThread()

        self.history_list.sequence_ready_signal.connect(self.history_list.updateHistoryFigure2)
        self.history_list.figure_ready_signal.connect(self.toolbar_figures.doScreenshot)

        # Define the arduinos
        self.arduino_autotuning = autotuning.Arduino(baudrate=hw.ard_br_autotuning)
        self.arduino_autotuning.connect(serial_number=hw.ard_sn_autotuning)
        if hw.ard_sn_autotuning==hw.ard_sn_interlock:
            self.arduino_interlock = self.arduino_autotuning
        else:
            self.arduino_interlock = autotuning.Arduino(hw.ard_br_interlock)
            self.arduino_interlock.connect(serial_number=hw.ard_sn_interlock)

    def set_demo(self, demo):
        self.demo = demo

    def set_session(self, session):
        # Set window title
        self.session = session
        self.setWindowTitle("MaRGE " + session["software_version"] + ": " + session['directory'])
        # Add the session to all sequences
        for sequence in defaultsequences.values():
            sequence.session = session

    def initializeThread(self):
        # Start the sniffer
        thread = threading.Thread(target=self.history_list.waitingForRun)
        thread.start()
        thread = threading.Thread(target=self.history_list.waitingForRecon)
        thread.start()
        print("Sniffer initialized.\n")

    def set_console(self):
        self.layout_left.addWidget(self.console)

    def changeEvent(self, event):
        if event.type() == QEvent.ActivationChange:  # Event type 99
            if self.isActiveWindow():
                self.set_console()
        super().changeEvent(event)

    def closeEvent(self, event):
        """
        Shuts down the application on close.

        This method is called when the application is being closed. It sets the `app_open` flag to False, restores
        `sys.stdout` to its default value, and performs additional cleanup tasks if the `demo` flag is not set.
        It also prints a closing message to the console.

        Args:
            event (QCloseEvent): The close event triggered by the user.

        Returns:
            None
        """
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
            
        print('\nMain GUI closed successfully!')

        self.parent.show()

        super().closeEvent(event)
