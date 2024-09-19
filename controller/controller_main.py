"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import sys
import threading

from seq.sequences import defaultsequences
from ui.window_main import MainWindow


class MainController(MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainController, self).__init__(*args, **kwargs)

        self.saveSessionToSequences(self.session)

        self.initializeThread()

        self.console.setup_console()

    def saveSessionToSequences(self, session):
        # Add the session to all sequences
        for sequence in defaultsequences.values():
            sequence.session = session

    def initializeThread(self):
        # Start the sniffer
        thread = threading.Thread(target=self.history_list.waitingForRun)
        thread.start()
        print("Sniffer initialized.\n")

    def mousePressEvent(self, event):
        # Send self.main.post_gui.console.setup_console()prints to current window console
        self.console.setup_console()

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
