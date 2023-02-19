"""
session_controller.py
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import os
import sys
import threading

from seq.sequences import defaultsequences
from ui.window_main import MainWindow


class MainController(MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainController, self).__init__(*args, **kwargs)

        # Add the session to all sequences
        for sequence in defaultsequences.values():
            sequence.session = self.session

        thread = threading.Thread(target=self.history_list.waitingForRun)
        thread.start()

    def closeEvent(self, event):
        """Shuts down application on close."""
        self.app_open = False
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        if not self.demo:
            os.system('ssh root@192.168.1.101 "killall marcos_server"') # Kill marcos server
        print('\nGUI closed successfully!')
        super().closeEvent(event)
