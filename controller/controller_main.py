"""
session_controller.py
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import subprocess
import sys
import threading

from seq.sequences import defaultsequences
from ui.window_main import MainWindow

from configs import hw_config as hw


class MainController(MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainController, self).__init__(*args, **kwargs)

        # Add the session to all sequences
        for sequence in defaultsequences.values():
            sequence.session = self.session

        # Start the sniffer
        thread = threading.Thread(target=self.history_list.waitingForRun)
        thread.start()

    def closeEvent(self, event):
        """Shuts down application on close."""
        self.app_open = False
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        if not self.demo:
            subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
        print('\nGUI closed successfully!')
        super().closeEvent(event)
