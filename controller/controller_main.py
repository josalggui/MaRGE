"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

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
        self.app_open = False
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        if not self.demo:
            # Close server
            try:
                subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
            except:
                print("ERROR: Server connection not found! Please verify if the blue LED is illuminated on the Red Pitaya.")

            # Disable power modules
            self.toolbar_marcos.arduino.send("GPA_ON 0;")
            self.toolbar_marcos.arduino.send("RFPA_RF 0;")
            
        print('GUI closed successfully!')
        super().closeEvent(event)
