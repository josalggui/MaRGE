"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
from ui.window_session import SessionWindow
from controller.controller_main import MainController
import os
import sys
import configs.hw_config as hw
import subprocess


class SessionController(SessionWindow):
    """
    Controller class for managing the session.

    Inherits:
        SessionWindow: Base class for the session window.
    """
    def __init__(self):
        """
        Initializes the SessionController.
        """
        super(SessionController, self).__init__()
        self.main_gui = None

        # Set slots for toolbar actions
        self.launch_gui_action.triggered.connect(self.runMainGui)
        self.demo_gui_action.triggered.connect(self.runDemoGui)
        self.close_action.triggered.connect(self.close)

    def runMainGui(self):
        """
        Runs the main GUI and sets up the session.

        Creates a folder for the session and opens the main GUI.
        """
        self.updateSessionDict()

        # Create folder
        self.session['directory'] = 'experiments/acquisitions/%s/%s/%s/%s' % (
            self.session['project'], self.session['subject_id'], self.session['study'], self.session['side'])
        if not os.path.exists(self.session['directory']):
            os.makedirs(self.session['directory'])

        # Open the main gui
        if self.main_gui is None:
            self.main_gui = MainController(self.session, demo=False, parent=self)
        else:
            self.main_gui.saveSessionToSequences(self.session)
            self.main_gui.console.setup_console()
            self.main_gui.history_list.delete_items()
            self.main_gui.console.clear_console()
            self.main_gui.setWindowTitle(self.session['directory'])
            self.main_gui.setDemoMode(False)

        self.hide()
        self.main_gui.show()

    def runDemoGui(self):
        """
        Runs the main GUI in DEMO mode and sets up the session.

        Creates a folder for the session and opens the main GUI.
        """
        self.updateSessionDict()

        # Create folder
        self.session['directory'] = 'experiments/acquisitions/%s/%s/%s/%s' % (
            self.session['project'], self.session['subject_id'], self.session['study'], self.session['side'])
        if not os.path.exists(self.session['directory']):
            os.makedirs(self.session['directory'])

        # Open the main gui
        if self.main_gui is None:
            self.main_gui = MainController(self.session, demo=True, parent=self)
        else:
            self.main_gui.saveSessionToSequences(self.session)
            self.main_gui.console.setup_console()
            self.main_gui.history_list.delete_items()
            self.main_gui.console.clear_console()
            self.main_gui.setWindowTitle(self.session['directory'])
            self.main_gui.setDemoMode(True)

        self.hide()
        self.main_gui.show()

    def closeEvent(self, event):
        """
        Event handler for the session window close event.

        Args:
            event: The close event.
        """
        if self.main_gui is not None:
            self.main_gui.app_open = False
            if not self.main_gui.demo:
                # Close server
                try:
                    subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
                except:
                    print(
                        "ERROR: Server connection not found! Please verify if the blue LED is illuminated on the Red Pitaya.")

                # Disable power modules
                self.main_gui.toolbar_marcos.arduino.send("GPA_ON 0;")
                self.main_gui.toolbar_marcos.arduino.send("RFPA_RF 0;")
        print('GUI closed successfully!')
        super().closeEvent(event)

    def close(self):
        """
        Closes the session and exits the program.
        """
        if self.main_gui is not None:
            self.main_gui.app_open = False
            if not self.main_gui.demo:
                # Close server
                try:
                    subprocess.run(
                        [hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
                except:
                    print(
                        "ERROR: Server connection not found! Please verify if the blue LED is illuminated on the Red Pitaya.")

                # Disable power modules
                self.main_gui.toolbar_marcos.arduino.send("GPA_ON 0;")
                self.main_gui.toolbar_marcos.arduino.send("RFPA_RF 0;")
        print('GUI closed successfully!')
        sys.exit()

    def updateSessionDict(self):
        """
        Updates the session dictionary with the current session information.
        """
        self.session = {
            'project': self.project_combo_box.currentText(),
            'study': self.study_combo_box.currentText(),
            'side': self.side_combo_box.currentText(),
            'orientation': self.orientation_combo_box.currentText(),
            'subject_id': self.id_line_edit.text(),
            'study_id': self.idS_line_edit.text(),
            'subject_name': self.name_line_edit.text(),
            'subject_surname': self.surname_line_edit.text(),
            'subject_birthday': self.birthday_line_edit.text(),
            'subject_weight': self.weight_line_edit.text(),
            'subject_height': self.height_line_edit.text(),
            'scanner': self.scanner_line_edit.text(),
            'rf_coil': self.rf_coil_combo_box.currentText(),
            'seriesNumber': 0,
        }
        hw.b1Efficiency = hw.antenna_dict[self.session['rf_coil']]
