"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import csv
import shutil
import os
import sys
import subprocess

import qdarkstyle

import marge.configs.hw_config as hw
from .controller_main import MainController
from marge.ui.window_session import SessionWindow


from marge.controller.controller_console import ConsoleController



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
        super().__init__()
        self.console = ConsoleController()  # Initialisation console si nécessaire

        self.session = None
        self.main_gui = None
        self.tab_session.rf_coil_combo_box.addItems(hw.antenna_dict.keys())

        # Set slots for toolbar actions
        self.launch_gui_action.triggered.connect(self.runMainGui)
        self.demo_gui_action.triggered.connect(self.runDemoGui)
        self.update_action.triggered.connect(self.update_hardware)
        self.close_action.triggered.connect(self.close)
        self.switch_theme_action.triggered.connect(self.switch_theme)


        # Check if system is ready
        self.check_system()

    def check_system(self):
        check = True
        if not os.path.exists("configs/sys_projects.csv"):
            print("ERROR: Projects not configured. Add at least one project.")
            check = False
        else:
            self.check_projects.setChecked(True)
        if not os.path.exists("configs/sys_study.csv"):
            print("ERROR: Study cases not configured. Add at least one study case.")
            check = False
        else:
            self.check_study.setChecked(True)
        if not os.path.exists("configs/hw_gradients.csv"):
            print("ERROR: Gradient hardware not configured. Go to hardware config.")
            check = False
        else:
            self.check_gradients.setChecked(True)
        if not os.path.exists("configs/hw_others.csv"):
            print("ERROR: Other hardware not configured. Go to hardware config.")
            check = False
        else:
            self.check_others.setChecked(True)
        if not os.path.exists("configs/hw_redpitayas.csv"):
            print("ERROR: Red Pitayas not configured. Go to hardware config.")
            check = False
        else:
            self.check_redpitayas.setChecked(True)
        if not os.path.exists("configs/hw_rf.csv"):
            print("ERROR: RF hardware not configured. Go to hardware config.")
            check = False
        else:
            self.check_rf.setChecked(True)
        if len(hw.rp_ip_list) == 0:
            print("ERROR: Red pitaya ip address required. Go to hardware config.")
            check = False
        else:
            self.check_rp_ips.setChecked(True)
        if len(hw.antenna_dict) == 0:
            print("ERROR: Antenna definition required. Go to hardware config.")
            check = False
        else:
            self.check_rf_coils.setChecked(True)

        if check:
            print("READY: System configuration checks succeeded.")
            self.launch_gui_action.setDisabled(False)
            self.demo_gui_action.setDisabled(False)

    def update_hardware(self):
        self.tab_session.rf_coil_combo_box.clear()
        self.tab_session.rf_coil_combo_box.addItems(hw.antenna_dict.keys())
        self.check_system()

    def runMainGui(self):
        """
        Runs the main GUI and sets up the session.
        """
        self.updateSessionDict()

        # Create folder
        self.session['directory'] = os.path.join(
            'experiments', 'acquisitions',
            self.session['project'], self.session['subject_id'], self.session['study'], self.session['side'])
        if not os.path.exists(self.session['directory']):
            os.makedirs(self.session['directory'])

        # Save session in csv and copy config files
        with open(os.path.join(self.session['directory'], "session.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            for key, value in self.session.items():
                writer.writerow([key, value])

        # Copy configuration files to session directory
        shutil.copy2("configs/hw_gradients.csv", os.path.join(self.session["directory"], "hw_gradients.csv"))
        shutil.copy2("configs/hw_others.csv", os.path.join(self.session["directory"], "hw_others.csv"))
        shutil.copy2("configs/hw_redpitayas.csv", os.path.join(self.session["directory"], "hw_redpitayas.csv"))
        shutil.copy2("configs/hw_rf.csv", os.path.join(self.session["directory"], "hw_rf.csv"))
        shutil.copy2("configs/sys_projects.csv", os.path.join(self.session["directory"], "sys_projects.csv"))
        shutil.copy2("configs/sys_study.csv", os.path.join(self.session["directory"], "sys_study.csv"))

        self.session['seriesNumber'] = 0

        # Open the main gui
        if self.main_gui is None:
            self.main_gui = MainController(session=self.session, demo=False, parent=self)
        else:
            self.main_gui.set_session(self.session)
            self.main_gui.history_list.delete_items()
            self.main_gui.console.clear_console()
            self.main_gui.set_demo(False)

        self.hide()
        self.main_gui.show()

    def runDemoGui(self):
        """
        Runs the main GUI in DEMO mode and sets up the session.
        """
        self.updateSessionDict()

        # Create folder
        self.session['directory'] = os.path.join(
            'experiments', 'acquisitions',
            self.session['project'], self.session['subject_id'], self.session['study'], self.session['side'])
        if not os.path.exists(self.session['directory']):
            os.makedirs(self.session['directory'])

        # Save session in csv and copy config files
        with open(os.path.join(self.session['directory'], "session.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            for key, value in self.session.items():
                writer.writerow([key, value])

        shutil.copy2("configs/hw_gradients.csv", os.path.join(self.session["directory"], "hw_gradients.csv"))
        shutil.copy2("configs/hw_others.csv", os.path.join(self.session["directory"], "hw_others.csv"))
        shutil.copy2("configs/hw_redpitayas.csv", os.path.join(self.session["directory"], "hw_redpitayas.csv"))
        shutil.copy2("configs/hw_rf.csv", os.path.join(self.session["directory"], "hw_rf.csv"))
        shutil.copy2("configs/sys_projects.csv", os.path.join(self.session["directory"], "sys_projects.csv"))
        shutil.copy2("configs/sys_study.csv", os.path.join(self.session["directory"], "sys_study.csv"))

        self.session['seriesNumber'] = 0

        # Open the main gui
        if self.main_gui is None:
            self.main_gui = MainController(session=self.session, demo=True, parent=self)
        else:
            self.main_gui.set_session(self.session)
            self.main_gui.history_list.delete_items()
            self.main_gui.console.clear_console()
            self.main_gui.set_demo(True)

        self.hide()
        self.main_gui.show()

    def closeEvent(self, event):
        """
        Handle the window close event cleanly.

        Args:
            event: The close event.
        """
        if self.main_gui is not None:
            self.main_gui.app_open = False
            if not self.main_gui.demo:
                # Close server
                try:
                    subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
                except Exception as e:
                    print("ERROR: Server connection not found! Please verify if the blue LED is illuminated on the Red Pitaya.")
                    print(str(e))

                # Disable power modules
                try:
                    self.main_gui.toolbar_marcos.arduino.send("GPA_ON 0;")
                    self.main_gui.toolbar_marcos.arduino.send("RFPA_RF 0;")
                except Exception as e:
                    print("ERROR: Could not disable power modules.")
                    print(str(e))

        # Close console logging if exists
        if hasattr(self, 'console'):
            self.console.close_log()

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
                except Exception as e:
                    print("ERROR: Server connection not found! Please verify if the blue LED is illuminated on the Red Pitaya.")
                    print(str(e))

                # Disable power modules
                try:
                    self.main_gui.toolbar_marcos.arduino.send("GPA_ON 0;")
                    self.main_gui.toolbar_marcos.arduino.send("RFPA_RF 0;")
                except Exception as e:
                    print("ERROR: Could not disable power modules.")
                    print(str(e))

        # Close console logging if exists
        if hasattr(self, 'console'):
            self.console.close_log()

        print('GUI closed successfully!')
        sys.exit()

    def switch_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        if self.is_dark_theme:
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        else:
            self.setStyleSheet("")  # Light theme: default Qt style

    def updateSessionDict(self):
        """
        Updates the session dictionary with the current session information.
        """

        def get_text_or_placeholder(widget):
            return widget.text() if widget.text() else widget.placeholderText()

        self.session = {'project': self.tab_session.project_combo_box.currentText(),
                        'study': self.tab_session.study_combo_box.currentText(),
                        'side': self.tab_session.side_combo_box.currentText(),
                        'orientation': self.tab_session.orientation_combo_box.currentText(),
                        'subject_id': get_text_or_placeholder(self.tab_session.id_line_edit),
                        'study_id': get_text_or_placeholder(self.tab_session.idS_line_edit),
                        'subject_name': get_text_or_placeholder(self.tab_session.name_line_edit),
                        'subject_surname': get_text_or_placeholder(self.tab_session.surname_line_edit),
                        'subject_birthday': get_text_or_placeholder(self.tab_session.birthday_line_edit),
                        'subject_weight': get_text_or_placeholder(self.tab_session.weight_line_edit),
                        'subject_height': get_text_or_placeholder(self.tab_session.height_line_edit),
                        'scanner': get_text_or_placeholder(self.tab_session.scanner_line_edit),
                        'rf_coil': self.tab_session.rf_coil_combo_box.currentText(),
                        'software_version': get_text_or_placeholder(self.tab_session.software_line_edit),
                        'black_theme': self.is_dark_theme}

        # Save session theme

        hw.b1Efficiency = hw.antenna_dict.get(self.session['rf_coil'], 1.0)
