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
    Controller responsible for session setup, validation, and lifecycle management.

    This class manages session configuration through the GUI, including system
    and hardware validation, session metadata collection, theme handling, and
    controlled startup and shutdown of the main application. It coordinates the
    transition between the session setup interface and the main GUI, supporting
    both normal and DEMO operation modes.

    Inherits:
        SessionWindow: Base class providing the session configuration UI.
    """
    def __init__(self):
        """
        Initialize the session controller and configure the session setup interface.

        This constructor initializes the base session window, sets up console
        logging, prepares internal session state, and populates hardware-dependent
        UI elements such as the RF coil selector. The previously selected RF coil is
        restored from configuration if available.

        It also connects toolbar actions to their corresponding handlers, applies
        initial system validation checks, and prepares the controller for launching
        the main application in either standard or DEMO mode.

        Side Effects:
            - Initializes console logging.
            - Populates RF coil selection from hardware configuration.
            - Restores RF coil selection from configuration file when available.
            - Connects GUI actions to controller methods.
            - Performs an initial system readiness check.

        Returns:
            None
        """
        super().__init__()
        self.console = ConsoleController()  # Initialisation console si nécessaire

        self.session = None
        self.main_gui = None
        self.tab_session.rf_coil_combo_box.addItems(hw.antenna_dict.keys())

        # Nuevo #Posicionar el item del combo según la inf guardada en b1Efficiency.csv
        def read_rf_from_csv(path_csv: str) -> str | None:
            try:
                with open(path_csv, newline="", encoding="utf-8") as f:
                    row = next(csv.reader(f))
                coil = row[0].strip()
                return coil if coil else None
            except Exception:
                return None

        def select_rf_from_csv(combo, path_csv: str) -> None:
            if combo.count() == 0:
                return

            coil = read_rf_from_csv(path_csv)
            if coil:
                idx = combo.findText(coil)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                combo.setCurrentIndex(0)

        path_b1 = os.path.join("configs", "b1Efficiency.csv")
        select_rf_from_csv(self.tab_session.rf_coil_combo_box, path_b1)

        # Set slots for toolbar actions
        self.launch_gui_action.triggered.connect(self.runMainGui)
        self.demo_gui_action.triggered.connect(self.runDemoGui)
        self.update_action.triggered.connect(self.update_hardware)
        self.close_action.triggered.connect(self.close)
        self.switch_theme_action.triggered.connect(self.switch_theme)


        # Check if system is ready
        self.check_system()

    def check_system(self):
        """
        Verify that all required system configuration files and hardware settings
        are present and valid.

        This method checks for the existence of required configuration CSV files
        (projects, study cases, and hardware definitions), validates that essential
        hardware parameters (e.g., Red Pitaya IP addresses and RF antenna
        definitions) are defined, and updates the corresponding GUI check indicators.

        For each missing or invalid configuration, an error message is printed and
        the system is marked as not ready. If all checks pass, the system is marked
        as ready and GUI actions for launching or demoing the application are
        enabled.

        Side Effects:
            - Prints error or success messages to stdout.
            - Updates GUI checkboxes reflecting configuration status.
            - Enables or disables GUI actions based on overall system readiness.

        Returns:
            None
        """
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
        """
        Refresh hardware-related UI elements and revalidate system configuration.

        This method updates the RF coil selection combo box using the currently
        defined antenna configurations, then re-runs the full system configuration
        check to ensure all hardware requirements are satisfied after the update.

        Side Effects:
            - Clears and repopulates the RF coil combo box.
            - Triggers a full system configuration check.
            - May update GUI state and readiness indicators.

        Returns:
            None
        """
        self.tab_session.rf_coil_combo_box.clear()
        self.tab_session.rf_coil_combo_box.addItems(hw.antenna_dict.keys())
        self.check_system()

    def runMainGui(self):
        """
        Initialize a new acquisition session and launch the main GUI.

        This method updates the current session dictionary, creates the session
        directory structure, saves session metadata to disk, and copies all relevant
        system and hardware configuration files into the session folder for
        reproducibility.

        It then initializes or refreshes the main GUI controller with the current
        session settings, resets GUI state as needed, and switches control from the
        setup interface to the main application interface.

        Side Effects:
            - Creates a new session directory on disk.
            - Writes session metadata to a CSV file.
            - Copies system and hardware configuration files into the session folder.
            - Initializes or updates the main GUI controller.
            - Hides the current GUI and displays the main GUI.

        Returns:
            None
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
        shutil.copy2("configs/b1Efficiency.csv", os.path.join(self.session["directory"], "b1Efficiency.csv"))

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
        Initialize a new acquisition session and launch the main GUI in DEMO mode.

        This method performs the same session setup steps as a standard run,
        including updating the session dictionary, creating the session directory,
        saving session metadata, and copying all relevant system and hardware
        configuration files for reproducibility.

        The main GUI is then initialized or refreshed in DEMO mode, ensuring that
        hardware-dependent functionality is disabled or simulated as appropriate.
        Control is transferred from the setup interface to the main GUI.

        Side Effects:
            - Creates a new session directory on disk.
            - Writes session metadata to a CSV file.
            - Copies system and hardware configuration files into the session folder.
            - Initializes or updates the main GUI controller in DEMO mode.
            - Hides the current GUI and displays the main GUI.

        Returns:
            None
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
        shutil.copy2("configs/b1Efficiency.csv", os.path.join(self.session["directory"], "b1Efficiency.csv"))

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
        Handle application shutdown and safely release hardware resources.

        This method is triggered when the GUI window is closed. It ensures that any
        active acquisition or hardware communication is cleanly terminated,
        including stopping remote servers and disabling power modules when not
        running in DEMO mode. Console logging is also closed if present.

        Any errors encountered during shutdown are caught and reported to prevent
        an unclean application exit.

        Args:
            event: Qt close event triggering the shutdown sequence.

        Returns:
            None
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
                    self.main_gui.arduino_interlock.send("GPA_ON 0;")
                    self.main_gui.arduino_interlock.send("RFPA_RF 0;")
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
        Terminate the active session, safely shut down hardware, and exit the program.

        This method explicitly closes the application by stopping any active hardware
        communication, terminating remote servers, and disabling power modules when
        not operating in DEMO mode. Console logging is closed if present, and the
        Python process is then exited.

        Unlike the Qt close event handler, this method forces program termination
        via ``sys.exit()`` and is intended for controlled shutdown initiated by the
        application logic.

        Side Effects:
            - Terminates hardware communication and remote services.
            - Disables power modules when applicable.
            - Closes console logging.
            - Exits the Python process.

        Returns:
            None
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
                    self.main_gui.arduino_interlock.send("GPA_ON 0;")
                    self.main_gui.arduino_interlock.send("RFPA_RF 0;")
                except Exception as e:
                    print("ERROR: Could not disable power modules.")
                    print(str(e))

        # Close console logging if exists
        if hasattr(self, 'console'):
            self.console.close_log()

        print('GUI closed successfully!')
        sys.exit()

    def switch_theme(self):
        """
        Toggle the application theme between dark and default modes.

        This method switches the internal theme state and applies the corresponding
        Qt stylesheet. When enabled, a dark theme stylesheet is loaded and applied;
        otherwise, the application reverts to the default (light) style.

        Side Effects:
            - Updates the application stylesheet.
            - Toggles the internal theme state flag.

        Returns:
            None
        """
        self.is_dark_theme = not self.is_dark_theme
        if self.is_dark_theme:
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        else:
            self.setStyleSheet("")

    def updateSessionDict(self):
        """
        Collect current GUI inputs and update the session metadata dictionary.

        This method reads all relevant session, subject, and system fields from the
        GUI, falling back to placeholder text when input fields are empty, and
        stores the results in the session dictionary. The selected RF coil is also
        used to update the global B1 efficiency setting.

        In addition, the current RF coil and its B1 efficiency value are written to
        the configuration file to ensure consistency across sessions.

        Side Effects:
            - Updates ``self.session`` with current GUI values.
            - Updates global hardware B1 efficiency based on selected RF coil.
            - Writes RF coil B1 efficiency to ``configs/b1Efficiency.csv``.

        Returns:
            None
        """

        def get_text_or_placeholder(widget):
            return widget.text() if widget.text() else widget.placeholderText()

        self.session = {
            'project': self.tab_session.project_combo_box.currentText(),
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
            'user': get_text_or_placeholder(self.tab_session.user_line_edit),
            'rf_coil': self.tab_session.rf_coil_combo_box.currentText(),
            'software_version': get_text_or_placeholder(self.tab_session.software_line_edit),
            'scanner_name': get_text_or_placeholder(self.tab_others.input_boxes["Scanner name"]),
            'scanner_manufacturer': get_text_or_placeholder(self.tab_others.input_boxes["Manufacturer"]),
            'institution_name': get_text_or_placeholder(self.tab_others.input_boxes["Institution name"]),
            'black_theme': self.is_dark_theme,
        }

        hw.b1Efficiency = hw.antenna_dict.get(self.session['rf_coil'], 1.0)

        # Save current rf coil to configs/b1Efficiency.csv
        path_b1 = os.path.join("configs", "b1Efficiency.csv")
        coil = self.session.get('rf_coil', '')
        eff = float(hw.antenna_dict.get(coil, 1.0))
        with open(path_b1, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([coil, f"{eff:.6f}"])
