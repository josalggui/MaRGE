"""
session_controller.py
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import csv
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QToolBar, QAction, QStatusBar, QGridLayout, QWidget, \
    QComboBox, QLineEdit, QSizePolicy, QInputDialog, QMessageBox
from PyQt5.QtGui import QIcon
import qdarkstyle
from datetime import datetime
import configs.hw_config as hw
import configs.sys_config as sys_config
import os


class SessionWindow(QMainWindow):

    def __init__(self):
        super(SessionWindow, self).__init__()
        self.setWindowTitle("Session Window")
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        self.setMinimumWidth(400)

        # Set stylesheet
        self.styleSheet = qdarkstyle.load_stylesheet_pyqt5()
        self.setStyleSheet(self.styleSheet)

        # Add toolbar
        self.toolbar = QToolBar("Session toolbar")
        self.addToolBar(self.toolbar)

        # launch gui action
        self.launch_gui_action = QAction(QIcon("resources/icons/home.png"), "Launch GUI", self)
        self.launch_gui_action.setStatusTip("Launch GUI")
        self.toolbar.addAction(self.launch_gui_action)
        self.launch_gui_action.setDisabled(True)

        # demo gui action
        self.demo_gui_action = QAction(QIcon("resources/icons/demo.png"), "Launch GUI as DEMO", self)
        self.demo_gui_action.setStatusTip("Launch GUI as DEMO")
        self.toolbar.addAction(self.demo_gui_action)
        self.demo_gui_action.setDisabled(True)

        # demo gui action
        self.setup_hardware_action = QAction(QIcon("resources/icons/calibration-light.png"), "Setup scanner hardware", self)
        self.setup_hardware_action.setStatusTip("Setup scanner hardware")
        self.toolbar.addAction(self.setup_hardware_action)

        # Close button action
        self.close_action = QAction(QIcon("resources/icons/close.png"), "Close GUI", self)
        self.close_action.setStatusTip("Close the GUI")
        self.toolbar.addAction(self.close_action)

        # Create central widget that will contain the layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        # Add layout to input parameters
        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        # Create QComboBox for project
        row = 0
        self.main_layout.addWidget(QLabel("Project"), row, 0)
        self.project_combo_box = DynamicComboBox("configs/sys_projects.csv")
        self.project_combo_box.setStatusTip("Select the project")
        self.main_layout.addWidget(self.project_combo_box, row, 1)

        # Create QComboBox for study case
        row += 1
        self.main_layout.addWidget(QLabel("Study"), row, 0)
        self.study_combo_box = DynamicComboBox("configs/sys_study.csv")
        self.study_combo_box.setStatusTip("Select the study")
        self.main_layout.addWidget(self.study_combo_box, row, 1)

        # Create QComboBox for side
        row += 1
        self.main_layout.addWidget(QLabel("Side"), row, 0)
        self.side_combo_box = QComboBox()
        self.side_combo_box.addItems(sys_config.side)
        self.side_combo_box.setStatusTip("Select the subject side")
        self.main_layout.addWidget(self.side_combo_box, row, 1)

        # Create QComboBox for orientation
        row += 1
        self.main_layout.addWidget(QLabel("Orientation"), row, 0)
        self.orientation_combo_box = QComboBox()
        self.orientation_combo_box.addItems(sys_config.orientation)
        self.orientation_combo_box.setStatusTip("Select the subject orientation")
        self.main_layout.addWidget(self.orientation_combo_box, row, 1)

        # Create QLineEdit for user id
        row += 1
        self.main_layout.addWidget(QLabel("User"), row, 0)
        self.user_line_edit = QLineEdit()
        self.user_line_edit.setPlaceholderText("User")
        self.user_line_edit.setStatusTip("Write the user name")
        self.main_layout.addWidget(self.user_line_edit, row, 1)

        # Create QLineEdit for subject id
        row += 1
        date = datetime.now()
        date_string = date.strftime("%Y.%m.%d.%H.%M.%S")[:-3]
        self.main_layout.addWidget(QLabel("Subject ID"), row, 0)
        self.id_line_edit = QLineEdit()
        self.id_line_edit.setPlaceholderText(date_string)
        self.id_line_edit.setStatusTip("Write the subject id")
        self.main_layout.addWidget(self.id_line_edit, row, 1)

        # Create QLineEdit for study id
        row += 1
        date = datetime.now()
        self.main_layout.addWidget(QLabel("Study ID"), row, 0)
        self.idS_line_edit = QLineEdit()
        self.idS_line_edit.setPlaceholderText(date_string)
        self.idS_line_edit.setStatusTip("Write the study id")
        self.main_layout.addWidget(self.idS_line_edit, row, 1)

        # Create QLineEdit for subject name
        row += 1
        self.main_layout.addWidget(QLabel("Subject name"), row, 0)
        self.name_line_edit = QLineEdit()
        self.name_line_edit.setPlaceholderText("Name")
        self.name_line_edit.setStatusTip("Write the subject name")
        self.main_layout.addWidget(self.name_line_edit, row, 1)

        # Create QLineEdit for subject surname
        row += 1
        self.main_layout.addWidget(QLabel("Subject surname"), row, 0)
        self.surname_line_edit = QLineEdit()
        self.surname_line_edit.setPlaceholderText("Surname")
        self.surname_line_edit.setStatusTip("Write the subject surname")
        self.main_layout.addWidget(self.surname_line_edit, row, 1)

        # Create QLineEdit for subject birthday
        row += 1
        self.main_layout.addWidget(QLabel("Subject birthday"), row, 0)
        self.birthday_line_edit = QLineEdit()
        self.birthday_line_edit.setPlaceholderText("YY/MM/DD")
        self.birthday_line_edit.setStatusTip("Write the subject birthdate")
        self.main_layout.addWidget(self.birthday_line_edit, row, 1)

        # Create QLineEdit for subject wight
        row += 1
        self.main_layout.addWidget(QLabel("Subject weight"), row, 0)
        self.weight_line_edit = QLineEdit()
        self.weight_line_edit.setPlaceholderText("kg")
        self.weight_line_edit.setStatusTip("Write the subject weight")
        self.main_layout.addWidget(self.weight_line_edit, row, 1)

        # Create QLineEdit for subject height
        row += 1
        self.main_layout.addWidget(QLabel("Subject height"), row, 0)
        self.height_line_edit = QLineEdit()
        self.height_line_edit.setPlaceholderText("cm")
        self.height_line_edit.setStatusTip("Write the subject height")
        self.main_layout.addWidget(self.height_line_edit, row, 1)

        # Create QLineEdit for subject scanner
        row += 1
        self.main_layout.addWidget(QLabel("Scanner"), row, 0)
        self.scanner_line_edit = QLineEdit(hw.scanner_name)
        self.scanner_line_edit.setDisabled(True)
        self.scanner_line_edit.setStatusTip("Scanner version")
        self.main_layout.addWidget(self.scanner_line_edit, row, 1)

        # Create QComboBox for RF coil
        row += 1
        self.main_layout.addWidget(QLabel("RF coil"), row, 0)
        self.rf_coil_combo_box = QComboBox()
        self.rf_coil_combo_box.setStatusTip("Select the rf coil")
        self.main_layout.addWidget(self.rf_coil_combo_box, row, 1)

        # Session dictionary
        self.session = {}

        # Status bar
        self.setStatusBar(QStatusBar(self))

        # Show the window
        self.show()

class DynamicComboBox(QComboBox):
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

        # Load items from CSV
        self.load_items()

        # Add the special entry
        self.addItem("Add/Delete...")
        self.activated.connect(self.check_add_new)

    def check_add_new(self, index):
        if self.itemText(index) == "Add/Delete...":
            new_item, ok = QInputDialog.getText(self, "Modify List", "Enter new item to add/remove:")

            if ok and new_item:
                existing_index = self.findText(new_item)

                if existing_index != -1:  # Item already exists → Remove it
                    self.removeItem(existing_index)
                    QMessageBox.information(self, "Item Removed", f"'{new_item}' has been removed.")
                else:  # Item does not exist → Add it before "Add/Delete..."
                    self.insertItem(self.count() - 1, new_item)
                    self.setCurrentIndex(self.count() - 2)  # Select the new item

                self.save_items()  # Update CSV file

    def load_items(self):
        """Load saved items from CSV file."""
        if os.path.exists(self.file_name):
            with open(self.file_name, newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row:  # Avoid empty lines
                        self.addItem(row[0])

    def save_items(self):
        """Save current items to CSV file."""
        with open(self.file_name, "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for i in range(self.count()):
                item_text = self.itemText(i)
                if item_text != "Add/Delete...":  # Don't save the special entry
                    writer.writerow([item_text])


if __name__ == '__main__':
    # Only one QApplication for event loop
    app = QApplication(sys.argv)

    # Instantiate the window
    window = SessionWindow()

    # Start the event loop.
    app.exec()
