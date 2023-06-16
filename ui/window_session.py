"""
session_controller.py
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QToolBar, QAction, QStatusBar, QGridLayout, QWidget, \
    QComboBox, QLineEdit, QSizePolicy
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import qdarkstyle
from datetime import datetime
import configs.hw_config as hw
import configs.sys_config as sys_config
import copy


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
        self.launch_gui_action.setStatusTip("This is your button")
        self.toolbar.addAction(self.launch_gui_action)

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
        self.project_combo_box = QComboBox()
        self.project_combo_box.addItems(sys_config.projects)
        self.project_combo_box.setStatusTip("Select the project")
        self.main_layout.addWidget(self.project_combo_box, row, 1)

        # Create QComboBox for study case
        row += 1
        self.main_layout.addWidget(QLabel("Study"), row, 0)
        self.study_combo_box = QComboBox()
        self.study_combo_box.addItems(sys_config.study_case)
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
        self.user_line_edit = ClickableLineEdit("User")
        self.user_line_edit.setStatusTip("Write the user name")
        self.main_layout.addWidget(self.user_line_edit, row, 1)

        # Create QLineEdit for subject id
        row += 1
        date = datetime.now()
        date_string = date.strftime("%Y.%m.%d.%H.%M.%S")[:-3]
        self.main_layout.addWidget(QLabel("Subject ID"), row, 0)
        self.id_line_edit = ClickableLineEdit(date_string)
        self.id_line_edit.setStatusTip("Write the subject id")
        self.main_layout.addWidget(self.id_line_edit, row, 1)

        # Create QLineEdit for study id
        row += 1
        date = datetime.now()
        self.main_layout.addWidget(QLabel("Study ID"), row, 0)
        self.idS_line_edit = ClickableLineEdit(date_string)
        self.idS_line_edit.setStatusTip("Write the study id")
        self.main_layout.addWidget(self.idS_line_edit, row, 1)

        # Create QLineEdit for subject name
        row += 1
        self.main_layout.addWidget(QLabel("Subject name"), row, 0)
        self.name_line_edit = ClickableLineEdit("Name")
        self.name_line_edit.setStatusTip("Write the subject name")
        self.main_layout.addWidget(self.name_line_edit, row, 1)

        # Create QLineEdit for subject surname
        row += 1
        self.main_layout.addWidget(QLabel("Subject surname"), row, 0)
        self.surname_line_edit = ClickableLineEdit("Surname")
        self.surname_line_edit.setStatusTip("Write the subject surname")
        self.main_layout.addWidget(self.surname_line_edit, row, 1)

        # Create QLineEdit for subject birthday
        row += 1
        self.main_layout.addWidget(QLabel("Subject birthday"), row, 0)
        self.birthday_line_edit = ClickableLineEdit("YY/MM/DD")
        self.birthday_line_edit.setStatusTip("Write the subject birthdate")
        self.main_layout.addWidget(self.birthday_line_edit, row, 1)

        # Create QLineEdit for subject wight
        row += 1
        self.main_layout.addWidget(QLabel("Subject weight"), row, 0)
        self.weight_line_edit = ClickableLineEdit("kg")
        self.weight_line_edit.setStatusTip("Write the subject weight")
        self.main_layout.addWidget(self.weight_line_edit, row, 1)

        # Create QLineEdit for subject height
        row += 1
        self.main_layout.addWidget(QLabel("Subject height"), row, 0)
        self.height_line_edit = ClickableLineEdit("cm")
        self.height_line_edit.setStatusTip("Write the subject height")
        self.main_layout.addWidget(self.height_line_edit, row, 1)

        # Create QLineEdit for subject scanner
        row += 1
        self.main_layout.addWidget(QLabel("Scanner"), row, 0)
        self.scanner_line_edit = ClickableLineEdit(hw.scanner_name)
        self.scanner_line_edit.setDisabled(True)
        self.scanner_line_edit.setStatusTip("Scanner version")
        self.main_layout.addWidget(self.scanner_line_edit, row, 1)

        # Create QComboBox for RF coil
        row += 1
        self.main_layout.addWidget(QLabel("RF coil"), row, 0)
        self.rf_coil_combo_box = QComboBox()
        self.rf_coil_combo_box.addItems(hw.antenna_dict.keys())
        self.rf_coil_combo_box.setStatusTip("Select the rf coil")
        self.main_layout.addWidget(self.rf_coil_combo_box, row, 1)

        # Session dictionary
        self.session = {}

        # Status bar
        self.setStatusBar(QStatusBar(self))

        # Show the window
        self.show()


class ClickableLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super(ClickableLineEdit, self).__init__(*args, **kwargs)
        self.default_text = copy.copy(self.text())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.text() == self.default_text:
                self.setText("")
        super().mousePressEvent(event)


if __name__ == '__main__':
    # Only one QApplication for event loop
    app = QApplication(sys.argv)

    # Instantiate the window
    window = SessionWindow()

    # Start the event loop.
    app.exec()
