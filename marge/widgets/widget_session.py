import csv
import os
import subprocess
import sys
from datetime import datetime

from PyQt5.QtWidgets import QWidget, QSizePolicy, QGridLayout, QComboBox, QInputDialog, QMessageBox, QApplication, \
    QLabel, QLineEdit

from marge.configs import hw_config, sys_config


class SessionWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Session Window")
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        self.setMinimumWidth(400)

        # Add layout to input parameters
        self.main_layout = QGridLayout()
        self.setLayout(self.main_layout)

        # Create QComboBox for project
        row = 0
        self.main_layout.addWidget(QLabel("Project"), row, 0)
        self.project_combo_box = DynamicComboBox("configs/sys_projects.csv", title="Add/Delete Project")
        self.project_combo_box.setStatusTip("Select the project")
        self.main_layout.addWidget(self.project_combo_box, row, 1)

        # Create QComboBox for study case
        row += 1
        self.main_layout.addWidget(QLabel("Study"), row, 0)
        self.study_combo_box = DynamicComboBox("configs/sys_study.csv", title="Add/Delete Study")
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

        # Create QComboBox for RF coil
        row += 1
        self.main_layout.addWidget(QLabel("RF coil"), row, 0)
        self.rf_coil_combo_box = QComboBox()
        self.rf_coil_combo_box.setStatusTip("Select the rf coil")
        self.rf_coil_combo_box.addItems(hw_config.antenna_dict.keys())
        self.main_layout.addWidget(self.rf_coil_combo_box, row, 1)

        # Create QLineEdit for scanner
        row += 1
        self.main_layout.addWidget(QLabel("Scanner"), row, 0)
        self.scanner_line_edit = QLineEdit(hw_config.scanner_name)
        self.scanner_line_edit.setDisabled(True)
        self.scanner_line_edit.setStatusTip("Scanner version")
        self.main_layout.addWidget(self.scanner_line_edit, row, 1)

        # Create QLineEdit for software version
        # Get repo or pypi version
        try:
            tag = subprocess.check_output(
                ['git', 'describe', '--tags'],
                stderr=subprocess.STDOUT
            ).strip().decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(f"Error getting Git tag: {e.output.decode('utf-8')}")
            from importlib.metadata import version

            try:
                tag = version("marge-mri")  # Replace with your actual PyPI package name
            except Exception as e2:
                print(f"Error getting PyPI version: {e2}")
                tag = "unknown"
        row += 1
        self.main_layout.addWidget(QLabel("Software version"), row, 0)
        self.software_line_edit = QLineEdit(tag)
        self.software_line_edit.setDisabled(True)
        self.software_line_edit.setStatusTip("Software version")
        self.main_layout.addWidget(self.software_line_edit, row, 1)

        # Session dictionary
        self.session = {}

class DynamicComboBox(QComboBox):
    def __init__(self, file_name, title="Add/Delete Project"):
        super().__init__()
        self.file_name = file_name
        self.title = title

        # Load items from CSV
        self.load_items()

        # Add the special entry
        self.addItem("Add/Delete...")
        self.activated.connect(self.check_add_new)

    def check_add_new(self, index):
        if self.itemText(index) == "Add/Delete...":
            new_item, ok = QInputDialog.getText(self, self.title, "Enter new item:")

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


        # Ensure the configs directory exists before writing
        os.makedirs(os.path.dirname(self.file_name), exist_ok=True)

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
    window = SessionWidget()
    window.show()

    # Start the event loop.
    app.exec()