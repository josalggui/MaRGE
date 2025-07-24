import sys
import csv
from marge.configs import hw_config as hw
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton,
    QHBoxLayout, QLabel, QGridLayout
)

class OthersWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        self.main_layout = QVBoxLayout()
        self.layout = QGridLayout()
        self.main_layout.addLayout(self.layout)

        # Parameters to save inputs
        self.labels = []
        self.values = []
        self.tips = []

        # Add inputs
        self.add_input(label="Scanner name", value="Demo", tip="Name of the MRI scanner")
        self.add_input(label="FOVx (cm)", value="20.0", tip="Field of View in the X direction")
        self.add_input(label="FOVy (cm)", value="20.0", tip="Field of View in the Y direction")
        self.add_input(label="FOVz (cm)", value="20.0", tip="Field of View in the Z direction")
        self.add_input(label="Shimming factor", value="1e-5", tip="Factor used for shimming adjustments")
        self.add_input(label="Bash path", value="gnome-terminal", tip="Path for executing bash commands")
        self.add_input(label="Arduino autotuning", value="242353133363518050E0",
                       tip="Arduino serial number for autotuning")
        self.add_input(label="Arduino interlock", value="242353133363518050E1",
                       tip="Arduino serial number for interlock system")
        self.add_input(label="Arduino attenuator", value="242353133363518050E2",
                       tip="Arduino serial number for RF attenuation")

        # Dictionary to store references to input fields
        self.input_boxes = {}

        # Create blocks iteratively
        for row, (label, value) in enumerate(zip(self.labels, self.values)):
            label_widget = QLabel(label)
            input_box = QLineEdit(value)
            input_box.setStatusTip(self.tips[row])
            self.input_boxes[label] = input_box

            self.layout.addWidget(label_widget, row, 0)  # Label in column 0
            self.layout.addWidget(input_box, row, 1)  # Input box in column 1

        # Buttons
        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save_others_entries)

        layout = QHBoxLayout()
        layout.addWidget(self.save_button)
        self.main_layout.addLayout(layout)
        self.main_layout.addStretch()

        self.setLayout(self.main_layout)
        self.setWindowTitle('Others Entry')
        self.resize(400, 200)

        # Load saved gradient entries
        self.load_others_entries()

        # Update hardware
        self.update_hw_config_others()

    def add_input(self, label="", value="", tip=""):
        self.labels.append(label)
        self.values.append(value)
        self.tips.append(tip)

    def update_hw_config_others(self):
        hw.fov = [float(self.input_boxes["FOVx (cm)"].text()),
                  float(self.input_boxes["FOVy (cm)"].text()),
                  float(self.input_boxes["FOVz (cm)"].text())]
        hw.shimming_factor = float(self.input_boxes["Shimming factor"].text())
        hw.scanner_name = self.input_boxes["Scanner name"].text()
        hw.bash_path = self.input_boxes["Bash path"].text()
        hw.ard_sn_interlock = self.input_boxes["Arduino interlock"].text()
        hw.ard_sn_attenuator = self.input_boxes["Arduino attenuator"].text()
        hw.ard_sn_autotuning = self.input_boxes["Arduino autotuning"].text()

    def save_others_entries(self):
        file_name = "configs/hw_others.csv"
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "Value"])
            for label, input_box in self.input_boxes.items():
                writer.writerow([label, input_box.text()])  # Write each pair
        self.update_hw_config_others()
        print(f"Data saved for others.")

    def load_others_entries(self):
        """Load label-value pairs from a CSV file and update the input fields."""
        file_path = "configs/hw_others.csv"
        try:
            with open(file_path, mode="r", newline="") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row

                for row in reader:
                    if len(row) == 2:  # Ensure row has two columns
                        label, value = row
                        if label in self.input_boxes:
                            self.input_boxes[label].setText(value)  # Update input box
        except:
            print("No hardware configuration loaded for others hardware.")


if __name__ == '__main__':
    # Run the application
    app = QApplication(sys.argv)
    widget = OthersWidget()
    widget.show()
    sys.exit(app.exec())