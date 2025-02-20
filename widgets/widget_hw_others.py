import sys
import csv
import numpy as np
from configs import hw_config as hw
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton,
    QHBoxLayout, QLabel
)

class OthersWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        self.main_layout = QVBoxLayout()
        self.layout = QVBoxLayout()
        self.main_layout.addLayout(self.layout)

        # Labels and Boxes lists
        labels = ["Scanner name",
                    "FOVx (cm)",  # FOV
                    "FOVy (cm)",
                    "FOVz (cm)",
                    "Shimming factor",  # shimming
                    "Bash path",  # others
                    "Arduino autotuning",
                    "Arduino interlock",
                    "Arduino attenuator",
                  ]
        values = ["Demo",
                  "20.0",
                  "20.0",
                  "20.0",
                  "1e-5",
                  "gnome-terminal",
                  "242353133363518050E0",
                  "242353133363518050E1",
                  "242353133363518050E2",
                  ]

        # Dictionary to store references to input fields
        self.input_boxes = {}

        # Create blocks iteratively
        for label, value in zip(labels, values):
            row_layout = QHBoxLayout()
            label_widget = QLabel(label)
            input_box = QLineEdit(value)
            self.input_boxes[label] = input_box
            row_layout.addWidget(label_widget)
            row_layout.addWidget(input_box)
            self.layout.addLayout(row_layout)

        # Buttons
        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save_others_entries)

        layout = QHBoxLayout()
        layout.addWidget(self.save_button)
        self.main_layout.addStretch()
        self.main_layout.addLayout(layout)

        self.setLayout(self.main_layout)
        self.setWindowTitle('Others Entry')
        self.resize(400, 200)

        # Load saved gradient entries
        self.load_others_entries()

        # Update hardware
        self.update_hw_config_rp()

    def update_hw_config_rp(self):
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
        file_name = "../configs/hw_others.csv"
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "Value"])
            for label, input_box in self.input_boxes.items():
                writer.writerow([label, input_box.text()])  # Write each pair
        print(f"Data saved for others.")

    def load_others_entries(self):
        """Load label-value pairs from a CSV file and update the input fields."""
        file_path = "../configs/hw_others.csv"
        try:
            with open(file_path, mode="r", newline="") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row

                for row in reader:
                    if len(row) == 2:  # Ensure row has two columns
                        label, value = row
                        if label in self.input_boxes:
                            self.input_boxes[label].setText(value)  # Update input box

                print(f"Hardware configuration loaded.")
        except:
            print("No hardware configuration loaded for others hardware.")


if __name__ == '__main__':
    # Run the application
    app = QApplication(sys.argv)
    widget = OthersWidget()
    widget.show()
    sys.exit(app.exec())