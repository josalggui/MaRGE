import sys
import os
import csv
import numpy as np
from marge.configs import hw_config as hw
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton,
    QHBoxLayout, QLabel, QGridLayout
)

class GradientsWidget(QWidget):
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
        self.add_input(label="Gx max (mT/m)", value="50", tip="Maximum gradient along X direction")
        self.add_input(label="Gy max (mT/m)", value="80", tip="Maximum gradient along Y direction")
        self.add_input(label="Gz max (mT/m)", value="70", tip="Maximum gradient along Z direction")
        self.add_input(label="Max slew rate (mT/m/ms)", value="80", tip="Maximum slew rate")
        self.add_input(label="Gradient raster time (us)", value="50", tip="Time resolution of gradient system")
        self.add_input(label="Gradient rise time (us)", value="400", tip="Time to reach maximum gradient amplitude")
        self.add_input(label="Gradient steps", value="16", tip="Number of steps for gradient transition")
        self.add_input(label="Gradient delay (us)", value="9", tip="Delay before gradient application")
        self.add_input(label="GPA model", value="None",
                       tip="GPA model: Only 'Barthel' is supported; otherwise, leave it empty.")

        # Dictionary to store references to input fields
        self.input_boxes = {}

        # Create blocks iteratively
        for row, (label, value) in enumerate(zip(self.labels, self.values)):
            label_widget = QLabel(label)
            input_box = QLineEdit(value)
            input_box.setStatusTip(self.tips[row])
            self.input_boxes[label] = input_box

            self.layout.addWidget(label_widget, row, 0)  # Label in column 0
            self.layout.addWidget(input_box, row, 1)     # Input box in column 1

        # Buttons
        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save_gradient_entries)

        layout = QHBoxLayout()
        layout.addWidget(self.save_button)
        self.main_layout.addLayout(layout)
        self.main_layout.addStretch()

        self.setLayout(self.main_layout)
        self.setWindowTitle('Gradients Entry')
        self.resize(400, 200)

        # Load saved gradient entries
        self.load_gradient_entries()

        # Update hardware
        self.update_hw_config_gradients()

    def add_input(self, label="", value="", tip=""):
        self.labels.append(label)
        self.values.append(value)
        self.tips.append(tip)

    def update_hw_config_gradients(self):
        hw.gFactor = np.array([float(self.input_boxes["Gx max (mT/m)"].text()),
                               float(self.input_boxes["Gy max (mT/m)"].text()),
                               float(self.input_boxes["Gz max (mT/m)"].text())]) * 1e-3  # T/m
        hw.max_grad = np.min(hw.gFactor) * 1e3
        hw.max_slew_rate = float(self.input_boxes["Max slew rate (mT/m/ms)"].text())
        hw.grad_raster_time = float(self.input_boxes["Gradient raster time (us)"].text()) * 1e-6  # s
        hw.grad_rise_time = float(self.input_boxes["Gradient rise time (us)"].text()) * 1e-6  # s
        hw.grad_steps = int(self.input_boxes["Gradient steps"].text())
        hw.gradDelay = int(self.input_boxes["Gradient delay (us)"].text())
        hw.gpa_model = self.input_boxes["GPA model"].text()

    def save_gradient_entries(self):
        file_name = "configs/hw_gradients.csv"

        # ✅ Ensure the 'configs' directory exists
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "Value"])
            for label, input_box in self.input_boxes.items():
                writer.writerow([label, input_box.text()])
        self.update_hw_config_gradients()
        print(f"Data saved for gradients.")

    def load_gradient_entries(self):
        """Load label-value pairs from a CSV file and update the input fields."""
        file_path = "configs/hw_gradients.csv"
        try:
            with open(file_path, mode="r", newline="") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row
                for row in reader:
                    if len(row) == 2:
                        label, value = row
                        if label in self.input_boxes:
                            self.input_boxes[label].setText(value)
        except:
            print("No hardware configuration loaded for gradients.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = GradientsWidget()
    widget.show()
    sys.exit(app.exec())
