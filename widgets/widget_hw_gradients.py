import sys
import csv
import numpy as np
from configs import hw_config as hw
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton,
    QHBoxLayout, QLabel
)

class GradientsWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        self.main_layout = QVBoxLayout()
        self.layout = QVBoxLayout()
        self.main_layout.addLayout(self.layout)

        # Labels and Boxes lists
        labels = ["Gx max (mT/m)",  # Gradients
                  "Gy max (mT/m)",
                  "Gz max (mT/m)",
                  "Max slew rate (mT/m/ms)",
                  "Gradient raster time (us)",
                  "Gradient rise time (us)",
                  "Gradient steps",
                  "Gradient delay (us)",
                  ]
        values = ["50",
                  "80",
                  "70",
                  "80",
                  "50",
                  "400",
                  "16",
                  "9",
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
        self.save_button.clicked.connect(self.save_gradient_entries)

        layout = QHBoxLayout()
        layout.addWidget(self.save_button)
        self.main_layout.addStretch()
        self.main_layout.addLayout(layout)

        self.setLayout(self.main_layout)
        self.setWindowTitle('Gradients Entry')
        self.resize(400, 200)

        # Load saved gradient entries
        self.load_gradient_entries()

        # Update hardware
        self.update_hw_config_rp()

    def update_hw_config_rp(self):
        hw.gFactor = np.array([float(self.input_boxes["Gx max (mT/m)"].text()),
                               float(self.input_boxes["Gy max (mT/m)"].text()),
                               float(self.input_boxes["Gz max (mT/m)"].text())]) * 1e-3  # T/m
        hw.max_grad = np.min(hw.gFactor) * 1e3
        hw.max_slew_rate = float(self.input_boxes["Max slew rate (mT/m/ms)"].text())
        hw.grad_raster_time = float(self.input_boxes["Gradient raster time (us)"].text()) * 1e-6  # s
        hw.grad_rise_time = float(self.input_boxes["Gradient rise time (us)"].text()) * 1e-6  # s
        hw.grad_steps = int(self.input_boxes["Gradient steps"].text())
        hw.gradDelay = int(self.input_boxes["Gradient delay (us)"].text())

    def save_gradient_entries(self):
        file_name = "../configs/hw_gradients.csv"
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "IP Address"])
            for label, input_box in self.input_boxes.items():
                writer.writerow([label, input_box.text()])  # Write each pair
        print(f"Data saved for gradients entries")

    def load_gradient_entries(self):
        """Load label-value pairs from a CSV file and update the input fields."""
        file_path = "../configs/hw_gradients.csv"
        with open(file_path, mode="r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row

            for row in reader:
                if len(row) == 2:  # Ensure row has two columns
                    label, value = row
                    if label in self.input_boxes:
                        self.input_boxes[label].setText(value)  # Update input box

            print(f"Hardware configuration loaded.")


if __name__ == '__main__':
    # Run the application
    app = QApplication(sys.argv)
    widget = GradientsWidget()
    widget.show()
    sys.exit(app.exec())