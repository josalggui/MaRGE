import sys
import csv
import numpy as np
from configs import hw_config as hw
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton,
    QHBoxLayout, QLabel
)

class RfInOutWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        self.main_layout = QVBoxLayout()
        self.layout = QVBoxLayout()
        self.main_layout.addLayout(self.layout)

        # Labels and Boxes lists
        labels = ["RF de-blanking time (us)",  # RF
                  "RF dead time (us)",
                  "Larmor frequency (MHz)",
                  "Reference time (us)",
                  "Oversampling factor",  # ADC
                  "Max readout points",
                  "Add readout points",
                  "LNA gain (dB)",
                  "RF gain min (dB)",
                  "RF gain max (dB)",
                  ]
        values = ["15",
                  "400",
                  "3.066",
                  "70",
                  "5",
                  "1e5",
                  "5",
                  "45",
                  "45",
                  "76",
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
        self.save_button.clicked.connect(self.save_rf_in_out_entries)

        layout = QHBoxLayout()
        layout.addWidget(self.save_button)
        self.main_layout.addStretch()
        self.main_layout.addLayout(layout)

        self.setLayout(self.main_layout)
        self.setWindowTitle('RF in/out Entry')
        self.resize(400, 200)

        # Load saved gradient entries
        self.load_rf_in_out_entries()

        # Update hardware
        self.update_hw_config_rp()

    def update_hw_config_rp(self):
        hw.blkTime = float(self.input_boxes["RF de-blanking time (us)"].text())
        hw.deadTime = float(self.input_boxes["RF dead time (us)"].text())
        hw.larmorFreq = float(self.input_boxes["Larmor frequency (MHz)"].text())
        hw.oversamplingFactor = int(self.input_boxes["RF dead time (us)"].text())
        hw.maxRdPoints = int(float(self.input_boxes["Max readout points"].text()))
        hw.addRdPoints = int(self.input_boxes["Add readout points"].text())
        hw.reference_time = float(self.input_boxes["Reference time (us)"].text())
        hw.lnaGain = float(self.input_boxes["LNA gain (dB)"].text())
        hw.rf_min_gain = float(self.input_boxes["RF gain min (dB)"].text())
        hw.rf_max_gain = float(self.input_boxes["RF gain max (dB)"].text())

    def save_rf_in_out_entries(self):
        file_name = "../configs/hw_rf_in_out.csv"
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "Value"])
            for label, input_box in self.input_boxes.items():
                writer.writerow([label, input_box.text()])  # Write each pair
        print(f"Data saved for gradients entries")

    def load_rf_in_out_entries(self):
        """Load label-value pairs from a CSV file and update the input fields."""
        file_path = "../configs/hw_rf_in_out.csv"
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
            print("No hardware configuration loaded for rf input and outputs.")


if __name__ == '__main__':
    # Run the application
    app = QApplication(sys.argv)
    widget = RfInOutWidget()
    widget.show()
    sys.exit(app.exec())