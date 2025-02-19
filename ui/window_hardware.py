import sys
import csv
import qdarkstyle
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QSizePolicy, QToolBar, QApplication, QWidget, QGridLayout, QHBoxLayout, QLabel, \
    QComboBox, QLineEdit, QAction, QVBoxLayout
from configs import hw_config as hw
import numpy as np


class HardwareWindow(QMainWindow):

    def __init__(self):
        super(HardwareWindow, self).__init__()
        self.setWindowTitle("Scanner Window")
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        self.setMinimumWidth(400)
        self.setMaximumHeight(400)

        # Set stylesheet
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        toolbar = QToolBar("Scanner toolbar")
        self.addToolBar(toolbar)

        # launch gui action
        save_action = QAction(QIcon("resources/icons/saveParameters.png"), "Save parameters", self)
        save_action.setStatusTip("Save parameters")
        save_action.triggered.connect(self.save_to_csv)
        toolbar.addAction(save_action)

        # Create central widget that will contain the layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Add layout to input parameters
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Labels 1
        labels_1 = ["Scanner name",
                    "Gx max (mT/m)",  # Gradients
                    "Gy max (mT/m)",
                    "Gz max (mT/m)",
                    "Max slew rate (mT/m/ms)",
                    "Gradient raster time (us)",
                    "Gradient rise time (us)",
                    "Gradient steps",
                    "Gradient delay (us)",
                    "RF de-blanking time (us)",  # RF
                    "RF dead time (us)",
                    "Larmor frequency (MHz)",
                    "Reference time (us)",
                    "Oversampling factor",  # ADC
                    "Max readout points",
                    "Add readout points",
                    "FOVx (cm)",  # FOV
                    "FOVy (cm)",
                    "FOVz (cm)",
                    "Shimming factor",  # shimming
                    "Bash path",  # others
                    "LNA gain (dB)",
                    "RF gain min (dB)",
                    "RF gain max (dB)",
                    "Arduino autotuning",
                    "Arduino interlock",
                    "Arduino attenuator",
                    ]
        values = ["Demo",
                  "50",
                  "80",
                  "70",
                  "80",
                  "50",
                  "400",
                  "16",
                  "9",
                  "15",
                  "400",
                  "3.066",
                  "70",
                  "5",
                  "1e5",
                  "5",
                  "20.0",
                  "20.0",
                  "20.0",
                  "1e-5",
                  "gnome-terminal",
                  "45",
                  "45",
                  "76",
                  "242353133363518050E0",
                  "242353133363518050E1",
                  "242353133363518050E2",
                  ]

        # Dictionary to store references to input fields
        self.input_boxes = {}

        # Create blocks iteratively
        for label_1, value in zip(labels_1, values):
            # Create a horizontal layout for each row
            row_layout = QHBoxLayout()

            # First label, box with value and second label
            label_widget_1 = QLabel(label_1)
            input_box = QLineEdit(value)

            # Store reference to input box
            self.input_boxes[label_1] = input_box  # Store it using the label as a key

            # Add widgets to row layout
            row_layout.addWidget(label_widget_1)
            row_layout.addWidget(input_box)

            # Add row layout to main layout
            main_layout.addLayout(row_layout)

        # Load data from csv
        try:
            self.load_from_csv()
        except:
            print("WARNING: no hardware configuration found")
            print("Default values will be used.")

        # Update hw_config
        self.update_hw_config()

        self.show()

    def update_hw_config(self):
        hw.gFactor = np.array([float(self.input_boxes["Gx max (mT/m)"].text()),
                               float(self.input_boxes["Gy max (mT/m)"].text()),
                               float(self.input_boxes["Gz max (mT/m)"].text())]) * 1e-3  # T/m
        hw.max_grad = np.min(hw.gFactor) * 1e3
        hw.max_slew_rate = float(self.input_boxes["Max slew rate (mT/m/ms)"].text())
        hw.grad_raster_time = float(self.input_boxes["Gradient raster time (us)"].text()) * 1e-6  # s
        hw.grad_rise_time = float(self.input_boxes["Gradient rise time (us)"].text()) * 1e-6  # s
        hw.grad_steps = int(self.input_boxes["Gradient steps"].text())
        hw.gradDelay = int(self.input_boxes["Gradient delay (us)"].text())
        hw.blkTime = float(self.input_boxes["RF de-blanking time (us)"].text())
        hw.deadTime = float(self.input_boxes["RF dead time (us)"].text())
        hw.larmorFreq = float(self.input_boxes["Larmor frequency (MHz)"].text())
        hw.oversamplingFactor = int(self.input_boxes["RF dead time (us)"].text())
        hw.maxRdPoints = int(float(self.input_boxes["Max readout points"].text()))
        hw.addRdPoints = int(self.input_boxes["Add readout points"].text())
        hw.reference_time = float(self.input_boxes["Reference time (us)"].text())
        hw.fov = [float(self.input_boxes["FOVx (cm)"].text()),
                  float(self.input_boxes["FOVy (cm)"].text()),
                  float(self.input_boxes["FOVz (cm)"].text())]
        hw.shimming_factor = float(self.input_boxes["Shimming factor"].text())
        hw.scanner_name = self.input_boxes["Scanner name"].text()
        hw.bash_path = self.input_boxes["Bash path"].text()
        hw.lnaGain = float(self.input_boxes["LNA gain (dB)"].text())
        hw.rf_min_gain = float(self.input_boxes["RF gain min (dB)"].text())
        hw.rf_max_gain = float(self.input_boxes["RF gain max (dB)"].text())
        hw.ard_sn_interlock = self.input_boxes["Arduino interlock"].text()
        hw.ard_sn_attenuator = self.input_boxes["Arduino attenuator"].text()
        hw.ard_sn_autotuning = self.input_boxes["Arduino autotuning"].text()



    def save_to_csv(self):
        """Save label-value pairs to a CSV file."""
        file_path = "../configs/hw_config.csv"
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Label", "Value"])  # Write header

            for label, input_box in self.input_boxes.items():
                writer.writerow([label, input_box.text()])  # Write each pair

        print(f"Hardware configuration saved.")

    def load_from_csv(self):
        """Load label-value pairs from a CSV file and update the input fields."""
        file_path = "../configs/hw_config.csv"
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
    # Only one QApplication for event loop
    app = QApplication(sys.argv)

    # Instantiate the window
    window = HardwareWindow()

    # Start the event loop.
    app.exec()