import os
import sys
import csv
from marge.configs import hw_config as hw
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton,
    QHBoxLayout, QLabel, QGridLayout
)


class ConsoleWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        self.main_layout = QVBoxLayout()
        self.dynamic_container = QVBoxLayout()  # Store reference
        self.layout = QGridLayout()

        # Parameters to save inputs
        self.labels = []
        self.values = []
        self.tips = []

        # Add inputs
        self.add_input(label="Red Pitaya model", value="rp-122", tip="Model of the Red Pitaya board")
        self.add_input(label="Maximum input voltage (mV)", value="225",
                       tip="Maximum voltage that can be input to the system")
        self.add_input(label="Gradient board model", value="gpa-fhdo", tip="Model of the gradient controller board")
        self.add_input(label="Clock frequency (MHz)", value="122.88", tip="Clock frequency of the system")
        self.add_input(label="ADC factor (mV/unit)", value="13.788", tip="ADC conversion factor from unit to mV")
        self.add_input(label="CIC delay points", value="3", tip="Number of delay points in the CIC filter")

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

        # Input field for RP ip address
        self.text_box = QLineEdit(self)
        self.text_box.setPlaceholderText("Red Pitaya ip address")

        # Buttons
        self.add_button = QPushButton('Add', self)
        self.add_button.clicked.connect(self.add_rp)

        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save_rp_entries)

        # self.layout.addLayout(self.dynamic_container)

        layout = QHBoxLayout()
        layout.addWidget(self.text_box)
        layout.addWidget(self.add_button)
        layout.addWidget(self.save_button)
        self.main_layout.addLayout(self.layout)
        self.main_layout.addLayout(self.dynamic_container)
        self.main_layout.addLayout(layout)
        self.main_layout.addStretch()

        self.setLayout(self.main_layout)
        self.setWindowTitle('Red Pitaya Entry')
        self.resize(400, 200)

        # Counter for IDs
        self.rp_counter = 1

        # Store added RPs in a list
        self.added_rps = []

        # Load saved RP entries
        self.load_rp_entries()

        # Update hardware
        self.update_hw_config_rp()

    def add_input(self, label="", value="", tip=""):
        self.labels.append(label)
        self.values.append(value)
        self.tips.append(tip)

    def update_hw_config_rp(self):
        hw.rp_ip_list = []
        hw.rp_port = []
        for _, ip in self.added_rps:
            hw.rp_ip_list.append(ip)
            hw.rp_port.append(11111)
        hw.rp_version = self.input_boxes["Red Pitaya model"].text()
        hw.rp_max_input_voltage = float(self.input_boxes["Maximum input voltage (mV)"].text())
        hw.grad_board = self.input_boxes["Gradient board model"].text()
        hw.fpga_clk_freq_MHz = float(self.input_boxes["Clock frequency (MHz)"].text())
        hw.adcFactor = float(self.input_boxes["ADC factor (mV/unit)"].text())
        hw.cic_delay_points = int(self.input_boxes["CIC delay points"].text())
        hw.rp_ip_address = "192.168.1.101"

    def add_rp(self):
        text = self.text_box.text().strip()
        if text:
            row_layout = QHBoxLayout()
            identifier_label = QLabel(f"RP-{self.rp_counter}")  # Unique identifier
            text_label = QLabel(text)  # User input

            # Add Delete button for this RP entry
            delete_button = QPushButton('Delete')
            delete_button.clicked.connect(lambda: self.delete_rp(row_layout))

            row_layout.addWidget(identifier_label)
            row_layout.addWidget(text_label)
            row_layout.addWidget(delete_button)

            self.dynamic_container.addLayout(row_layout)  # Add to container

            # Add the RP to the list for saving/deleting purposes
            self.added_rps.append((self.rp_counter, text))

            self.rp_counter += 1  # Increment counter
            self.text_box.clear()  # Clear input field

    def delete_rp(self, row_layout):
        # Remove the RP entry from the container and list
        for i, (rp_id, _) in enumerate(self.added_rps):
            if row_layout.itemAt(0).widget().text() == f"RP-{rp_id}":
                self.added_rps.pop(i)
                break

        # Remove layout (delete UI elements)
        for i in reversed(range(row_layout.count())):
            widget = row_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        self.dynamic_container.update()

    def save_rp_entries(self):
        file_name = "configs/hw_redpitayas.csv"
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "Value"])
            for label, input_box in self.input_boxes.items():
                writer.writerow([label, input_box.text()])  # Write each pair
            for rp_id, ip in self.added_rps:
                writer.writerow([f"RP-{rp_id}", ip])
        self.update_hw_config_rp()
        print(f"Data saved for red pitayas")

    def load_rp_entries(self):
        file_name = "configs/hw_redpitayas.csv"
        if os.path.exists(file_name):
            with open(file_name, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header row
                for row in reader:
                    label, value = row
                    if label in self.input_boxes:
                        self.input_boxes[label].setText(value)
                    else:
                        self.added_rps.append((label.split('-')[1], value))
                        self.add_rp_from_data(label.split('-')[1], value)
        else:
            print("No hardware configuration loaded for red pitayas.")

    def add_rp_from_data(self, rp_id, ip):
        row_layout = QHBoxLayout()
        identifier_label = QLabel(f"RP-{rp_id}")
        text_label = QLabel(ip)

        # Add Delete button for this RP entry
        delete_button = QPushButton('Delete')
        delete_button.clicked.connect(lambda: self.delete_rp(row_layout))

        row_layout.addWidget(identifier_label)
        row_layout.addWidget(text_label)
        row_layout.addWidget(delete_button)

        self.dynamic_container.addLayout(row_layout)  # Add to container


if __name__ == '__main__':
    # Run the application
    app = QApplication(sys.argv)
    widget = ConsoleWidget()
    widget.show()
    sys.exit(app.exec())