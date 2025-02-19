import os
import sys
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QMessageBox,
    QHBoxLayout, QLabel
)


class RfWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the main layout
        self.main_layout = QVBoxLayout()

        # Container for dynamically added layouts
        self.dynamic_container = QVBoxLayout()
        self.main_layout.addLayout(self.dynamic_container)

        # Create input text box
        self.text_box = QLineEdit(self)
        self.text_box.setPlaceholderText("RF coil name")

        # Create buttons
        self.add_button = QPushButton('Add', self)
        self.add_button.clicked.connect(self.add_rf_coil)

        self.delete_button = QPushButton('Delete', self)
        self.delete_button.clicked.connect(self.delete_rf_coil)

        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save_to_csv)

        # Create horizontal layout for input and buttons
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.text_box)
        input_layout.addWidget(self.add_button)
        input_layout.addWidget(self.delete_button)
        input_layout.addWidget(self.save_button)

        # Add the input layout at the bottom
        self.main_layout.addLayout(input_layout)

        # Store references to dynamically added (label, text_box) pairs
        self.rf_entries = []

        # Load existing RF coils from CSV
        self.load_rf_coils()

        # Set the layout on the application's window
        self.setLayout(self.main_layout)
        self.setWindowTitle('RF Coil Entry')
        self.resize(400, 200)

    def add_rf_coil(self, rf_name="", rf_value=""):
        text = rf_name if rf_name else self.text_box.text().strip()  # Use argument or input
        if text:
            # Create new horizontal layout
            new_layout = QHBoxLayout()

            # Create label and text box
            label = QLabel(f"RF Coil: {text}", self)
            text_box = QLineEdit(self)
            text_box.setText(rf_value)

            # Add to layout
            new_layout.addWidget(label)
            new_layout.addWidget(text_box)

            # Add this layout to the dynamic container
            self.dynamic_container.insertLayout(0, new_layout)  # Insert at the top

            # Store the reference
            self.rf_entries.append((label, text_box, new_layout))

            # Clear the input box
            self.text_box.clear()
        else:
            print("ERROR: please enter an RF coil name!")

    def delete_rf_coil(self):
        text = self.text_box.text().strip()
        if not text:
            print("ERROR: Enter an RF coil name to delete!")
            return

        for i, (label, text_box, layout) in enumerate(self.rf_entries):
            if label.text() == f"RF Coil: {text}":
                # Remove widgets from layout
                label.deleteLater()
                text_box.deleteLater()

                # Remove the layout from the parent
                self.main_layout.removeItem(layout)

                # Remove from the list
                del self.rf_entries[i]

                print(f"RF coil '{text}' deleted!")
                return

        print("ERROR: RF coil not found!")

    def save_to_csv(self):
        if not self.rf_entries:
            print("ERROR: No RF coils to save!")
            return

        filename = "../configs/rf_coils.csv"
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["RF Coil Name", "Value"])

            for label, text_box in self.rf_entries:
                rf_name = label.text().replace("RF Coil: ", "").strip()
                rf_value = text_box.text().strip()
                writer.writerow([rf_name, rf_value])

        print("RF data saved")

    def load_rf_coils(self):
        filename = "../configs/rf_coils.csv"
        if os.path.exists(filename):
            try:
                with open(filename, mode="r", newline="") as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header row

                    for row in reader:
                        if len(row) == 2:
                            rf_name, rf_value = row
                            self.add_rf_coil(rf_name, rf_value)

            except Exception as e:
                print("ERROR: Failed to load rf data")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = RfWidget()
    widget.show()
    sys.exit(app.exec())
