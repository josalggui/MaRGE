import os
import sys
import csv
from configs import hw_config as hw
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton,
    QHBoxLayout, QLabel
)


class RfWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        self.main_layout = QVBoxLayout()
        self.dynamic_container = QVBoxLayout()
        self.main_layout.addLayout(self.dynamic_container)

        # Input field for RF coil name
        self.text_box = QLineEdit(self)
        self.text_box.setPlaceholderText("RF coil name")

        # Buttons for adding, deleting, and saving RF coils
        self.add_button = QPushButton('Add', self)
        self.add_button.clicked.connect(self.add_rf_coil)

        self.delete_button = QPushButton('Delete', self)
        self.delete_button.clicked.connect(self.delete_rf_coil)

        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save_to_csv)

        # Layout for input field and buttons
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.text_box)
        input_layout.addWidget(self.add_button)
        input_layout.addWidget(self.delete_button)
        input_layout.addWidget(self.save_button)

        self.main_layout.addLayout(input_layout)
        self.rf_entries = []

        # Load existing RF coils from CSV
        self.load_rf_coils()

        # Update hardware configuration dictionary
        self.update_hw_config_rf()

        self.setLayout(self.main_layout)
        self.setWindowTitle('RF Coil Entry')
        self.resize(400, 200)

    def update_hw_config_rf(self):
        """Updates the hardware configuration dictionary with RF coil values."""
        for label, text_box, _ in self.rf_entries:
            rf_name = label.text().replace("RF coil efficiency (rad / us / unit amplitude) ", "").strip()
            rf_value = float(text_box.text().strip())
            hw.antenna_dict[rf_name] = rf_value

    def add_rf_coil(self, rf_name="", rf_value=""):
        """Adds a new RF coil entry to the UI."""
        text = rf_name if rf_name else self.text_box.text().strip()
        if text:
            # Check if the RF coil already exists
            for label, _, _ in self.rf_entries:
                if label.text() == f"RF coil efficiency (rad / us / unit amplitude) {text}":
                    print(f"ERROR: RF Coil '{text}' already exists!")
                    return

            # Create new layout for the RF coil entry
            new_layout = QHBoxLayout()
            label = QLabel(f"RF coil efficiency (rad / us / unit amplitude) {text}", self)
            text_box = QLineEdit(self)
            text_box.setText(rf_value)

            new_layout.addWidget(label)
            new_layout.addWidget(text_box)

            self.dynamic_container.addLayout(new_layout)
            self.rf_entries.append((label, text_box, new_layout))

            self.text_box.clear()
        else:
            print("ERROR: Please enter an RF coil name!")

    def delete_rf_coil(self):
        """Deletes an RF coil entry from the UI."""
        text = self.text_box.text().strip()
        if not text:
            print("ERROR: Enter an RF coil name to delete!")
            return

        for i, (label, text_box, layout) in enumerate(self.rf_entries):
            if label.text() == f"RF coil efficiency (rad / us / unit amplitude) {text}":
                label.deleteLater()
                text_box.deleteLater()

                while layout.count():
                    item = layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()

                self.dynamic_container.removeItem(layout)
                del self.rf_entries[i]

                print(f"RF Coil '{text}' deleted!")
                return

        print("ERROR: RF coil not found!")

    def save_to_csv(self):
        """Saves the current RF coil entries to a CSV file."""
        if not self.rf_entries:
            print("ERROR: No RF coils to save!")
            return

        filename = os.path.abspath("../configs/rf_coils.csv")
        try:
            with open(filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["RF Coil Name", "Value"])

                for label, text_box, _ in self.rf_entries:
                    rf_name = label.text().replace("RF coil efficiency (rad / us / unit amplitude) ", "").strip()
                    rf_value = text_box.text().strip()
                    writer.writerow([rf_name, rf_value])

            print("RF data saved successfully!")
        except Exception as e:
            print(f"ERROR: Failed to save data: {e}")

    def load_rf_coils(self):
        """Loads RF coil entries from a CSV file."""
        filename = os.path.abspath("../configs/rf_coils.csv")
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
                print(f"ERROR: Failed to load RF data: {e}")


if __name__ == '__main__':
    # Run the application
    app = QApplication(sys.argv)
    widget = RfWidget()
    widget.show()
    sys.exit(app.exec())