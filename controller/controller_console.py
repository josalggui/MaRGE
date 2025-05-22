import datetime
import sys
import os
import atexit

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit
)

class ConsoleController(QMainWindow):
    def __init__(self):
        super().__init__()

        # === LOG SYSTEM: GARDER INTACT ===
        log_folder = "logs"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
            print(f"[ConsoleController] Created log folder at: {log_folder}")
        else:
            print(f"[ConsoleController] Log folder already exists at: {log_folder}")

        session_log_file = os.path.join(log_folder, "current_session_log.txt")
        if os.path.exists(session_log_file):
            with open(session_log_file, "r", encoding="utf-8") as f:
                self.log_filename = f.read().strip()
        else:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_filename = os.path.join(log_folder, f"log_{now}.txt")
            with open(session_log_file, "w", encoding="utf-8") as f:
                f.write(self.log_filename)

        try:
            self.log_file = open(self.log_filename, "a", encoding="utf-8")
            print(f"[ConsoleController] Logging to file: {self.log_filename}")
        except Exception as e:
            self.log_file = None
            print(f"[ConsoleController] Failed to open log file: {e}")

        sys.stdout = EmittingStream(textWritten=self.write_console)
        atexit.register(self.close_log)

        # === GUI CENTRAL WIDGET ===
        central_widget = QWidget()
        layout = QVBoxLayout()

        # === CONSOLE ===
        self.console = QTextEdit()
        self.console.setReadOnly(True)

        # === BUTTON SWITCH THEME ===
        self.theme_button = QPushButton("Switch Theme")
        self.theme_button.clicked.connect(self.toggle_theme)
        self.dark_mode = True  # Start in dark mode

        # Add widgets to layout
        layout.addWidget(self.theme_button)
        layout.addWidget(self.console)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Apply initial dark theme style
        self.apply_dark_theme()

        # Initial console messages
        print("READY - MaRGE has started successfully.")
        print("WARNING - This is a test warning message.")
        print("ERROR - This is a test error message.")

    def write_console(self, text):
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.End)

        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        timestamp_html = f"<b>{current_time}</b>"

        if "ERROR" in text:
            text = f'<span style="color:red;"><b>ERROR</b></span>{text[5:]}'
        elif "WARNING" in text:
            text = f'<span style="color:orange;"><b>WARNING</b></span>{text[7:]}'
        elif "READY" in text:
            text = f'<span style="color:green;"><b>READY</b></span>{text[5:]}'

        if text.strip() == "":
            formatted_text = text
        else:
            formatted_text = f"{timestamp_html} - {text}".replace("\n", "<br>")

        if "<b>" in formatted_text and "</b>" in formatted_text:
            cursor.insertHtml(formatted_text)
        else:
            cursor.insertText(formatted_text)

        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

        # Clean text for log file
        clean_text = (
            text.replace("<br>", "\n")
                .replace("<b>", "")
                .replace("</b>", "")
                .replace('<span style="color:red;">', '')
                .replace('<span style="color:orange;">', '')
                .replace('<span style="color:green;">', '')
                .replace('</span>', '')
        )

        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            try:
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                self.log_file.write(f"{timestamp} - {clean_text.strip()}\n")
                self.log_file.flush()
            except Exception as e:
                print(f"[ConsoleController] Log write error: {e}")

    def clear_console(self):
        self.console.clear()

    def close_log(self):
        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            self.log_file.close()

    def toggle_theme(self):
        if self.dark_mode:
            self.apply_light_theme()
            self.dark_mode = False
        else:
            self.apply_dark_theme()
            self.dark_mode = True

    def apply_dark_theme(self):
        # Dark mode style for entire window and widgets
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3c3f41;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #505354;
            }
        """)

    def apply_light_theme(self):
        # Light mode style for entire window and widgets
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTextEdit {
                background-color: white;
                color: black;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: black;
                border: 1px solid #cccccc;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #d6d6d6;
            }
        """)

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    @pyqtSlot()
    def flush(self):
        pass

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = ConsoleController()
    window.show()
    sys.exit(app.exec_())
