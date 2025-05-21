import datetime
import sys
import os

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from widgets.widget_console import ConsoleWidget

import atexit

class ConsoleController(ConsoleWidget):
    def __init__(self):
        super().__init__()

        # Créer dossier logs si besoin
        log_folder = "logs"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
            print(f"[ConsoleController] Created log folder at: {log_folder}")
        else:
            print(f"[ConsoleController] Log folder already exists at: {log_folder}")

        # Charger ou créer un nom de fichier log unique pour cette session
        session_log_file = os.path.join(log_folder, "current_session_log.txt")
        if os.path.exists(session_log_file):
            with open(session_log_file, "r", encoding="utf-8") as f:
                self.log_filename = f.read().strip()
        else:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_filename = os.path.join(log_folder, f"log_{now}.txt")
            with open(session_log_file, "w", encoding="utf-8") as f:
                f.write(self.log_filename)

        # Ouvrir le fichier log
        try:
            self.log_file = open(self.log_filename, "a", encoding="utf-8")
            print(f"[ConsoleController] Logging to file: {self.log_filename}")
        except Exception as e:
            self.log_file = None
            print(f"[ConsoleController] Failed to open log file: {e}")

        # Rediriger stdout
        sys.stdout = EmittingStream(textWritten=self.write_console)

        print("READY - MaRGE has started successfully.")
        print("WARNING - This is a test warning message.")
        print("ERROR - This is a test error message.")
        atexit.register(self.close_log)

    def write_console(self, text):
        # Console GUI
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

        # Nettoyage HTML pour le log
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


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    @pyqtSlot()
    def flush(self):
        pass
