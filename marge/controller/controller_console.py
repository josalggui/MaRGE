import datetime
import sys
import os
import atexit

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from marge.widgets.widget_console import ConsoleWidget


class ConsoleController(ConsoleWidget):
    def __init__(self, log_name=None):
        super().__init__()

        # Créer le dossier logs si besoin
        log_folder = "logs"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        # Créer un nouveau nom de fichier log à chaque démarrage
        if log_name is None:
            pass
        else:
            self.log_filename = os.path.join(log_folder, log_name)

        # Ouvrir le fichier log
        try:
            self.log_file = open(self.log_filename, "a", encoding="utf-8")
            print(f"[ConsoleController] Logging to file: {self.log_filename}")
        except Exception as e:
            self.log_file = None
            # print(f"[ConsoleController] Failed to open log file: {e}")

        # Rediriger stdout
        sys.stdout = EmittingStream(textWritten=self.write_console)

        print("READY - MaRGE has started successfully.")
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
