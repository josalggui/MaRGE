"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import sys

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from widgets.widget_console import ConsoleWidget


class ConsoleController(ConsoleWidget):
    def __init__(self):
        super().__init__()

        # Redirect the output of print to the console widget
        sys.stdout = EmittingStream(textWritten=self.write_console)

    def write_console(self, text):
        """Write text to the console widget"""
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    @pyqtSlot()
    def flush(self):
        pass
