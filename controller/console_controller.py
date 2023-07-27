import sys
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from widgets.console_widget import ConsoleWidget


class ConsoleController(ConsoleWidget):
    """
    Controller class for the console widget.

    Inherits from ConsoleWidget.
    """

    def __init__(self):
        super().__init__()

        # Redirect the output of print to the console widget
        sys.stdout = EmittingStream(textWritten=self.write_console)

    def write_console(self, text):
        """
        Write text to the console widget.

        Args:
            text (str): The text to be written to the console widget.
        """
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()


class EmittingStream(QObject):
    """
    Custom stream class that emits a signal whenever text is written.

    Inherits from QObject.
    """

    textWritten = pyqtSignal(str)

    def write(self, text):
        """
        Write text to the stream.

        Args:
            text (str): The text to be written to the stream.
        """
        self.textWritten.emit(str(text))

    @pyqtSlot()
    def flush(self):
        """
        Flush the stream.
        """
        pass
