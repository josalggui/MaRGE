"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""

import sys
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from widgets.widget_console import ConsoleWidget


class ConsoleController(ConsoleWidget):
    """
    Console controller class.

    This class extends the `ConsoleWidget` class and serves as a controller for the console functionality. It redirects
    the output of print statements to the console widget.

    Methods:
        __init__(): Initialize the ConsoleController instance.
        write_console(text): Write text to the console widget.

    Signals:
        None
    """

    def __init__(self):
        super().__init__()

        # Redirect the output of print to the console widget
        sys.stdout = EmittingStream(textWritten=self.write_console)

    def write_console(self, text):
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()


class EmittingStream(QObject):
    """
    Emitting stream class.

    This class emits a signal with the text written and provides a write method to redirect the output.

    Methods:
        write(text): Write text and emit the signal.
        flush(): Placeholder method for flushing the stream.

    Signals:
        textWritten (str): A signal emitted with the text written.
    """

    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    @pyqtSlot()
    def flush(self):
        pass