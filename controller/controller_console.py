"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import datetime
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

        # Get the current time and format it
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        current_time = "<b>" + current_time + "</b>"

        # Give format to the text
        if "ERROR" in text:
            text = f'<span style="color:red;"><b>ERROR</b></span>{text[5:]}'

        if "WARNING" in text:
            text = f'<span style="color:orange;"><b>WARNING</b></span>{text[7:]}'

        if "READY" in text:
            text = f'<span style="color:green;"><b>READY</b></span>{text[5:]}'

        # Prepend the time to the text
        if text == "\n" or text == " ":
            formatted_text = text
        else:
            formatted_text = f"{current_time} - {text}"
            formatted_text = formatted_text.replace("\n", "<br>")

        # Check if the text contains a marker for bold formatting
        if "<b>" in formatted_text and "</b>" in formatted_text:
            # Insert the text as HTML to allow formatting
            cursor.insertHtml(formatted_text)
        else:
            # Insert plain text
            cursor.insertText(formatted_text)

        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def clear_console(self):
        """
        Clear the console widget by removing all its contents.
        """
        self.console.clear()  # Clears the console content

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