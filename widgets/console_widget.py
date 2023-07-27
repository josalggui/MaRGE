from PyQt5.QtWidgets import QTextEdit, QMainWindow, QSizePolicy


class ConsoleWidget(QMainWindow):
    """
    Console widget class for displaying console output.

    Inherits from QMainWindow to provide a window for the console widget.

    Attributes:
        console (QTextEdit): Text edit widget for displaying the console output.
    """

    def __init__(self):
        """
        Initialize the ConsoleWidget.

        Args:
            None
        """
        super().__init__()

        # Create the console widget
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.setCentralWidget(self.console)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.setMaximumWidth(400)
