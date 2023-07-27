from PyQt5.QtWidgets import QToolBar, QPushButton


class ToolBarWidget(QToolBar):
    """
    ToolBarWidget class for displaying a toolbar with buttons.

    Inherits from QToolBar provided by PyQt5 to display a toolbar with buttons.

    Attributes:
        main: The parent widget.
    """

    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the ToolBarWidget.

        Args:
            parent: The parent widget.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(ToolBarWidget, self).__init__(*args, **kwargs)
        self.main = parent

        self.image_loading_button = QPushButton('File')
        self.addWidget(self.image_loading_button)
