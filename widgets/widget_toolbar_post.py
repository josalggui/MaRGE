from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QToolBar, QAction


class ToolBarWidgetPost(QToolBar):
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
        super(ToolBarWidgetPost, self).__init__(*args, **kwargs)
        self.main = parent

        # Load data
        self.action_load = QAction(QIcon("resources/icons/addSequence.png"), "Load raw-data", self)
        self.action_load.setStatusTip("Open new raw-data")
        self.addAction(self.action_load)
