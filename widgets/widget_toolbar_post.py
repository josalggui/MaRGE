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
        self.action_load = QAction(QIcon("resources/icons/addSequence.png"), "Load raw-data .mat", self)
        self.action_load.setStatusTip("Open new raw-data .mat")
        self.addAction(self.action_load)

        self.action_loadrmd = QAction(QIcon("resources/icons/addSequence.png"), "Load raw-data .h5", self)
        self.action_loadrmd.setStatusTip("Open new raw-data .h5")
        self.addAction(self.action_loadrmd)
       
        self.action_printrmd = QAction(QIcon("resources/icons/tableau_rmd.png"), "Show ISMRMRD data", self)
        self.action_printrmd.setStatusTip("Show ISMRMRD data")
        self.addAction(self.action_printrmd)
        
        self.action_convert = QAction(QIcon("resources/icons/convert.png"), "Converter .mat to .h5", self)
        self.action_convert.setStatusTip("Converter .mat to .h5")
        self.addAction(self.action_convert)
        
