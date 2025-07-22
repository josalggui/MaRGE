from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QToolBar, QAction
from importlib import resources

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


        with resources.path("marge.resources.icons", "addSequence.png") as path_add_seq:
            self.action_load = QAction(QIcon(str(path_add_seq)), "Load raw-data .mat", self)
        self.action_load.setStatusTip("Open new raw-data .mat")
        self.addAction(self.action_load)

        with resources.path("marge.resources.icons", "addSequence.png") as path_add_seq:
            self.action_loadrmd = QAction(QIcon(str(path_add_seq)), "Load raw-data .h5", self)
        self.action_loadrmd.setStatusTip("Open new raw-data .h5")
        self.addAction(self.action_loadrmd)

        with resources.path("marge.resources.icons", "tableau_rmd.png") as path_tableau_rmd:
            self.action_printrmd = QAction(QIcon(str(path_tableau_rmd)), "Show ISMRMRD data", self)
        self.action_printrmd.setStatusTip("Show ISMRMRD data")
        self.addAction(self.action_printrmd)

        with resources.path("marge.resources.icons", "convert.png") as path_convert:
            self.action_convert = QAction(QIcon(str(path_convert)), "Converter .mat to .h5", self)
        self.action_convert.setStatusTip("Converter .mat to .h5")
        self.addAction(self.action_convert)