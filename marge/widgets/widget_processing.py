from PyQt5.QtWidgets import QTabWidget

from marge.controller.controller_post import PostProcessingTabController
from controller.controller_preprocessing import PreProcessingTabController
from marge.controller.controller_reconstruction import ReconstructionTabController
from marge.controller.controller_visualisation import VisualisationTabController


class ProcessingWidget(QTabWidget):
    """
    TabWidget class for displaying a tab widget with different tabs.

    Inherits from QTabWidget provided by PyQt5 to display a tab widget with different tabs.

    Attributes:
        main: The parent widget.
    """

    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the TabWidget.

        Args:
            parent: The parent widget.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(ProcessingWidget, self).__init__(*args, **kwargs)
        self.main = parent

        self.setMaximumWidth(400)

        # Tabs
        self.main.preprocessing_controller = PreProcessingTabController(self.main)
        self.main.reconstruction_controller = ReconstructionTabController(self.main)
        self.main.postprocessing_controller = PostProcessingTabController(self.main)
        self.main.visualisation_controller = VisualisationTabController(self.main)

        # Adding Tabs in the QTabWidget
        self.addTab(self.main.preprocessing_controller, 'Pre-Processing')
        self.addTab(self.main.reconstruction_controller, 'Reconstruction')
        self.addTab(self.main.postprocessing_controller, 'Post-Processing')
        self.addTab(self.main.visualisation_controller, 'Visualisation')
