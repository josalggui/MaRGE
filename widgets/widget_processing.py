from PyQt5.QtWidgets import QTabWidget, QWidget, QVBoxLayout

from controller.controller_postpocessing import PostProcessingTabController
from controller.controller_preprocessing import PreProcessingTabController
from controller.controller_reconstruction import ReconstructionTabController
from controller.controller_visualisation import VisualisationTabController


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
        preprocessing_tab = QWidget()
        reconstruction_tab = QWidget()
        postprocessing_tab = QWidget()
        visualisation_tab = QWidget()

        # Adding Tabs in the QTabWidget
        self.addTab(PreProcessingTabController(self.main), 'Pre-Processing')
        self.addTab(ReconstructionTabController(self.main), 'Reconstruction')
        self.addTab(PostProcessingTabController(self.main), 'Post-Processing')
        self.addTab(VisualisationTabController(self.main), 'Visualisation')

        # Preprocessing layout
        self.preprocessing_layout = QVBoxLayout(preprocessing_tab)

        # Reconstruction layout
        self.reconstruction_layout = QVBoxLayout(reconstruction_tab)

        # Postprocessing layout
        self.postprocessing_layout = QVBoxLayout(postprocessing_tab)

        # Visualisation layout
        self.visualisation_layout = QVBoxLayout(visualisation_tab)
