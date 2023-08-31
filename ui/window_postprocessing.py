import sys
import qdarkstyle
from PyQt5.QtCore import QThreadPool

from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QApplication, QStatusBar, QMenuBar, \
    QSplitter, QTextEdit

from controller.reconstruction_tab_controller import ReconstructionTabController
from controller.postpocessing_tab_controller import PostProcessingTabController
from controller.visualisation_tab_controller import VisualisationTabController
from controller.preprocessing_tab_controller import PreProcessingTabController
# from controller.history_list_controller import HistoryListController
from controller.controller_history_list import HistoryListControllerPos
from controller.imageview_controller import ImageViewController
from controller.controller_console import ConsoleControllerPost
from controller.toolbar_controller import ToolBarController
from widgets.history_list_widget import HistoryListWidget
from controller.tab_controller import TabController


class MainWindow(QMainWindow):
    """
    Main window class for the application.

    Inherits from QMainWindow to provide the main application window.

    Attributes:
        threadpool (QThreadPool): Thread pool for parallel running.
        main_widget (QWidget): Central widget for the main window.
        main_layout (QHBoxLayout): Main layout for the main window.
        left_layout (QVBoxLayout): Layout for the left side of the main window.
        right_layout (QVBoxLayout): Layout for the right side of the main window.
        image_view_layout (QVBoxLayout): Layout for the image view widgets.
        image_view_splitter (QSplitter): Splitter widget for the image view widgets.
        image_view_widget (ImageViewController): Image view widget for displaying images.
        history_layout (QHBoxLayout): Layout for the history list and history controller.
        history_controller (HistoryListController): Controller for the history list.
        history_widget (HistoryListWidget): Widget for displaying the history list.
        toolbar_controller (ToolBarController): Controller for the toolbar.
        tab_controller (TabController): Controller for the tab widget.
        postprocessing_controller (PostProcessingTabController): Controller for the post-processing tab.
        preprocessing_controller (PreProcessingTabController): Controller for the pre-processing tab.
        reconstruction_controller (ReconstructionTabController): Controller for the reconstruction tab.
        visualisation_controller (VisualisationTabController): Controller for the visualisation tab.
        console (ConsoleController): Controller for the console.
    """

    def __init__(self):
        """
        Initialize the MainWindow.

        Args:
            None
        """
        super().__init__()

        # Set stylesheet
        self.styleSheet = qdarkstyle.load_stylesheet_pyqt5()
        self.setStyleSheet(self.styleSheet)

        # Set window parameters
        self.setWindowTitle('Test')
        self.setGeometry(0, 0, 1100, 800)

        # Threadpool for parallel running
        self.threadpool = QThreadPool()

        # Status bar and a menu bar adding
        self.setStatusBar(QStatusBar(self))
        self.setMenuBar(QMenuBar(self))

        # Set the central widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        # Set layouts
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)

        self.left_layout = QVBoxLayout()
        self.main_layout.addLayout(self.left_layout)

        self.right_layout = QVBoxLayout()
        self.main_layout.addLayout(self.right_layout)

        self.image_view_layout = QHBoxLayout()
        self.right_layout.addLayout(self.image_view_layout)

        self.image_view_splitter = QSplitter()
        self.image_view_layout.addWidget(self.image_view_splitter)

        # Image view addition
        self.image_view_widget = ImageViewController(parent=self)
        self.image_view_splitter.addWidget(self.image_view_widget)

        # Layout for output history
        self.output_layout_h = QHBoxLayout()
        self.right_layout.addLayout(self.output_layout_h)

        # Add list to show the history
        self.history_list = HistoryListControllerPos(parent=self)
        self.output_layout_h.addWidget(self.history_list)
        self.history_list.setMaximumHeight(200)
        self.history_list.setMinimumHeight(200)

        # Table with list of applied methods to selected item in the history
        self.methods_list = QTextEdit()
        self.methods_list.setMaximumHeight(200)
        self.methods_list.setMinimumHeight(200)
        self.output_layout_h.addWidget(self.methods_list)

        # Toolbar addition
        self.toolbar_controller = ToolBarController(parent=self)
        self.addToolBar(self.toolbar_controller)

        # Tab addition
        self.tab_controller = TabController(parent=self)
        self.left_layout.addWidget(self.tab_controller)

        self.postprocessing_controller = PostProcessingTabController(parent=self)
        self.tab_controller.postprocessing_layout.addWidget(self.postprocessing_controller)

        self.preprocessing_controller = PreProcessingTabController(parent=self)
        self.tab_controller.preprocessing_layout.addWidget(self.preprocessing_controller)

        self.reconstruction_controller = ReconstructionTabController(parent=self)
        self.tab_controller.reconstruction_layout.addWidget(self.reconstruction_controller)

        self.visualisation_controller = VisualisationTabController(parent=self)
        self.tab_controller.visualisation_layout.addWidget(self.visualisation_controller)

        # Console addition
        self.console = ConsoleControllerPost()
        self.left_layout.addWidget(self.console)
        self.console.setMaximumHeight(200)

        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    ui = MainWindow()

    sys.exit(app.exec())
