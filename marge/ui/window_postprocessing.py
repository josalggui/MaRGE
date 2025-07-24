import sys
import qdarkstyle
from PyQt5.QtCore import QThreadPool

from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QApplication, QStatusBar, QMenuBar, \
    QSplitter, QTextEdit, QSizePolicy

from marge.controller.controller_reconstruction import ReconstructionTabController
from marge.controller.controller_post import PostProcessingTabController
from marge.controller.controller_toolbar_figures import FiguresControllerPos
from marge.controller.controller_visualisation import VisualisationTabController
from marge.controller.controller_preprocessing import PreProcessingTabController
from marge.controller.controller_history_list import HistoryListControllerPos
from marge.controller.controller_imageview import ImageViewController
from marge.controller.controller_figures import FiguresLayoutController
from marge.controller.controller_console import ConsoleController
from marge.controller.controller_toolbar_post import ToolBarControllerPost, MainWindow_toolbar
from marge.controller.controller_processing import ProcessingController


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

    def __init__(self, session, main):
        """
        Initialize the MainWindow.

        Args:
            None
        """
        super().__init__()
        self.session = session
        self.main = main

        # Set stylesheet based on theme
        if self.session["black_theme"]:
            self.styleSheet = qdarkstyle.load_stylesheet_pyqt5()
        else:
            self.styleSheet = ""
        self.setStyleSheet(self.styleSheet)

        # Set window parameters
        self.setWindowTitle(session['directory'])

        # Threadpool for parallel running
        self.threadpool = QThreadPool()

        # Toolbar addition
        self.main_window = MainWindow_toolbar()
        self.toolbar_image = ToolBarControllerPost(self.main_window, self, "Image toolbar")
        self.addToolBar(self.toolbar_image)

        # Image toolbar
        self.toolbar_image_2 = FiguresControllerPos(self, "Image toolbar 2")
        self.addToolBar(self.toolbar_image_2)

        # Status bar and a menu bar adding
        self.setStatusBar(QStatusBar(self))
        self.setMenuBar(QMenuBar(self))

        # Set the central widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        # Set layouts
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)

        # Layout for processing and console
        self.left_layout = QVBoxLayout()
        self.main_layout.addLayout(self.left_layout)

        # Layout for figures and history list
        self.right_layout = QVBoxLayout()
        self.main_layout.addLayout(self.right_layout)

        # Tab addition
        self.tab_controller = ProcessingController(parent=self)
        self.left_layout.addWidget(self.tab_controller)

        # Image view addition
        self.image_view_widget = FiguresLayoutController(self)
        self.image_view_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_layout.addWidget(self.image_view_widget)

        # Layout for output history
        self.output_layout_h = QHBoxLayout()
        self.right_layout.addLayout(self.output_layout_h)

        # Add list to show the history
        self.history_list = HistoryListControllerPos(parent=self)
        self.history_list.setMaximumHeight(200)
        self.history_list.setMinimumHeight(200)
        self.output_layout_h.addWidget(self.history_list)

        # Table with list of applied methods to selected item in the history
        self.methods_list = QTextEdit()
        self.methods_list.setMaximumHeight(200)
        self.methods_list.setMinimumHeight(200)
        self.output_layout_h.addWidget(self.methods_list)



if __name__ == '__main__':
    app = QApplication(sys.argv)

    session = {'directory': '../test'}
    ui = MainWindow(session=session, main=None)
    ui.show()

    sys.exit(app.exec())
