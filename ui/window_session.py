import sys
import qdarkstyle
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QTabWidget, QStatusBar, QToolBar, QAction
from widgets.widget_hw_console import ConsoleWidget
from widgets.widget_hw_others import OthersWidget
from widgets.widget_hw_gradients import GradientsWidget
from widgets.widget_hw_rf import RfWidget
from widgets.widget_session import SessionWidget


class SessionWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Session and Hardware Window for MaRGE")
        self.setGeometry(100, 100, 600, 400)

        # Set stylesheet
        self.styleSheet = qdarkstyle.load_stylesheet_pyqt5()
        self.setStyleSheet(self.styleSheet)

        # Add toolbar
        self.toolbar = QToolBar("Session toolbar")
        self.addToolBar(self.toolbar)

        # launch gui action
        self.launch_gui_action = QAction(QIcon("resources/icons/home.png"), "Launch GUI", self)
        self.launch_gui_action.setStatusTip("Launch GUI")
        self.toolbar.addAction(self.launch_gui_action)
        self.launch_gui_action.setDisabled(True)

        # demo gui action
        self.demo_gui_action = QAction(QIcon("resources/icons/demo.png"), "Launch GUI as DEMO", self)
        self.demo_gui_action.setStatusTip("Launch GUI as DEMO")
        self.toolbar.addAction(self.demo_gui_action)
        self.demo_gui_action.setDisabled(True)

        # demo gui action
        self.update_action = QAction(QIcon("resources/icons/arrow-sync.svg"), "Update scanner hardware",
                                             self)
        self.update_action.setStatusTip("Update scanner hardware")
        self.toolbar.addAction(self.update_action)

        # Close button action
        self.close_action = QAction(QIcon("resources/icons/close.png"), "Close GUI", self)
        self.close_action.setStatusTip("Close the GUI")
        self.toolbar.addAction(self.close_action)

        # Create a QTabWidget
        self.tabs = QTabWidget()

        # Create the individual tabs
        self.tab_session = SessionWidget()
        self.tab_console = ConsoleWidget()
        self.tab_gradients = GradientsWidget()
        self.tab_rf = RfWidget()
        self.tab_others = OthersWidget()

        # Add the tabs to the QTabWidget
        self.tabs.addTab(self.tab_session, "Session")
        self.tabs.addTab(self.tab_console, "Console")
        self.tabs.addTab(self.tab_gradients, "Gradients")
        self.tabs.addTab(self.tab_rf, "RF")
        self.tabs.addTab(self.tab_others, "Others")

        # Set the central widget of the main window to be the tabs
        self.setCentralWidget(self.tabs)

        # Status bar
        self.setStatusBar(QStatusBar(self))

    def setupTab1(self):
        widget = ConsoleWidget()
        layout = QVBoxLayout()
        layout.addWidget(widget)
        self.tab1.setLayout(layout)

    def setupTab2(self):
        widget = GradientsWidget()
        layout = QVBoxLayout()
        layout.addWidget(widget)
        self.tab2.setLayout(layout)

    def setupTab3(self):
        widget = RfWidget()
        layout = QVBoxLayout()
        layout.addWidget(widget)
        self.tab3.setLayout(layout)

    def setupTab4(self):
        widget = OthersWidget()
        layout = QVBoxLayout()
        layout.addWidget(widget)
        self.tab4.setLayout(layout)

if __name__ == '__main__':
    # Only one QApplication for event loop
    app = QApplication(sys.argv)

    # Instantiate the window
    window = SessionWindow()
    window.show()

    # Start the event loop.
    app.exec()