import sys
import qdarkstyle
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QTabWidget, QStatusBar
from widgets.widget_hw_console import ConsoleWidget
from widgets.widget_hw_others import OthersWidget
from widgets.widget_hw_gradients import GradientsWidget
from widgets.widget_hw_rf import RfWidget


class HardwareWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hardware Entry")
        self.setGeometry(100, 100, 600, 400)

        # Create a QTabWidget
        self.tabs = QTabWidget()

        # Set stylesheet
        self.styleSheet = qdarkstyle.load_stylesheet_pyqt5()
        self.setStyleSheet(self.styleSheet)

        # Create the individual tabs
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()

        # Add the tabs to the QTabWidget
        self.tabs.addTab(self.tab1, "Console")
        self.tabs.addTab(self.tab2, "Gradients")
        self.tabs.addTab(self.tab3, "RF")
        self.tabs.addTab(self.tab4, "Others")

        # Set up the layout for each tab
        self.setupTab1()
        self.setupTab2()
        self.setupTab3()
        self.setupTab4()

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
    window = HardwareWindow()
    window.show()

    # Start the event loop.
    app.exec()