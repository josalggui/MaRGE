import sys
import qdarkstyle
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QTabWidget, QStatusBar, QToolBar,
    QAction, QHBoxLayout, QGroupBox, QCheckBox
)
import importlib.resources as resources

from marge.controller.controller_console import ConsoleController
from marge.widgets.widget_hw_console import ConsoleWidget
from marge.widgets.widget_hw_others import OthersWidget
from marge.widgets.widget_hw_gradients import GradientsWidget
from marge.widgets.widget_hw_rf import RfWidget
from marge.widgets.widget_session import SessionWidget


class SessionWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Session and Hardware Window for MaRGE")
        self.setGeometry(100, 100, 600, 400)

        # Set initial stylesheet
        self.is_dark_theme = True
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        # Main layout
        widget_main = QWidget()
        layout_main = QHBoxLayout()
        widget_main.setLayout(layout_main)
        self.setCentralWidget(widget_main)
        layout_left = QVBoxLayout()
        layout_main.addLayout(layout_left)

        # Scanner checks
        checks = QGroupBox("Scanner checks")
        layout = QVBoxLayout()
        checks.setLayout(layout)
        layout_left.addWidget(checks)

        self.check_projects = QCheckBox("Projects")
        self.check_projects.setDisabled(True)
        layout.addWidget(self.check_projects)

        self.check_study = QCheckBox("Study cases")
        self.check_study.setDisabled(True)
        layout.addWidget(self.check_study)

        self.check_gradients = QCheckBox("Gradient hardware")
        self.check_gradients.setDisabled(True)
        layout.addWidget(self.check_gradients)

        self.check_redpitayas = QCheckBox("MaRCoS")
        self.check_redpitayas.setDisabled(True)
        layout.addWidget(self.check_redpitayas)

        self.check_rf = QCheckBox("RFs")
        self.check_rf.setDisabled(True)
        layout.addWidget(self.check_rf)

        self.check_rp_ips = QCheckBox("RP IPs")
        self.check_rp_ips.setDisabled(True)
        layout.addWidget(self.check_rp_ips)

        self.check_rf_coils = QCheckBox("RF coils")
        self.check_rf_coils.setDisabled(True)
        layout.addWidget(self.check_rf_coils)

        self.check_others = QCheckBox("Others")
        self.check_others.setDisabled(True)
        layout.addWidget(self.check_others)

        # Console
        self.console = ConsoleController()
        layout_left.addWidget(self.console)

        # Toolbar
        self.toolbar = QToolBar("Session toolbar")
        self.addToolBar(self.toolbar)

        with resources.path("marge.resources.icons", "home.png") as icon_path:
            self.launch_gui_action = QAction(QIcon(str(icon_path)), "Launch GUI", self)
        self.launch_gui_action.setStatusTip("Launch GUI")
        self.launch_gui_action.setDisabled(True)
        self.toolbar.addAction(self.launch_gui_action)

        with resources.path("marge.resources.icons", "demo.png") as icon_path:
            self.demo_gui_action = QAction(QIcon(str(icon_path)), "Launch GUI as DEMO", self)
        self.demo_gui_action.setStatusTip("Launch GUI as DEMO")
        self.demo_gui_action.setDisabled(True)
        self.toolbar.addAction(self.demo_gui_action)


        with resources.path("marge.resources.icons", "arrow-sync.svg") as icon_path:
            self.update_action = QAction(QIcon(str(icon_path)), "Update scanner hardware", self)
        self.update_action.setStatusTip("Update scanner hardware")
        self.toolbar.addAction(self.update_action)


        with resources.path("marge.resources.icons", "close.png") as icon_path:
            self.close_action = QAction(QIcon(str(icon_path)), "Close GUI", self)
        self.close_action.setStatusTip("Close the GUI")
        self.toolbar.addAction(self.close_action)

        # Add switch theme button
        with resources.path("marge.resources.icons", "adjust-contrast.svg") as icon_path:
            self.switch_theme_action = QAction(QIcon(str(icon_path)), "", self)
        self.switch_theme_action.setStatusTip("Switch between Dark and Light theme")
        self.toolbar.addAction(self.switch_theme_action)

        # Tabs
        self.tabs = QTabWidget()
        layout_main.addWidget(self.tabs)

        self.tab_session = SessionWidget()
        self.tab_console = ConsoleWidget()
        self.tab_gradients = GradientsWidget()
        self.tab_rf = RfWidget()
        self.tab_others = OthersWidget()

        self.tabs.addTab(self.tab_session, "Session")
        self.tabs.addTab(self.tab_console, "Console")
        self.tabs.addTab(self.tab_gradients, "Gradients")
        self.tabs.addTab(self.tab_rf, "RF")
        self.tabs.addTab(self.tab_others, "Others")

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
    app = QApplication(sys.argv)
    window = SessionWindow()
    window.show()
    app.exec()
