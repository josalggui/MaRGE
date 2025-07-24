"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation: MRILab, i3M, CSIC, Valencia, Spain
"""

import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QStatusBar, QWidget, QHBoxLayout, QVBoxLayout, QTableWidget,
    QSizePolicy
)
from PyQt5.QtCore import QThreadPool
import qdarkstyle

from marge.controller.controller_console import ConsoleController
from marge.controller.controller_figures import FiguresLayoutController
from marge.controller.controller_history_list import HistoryListController
from marge.controller.controller_menu import MenuController
from marge.controller.controller_protocol_inputs import ProtocolInputsController
from marge.controller.controller_protocol_list import ProtocolListController
from marge.controller.controller_toolbar_figures import FiguresController
from marge.controller.controller_toolbar_marcos import MarcosController
from marge.controller.controller_toolbar_protocols import ProtocolsController
from marge.controller.controller_toolbar_sequences import SequenceController
from marge.controller.controller_sequence_list import SequenceListController
from marge.controller.controller_sequence_inputs import SequenceInputsController
from marge.widgets.widget_custom_and_protocol import CustomAndProtocolWidget
from marge.controller.controller_postprocessing import ProcessingWindowController


class MainWindow(QMainWindow):
    def __init__(self, session, demo=False, parent=None):
        super(MainWindow, self).__init__()
        self.app_open = True
        self.toolbar_sequences = None
        self.toolbar_marcos = None
        self.session = session
        self.demo = demo
        self.parent = parent
        self.setWindowTitle(session.get('directory', 'MaRGE'))
        self.setGeometry(20, 40, 1680, 720)

        # Threadpool for parallel running
        self.threadpool = QThreadPool()

        # Set stylesheet based on theme
        if self.session["black_theme"]:
            self.styleSheet = qdarkstyle.load_stylesheet_pyqt5()
        else:
            self.styleSheet = ""
        self.setStyleSheet(self.styleSheet)

        # Create console
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"log_{now}.txt"
        self.console = ConsoleController(log_name)

        # Add marcos toolbar
        self.toolbar_marcos = MarcosController(self, "MaRCoS toolbar")
        self.addToolBar(self.toolbar_marcos)

        # Add sequence toolbar
        self.toolbar_sequences = SequenceController(self, "Sequence toolbar")
        self.addToolBar(self.toolbar_sequences)

        # Add image toolbar
        self.toolbar_figures = FiguresController(self, "Figures toolbar")
        self.addToolBar(self.toolbar_figures)

        # Add protocol toolbar
        self.toolbar_protocols = ProtocolsController(self, "Protocols toolbar")
        self.addToolBar(self.toolbar_protocols)

        # Add Scanner menu
        self.menu = self.menuBar()
        MenuController(main=self)

        # Status bar
        self.setStatusBar(QStatusBar(self))

        # Create central widget that will contain the layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        # Add main layout to input other layouts
        self.layout_main = QHBoxLayout()
        self.main_widget.setLayout(self.layout_main)

        # Layout for sequences and console
        self.layout_left = QVBoxLayout()
        self.layout_main.addLayout(self.layout_left)

        # Layout for outputs
        self.layout_right = QVBoxLayout()
        self.layout_main.addLayout(self.layout_right)

        # Add custom_and_protocol widget
        self.custom_and_protocol = CustomAndProtocolWidget()
        self.layout_left.addWidget(self.custom_and_protocol)

        # Add sequence list to custom tab
        self.sequence_list = SequenceListController(parent=self)
        self.custom_and_protocol.custom_layout.addWidget(self.sequence_list)

        # Add sequence inputs to custom tab
        self.sequence_inputs = SequenceInputsController(parent=self)
        self.custom_and_protocol.custom_layout.addWidget(self.sequence_inputs)

        # Add protocols list to protocol tab
        self.protocol_list = ProtocolListController(main=self)
        self.custom_and_protocol.protocol_layout.addWidget(self.protocol_list)

        # Add protocol sequences to protocol tab
        self.protocol_inputs = ProtocolInputsController(main=self)
        self.custom_and_protocol.protocol_layout.addWidget(self.protocol_inputs)

        # Add console
        self.layout_left.addWidget(self.console)

        # Add layout to show the figures
        self.figures_layout = FiguresLayoutController(self)
        self.figures_layout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout_right.addWidget(self.figures_layout)

        # Layout for output history
        self.layout_right_h = QHBoxLayout()
        self.layout_right.addLayout(self.layout_right_h)

        # Add list to show the history
        self.history_list = HistoryListController(parent=self)
        self.layout_right_h.addWidget(self.history_list)
        self.history_list.setMaximumHeight(200)
        self.history_list.setMinimumHeight(200)

        # Table with input parameters from historic images
        self.input_table = QTableWidget()
        self.input_table.setMaximumHeight(200)
        self.input_table.setMinimumHeight(200)
        self.layout_right_h.addWidget(self.input_table)

        # Create the post-processing toolbox
        self.post_gui = ProcessingWindowController(session=self.session, main=self)

