"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QToolBar, QAction

class FiguresToolBar(QToolBar):
    def __init__(self, main, *args, **kwargs):
        super(FiguresToolBar, self).__init__(*args, **kwargs)
        self.main = main

        # Expand figure
        self.action_full_screen = QAction(QIcon("resources/icons/expand.png"), "Full sreen", self)
        self.action_full_screen.setStatusTip("Figure layout in full screen")
        self.addAction(self.action_full_screen)

        # Action button to do a screenshot
        self.action_screenshot = QAction(QIcon("resources/icons/screenshot.png"), "Screenshot", self)
        self.action_screenshot.setStatusTip("Save screenshot in the session folder")
        self.addAction(self.action_screenshot)

        # Action button to open postprocessing gui
        self.action_open_directory = QAction(QIcon("resources/icons/open_directory.png"), "Open main directory",
                                             self)
        self.action_open_directory.setStatusTip("Open main directory")
        self.addAction(self.action_open_directory)

        # Action button to open postprocessing gui
        self.action_postprocessing = QAction(QIcon("resources/icons/postprocessing.png"), "Open post-processing GUI", self)
        self.action_postprocessing.setStatusTip("Open the post-processing GUI")
        self.addAction(self.action_postprocessing)

        # Add Switch Theme button to the marcos toolbar
        self.switch_theme_action = QAction(QIcon("resources/icons/adjust-contrast.svg"), "", self)
        self.addAction(self.switch_theme_action)

