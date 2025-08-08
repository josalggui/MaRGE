"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QToolBar, QAction
from importlib import resources
class FiguresToolBar(QToolBar):
    def __init__(self, main, *args, **kwargs):
        super(FiguresToolBar, self).__init__(*args, **kwargs)
        self.main = main

        # Expand figure
        with resources.path("marge.resources.icons", "expand.png") as icon_path:
            self.action_full_screen = QAction(QIcon(str(icon_path)), "Full screen", self)
        self.action_full_screen.setStatusTip("Figure layout in full screen")
        self.addAction(self.action_full_screen)

        # Action button to do a screenshot

        with resources.path("marge.resources.icons", "screenshot.png") as icon_path:
            self.action_screenshot = QAction(QIcon(str(icon_path)), "Screenshot", self)
        self.action_screenshot.setStatusTip("Save screenshot in the session folder")
        self.addAction(self.action_screenshot)

        # Action button to open postprocessing gui
        with resources.path("marge.resources.icons", "open_directory.png") as icon_path:
            self.action_open_directory = QAction(QIcon(str(icon_path)), "Open main directory", self)
        self.action_open_directory.setStatusTip("Open main directory")
        self.addAction(self.action_open_directory)

        # Action button to open postprocessing gui
        with resources.path("marge.resources.icons", "postprocessing.png") as icon_path:
            self.action_postprocessing = QAction(QIcon(str(icon_path)), "Open post-processing GUI", self)
        self.action_postprocessing.setStatusTip("Open the post-processing GUI")
        self.addAction(self.action_postprocessing)

        # Add Switch Theme button to the marcos toolbar
        with resources.path("marge.resources.icons", "adjust-contrast.svg") as icon_path:
            self.switch_theme_action = QAction(QIcon(str(icon_path)), "", self)
        self.addAction(self.switch_theme_action)


