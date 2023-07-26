"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QToolBar, QAction
from configs.sys_config import screenshot_folder


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
        self.action_screenshot.setStatusTip("Save screenshot in %s" % screenshot_folder)
        self.addAction(self.action_screenshot)
