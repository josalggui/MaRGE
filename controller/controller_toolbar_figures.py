"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import os
from datetime import datetime

from PyQt5.QtGui import QPixmap

from widgets.widget_toolbar_figures import FiguresToolBar
from configs.sys_config import screenshot_folder


class FiguresController(FiguresToolBar):
    def __init__(self, *args, **kwargs):
        super(FiguresController, self).__init__(*args, **kwargs)

        if not os.path.exists(screenshot_folder):
            os.makedirs(screenshot_folder)

        self.action_full_screen.setCheckable(True)
        self.action_full_screen.triggered.connect(self.doFullScreen)
        self.action_screenshot.triggered.connect(self.doScreenshot)

    def doFullScreen(self):
        if self.action_full_screen.isChecked():
            self.main.history_list.hide()
            self.main.sequence_list.hide()
            self.main.sequence_inputs.hide()
            self.main.console.hide()
            self.main.input_table.hide()
        else:
            self.main.history_list.show()
            self.main.sequence_list.show()
            self.main.sequence_inputs.show()
            self.main.console.show()
            self.main.input_table.show()

    def doScreenshot(self):
        name = datetime.now()
        name_string = name.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3]
        file_name = name_string+".png"
        screenshot = QPixmap(self.main.size())
        self.main.render(screenshot)
        screenshot.save(screenshot_folder+"/"+file_name)
