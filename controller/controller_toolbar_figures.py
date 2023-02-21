"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from widgets.widget_toolbar_figures import FiguresToolBar


class FiguresController(FiguresToolBar):
    def __init__(self, *args, **kwargs):
        super(FiguresController, self).__init__(*args, **kwargs)

        self.action_full_screen.setCheckable(True)
        self.action_full_screen.triggered.connect(self.doFullScreen)

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


