"""
session_controller.py
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from ui.window_main import MainWindow


class MainController(MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainController, self).__init__(*args, **kwargs)
