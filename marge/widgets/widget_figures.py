"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QSizePolicy
from pyqtgraph import LayoutWidget


class FiguresLayoutWidget(LayoutWidget):
    """Layout widget that organises the 1D and 3D figure panels in the main window."""
    def __init__(self):
        super().__init__()
