"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QTabWidget, QSizePolicy, QWidget, QVBoxLayout


class CustomAndProtocolWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        self.setMaximumWidth(400)

        # Create tabs for custom sequences and protocol
        self.custom_widget = QWidget()
        self.custom_layout = QVBoxLayout()
        self.custom_widget.setLayout(self.custom_layout)
        self.protocol_widget = QWidget()
        self.protocol_layout = QVBoxLayout()
        self.protocol_widget.setLayout(self.protocol_layout)

        self.addTab(self.custom_widget, "Custom")
        self.addTab(self.protocol_widget, "Protocols")

