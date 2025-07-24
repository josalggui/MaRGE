"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QToolBar, QAction
from importlib import resources


class ProtocolsToolBar(QToolBar):
    def __init__(self, main, *args, **kwargs):
        super(ProtocolsToolBar, self).__init__(*args, **kwargs)
        self.main = main

        # New protocol
        with resources.path("marge.resources.icons", "addProtocol.png") as path_add_protocol:
            self.action_new_protocol = QAction(QIcon(str(path_add_protocol)), "New Protocol", self)
        self.action_new_protocol.setStatusTip("Create a new protocol")
        self.addAction(self.action_new_protocol)

        # Del protocol
        with resources.path("marge.resources.icons", "deleteProtocol.png") as path_del_protocol:
            self.action_del_protocol = QAction(QIcon(str(path_del_protocol)), "Remove Protocol", self)
        self.action_del_protocol.setStatusTip("Remove a protocol")
        self.addAction(self.action_del_protocol)

        # New sequence
        with resources.path("marge.resources.icons", "addSequence.png") as path_add_sequence:
            self.action_new_sequence = QAction(QIcon(str(path_add_sequence)), "New Sequence", self)
        self.action_new_sequence.setStatusTip("Create a new sequence")
        self.addAction(self.action_new_sequence)

        # Del sequence
        with resources.path("marge.resources.icons", "deleteSequence.png") as path_del_sequence:
            self.action_del_sequence = QAction(QIcon(str(path_del_sequence)), "Remove Sequence", self)
        self.action_del_sequence.setStatusTip("Remove a sequence from protocol")
        self.addAction(self.action_del_sequence)
