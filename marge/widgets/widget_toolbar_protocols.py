"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QToolBar, QAction


class ProtocolsToolBar(QToolBar):
    def __init__(self, main, *args, **kwargs):
        super(ProtocolsToolBar, self).__init__(*args, **kwargs)
        self.main = main

        # New protocol
        self.action_new_protocol = QAction(QIcon("resources/icons/addProtocol.png"), "New Protocol", self)
        self.action_new_protocol.setStatusTip("Create a new protocol")
        self.addAction(self.action_new_protocol)

        # Del protocol
        self.action_del_protocol = QAction(QIcon("resources/icons/deleteProtocol.png"), "Remove Protocol", self)
        self.action_del_protocol.setStatusTip("Remove a protocol")
        self.addAction(self.action_del_protocol)

        # New sequence
        self.action_new_sequence = QAction(QIcon("resources/icons/addSequence.png"), "New Sequence", self)
        self.action_new_sequence.setStatusTip("Create a new sequence")
        self.addAction(self.action_new_sequence)

        # Del sequence
        self.action_del_sequence = QAction(QIcon("resources/icons/deleteSequence.png"), "Remove Sequence", self)
        self.action_del_sequence.setStatusTip("Remove a sequence from protocol")
        self.addAction(self.action_del_sequence)
