"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import csv
import os
from datetime import datetime

import numpy as np
from PyQt5.QtWidgets import QAction, QFileDialog

from seq.sequences import defaultsequences


class MenuController:
    def __init__(self, main):
        self.main = main

        # Add menus
        self.menu_scanner = self.main.menu.addMenu("Scanner")
        self.menu_protocols = self.main.menu.addMenu("Protocols")
        self.menu_sequences = self.main.menu.addMenu("Sequences")
        self.menu_session = self.main.menu.addMenu("Session")

        # Protocol menu
        self.menu_protocols.addAction(self.main.toolbar_protocols.action_new_protocol)
        self.menu_protocols.addAction(self.main.toolbar_protocols.action_del_protocol)
        self.menu_protocols.addAction(self.main.toolbar_protocols.action_new_sequence)
        self.menu_protocols.addAction(self.main.toolbar_protocols.action_del_sequence)

        # Scanner menu
        self.menu_scanner.addAction(self.main.toolbar_marcos.action_start)
        self.menu_scanner.addAction(self.main.toolbar_marcos.action_copybitstream)
        self.menu_scanner.addAction(self.main.toolbar_marcos.action_server)
        self.menu_scanner.addAction(self.main.toolbar_marcos.action_gpa_init)

        # Sequences menu
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_load_parameters)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_save_parameters)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_save_parameters_cal)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_add_to_list)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_acquire)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_bender)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_view_sequence)
