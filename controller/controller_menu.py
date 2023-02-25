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
        self.action_load_parameters = QAction("Load parameters", self.main)
        self.action_load_parameters.setStatusTip("Load sequence parameters")
        self.action_load_parameters.triggered.connect(self.loadParameters)
        self.menu_sequences.addAction(self.action_load_parameters)

        self.action_save_parameters = QAction("Save parameters", self.main)
        self.action_save_parameters.setStatusTip("Save sequence parameters")
        self.action_save_parameters.triggered.connect(self.saveParameters)
        self.menu_sequences.addAction(self.action_save_parameters)

        self.action_save_calibration = QAction("Save as quick calibration", self.main)
        self.action_save_calibration.setStatusTip("Save sequence parameters for quick calibration")
        self.action_save_calibration.triggered.connect(self.saveParametersCalibration)
        self.menu_sequences.addAction(self.action_save_calibration)

        self.menu_sequences.addAction(self.main.toolbar_sequences.action_add_to_list)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_acquire)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_view_sequence)

    def loadParameters(self):

        file_name, _ = QFileDialog.getOpenFileName(self.main, 'Open Parameters File', "experiments/parameterization/")

        seq = defaultsequences[self.main.sequence_list.getCurrentSequence()]
        map_vals_old = seq.mapVals
        with open(file_name, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for l in reader:
                map_vals_new = l

        seq.mapVals = {}

        # Get key for corresponding modified parameter
        for key in seq.mapKeys:
            data_len = seq.mapLen[key]
            val_old = map_vals_old[key]
            val_new = map_vals_new[key]
            val_new = val_new.replace('[', '')
            val_new = val_new.replace(']', '')
            val_new = val_new.split(',')
            if type(val_old) == str:
                val_old = [val_old]
            elif data_len == 1:
                val_old = [val_old]
            data_type = type(val_old[0])

            inputNum = []
            for ii in range(data_len):
                if data_type == float or data_type == np.float64:
                    try:
                        inputNum.append(float(val_new[ii]))
                    except:
                        inputNum.append(float(val_old[ii]))
                elif data_type == int:
                    try:
                        inputNum.append(int(val_new[ii]))
                    except:
                        inputNum.append(int(val_old[ii]))
                else:
                    try:
                        inputNum.append(str(val_new[0]))
                        break
                    except:
                        inputNum.append(str(val_old[0]))
                        break
            if data_type == str:
                seq.mapVals[key] = inputNum[0]
            else:
                if data_len == 1:  # Save value into mapVals
                    seq.mapVals[key] = inputNum[0]
                else:
                    seq.mapVals[key] = inputNum

        self.main.sequence_list.updateSequence()
        print("\nParameters of %s sequence loaded" % (self.main.sequence_list.getCurrentSequence()))

    def saveParameters(self):
        dt = datetime.now()
        dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3]
        seq = defaultsequences[self.main.sequence_list.getCurrentSequence()]

        # Save csv with input parameters
        if not os.path.exists('experiments/parameterization'):
            os.makedirs('experiments/parameterization')
        with open('experiments/parameterization/%s.%s.csv' % (seq.mapNmspc['seqName'], dt_string), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=seq.mapKeys)
            writer.writeheader()
            map_vals = {}
            for key in seq.mapKeys:  # take only the inputs from mapVals
                map_vals[key] = seq.mapVals[key]
            writer.writerows([seq.mapNmspc, map_vals])

        # self.messages("Parameters of %s sequence saved" %(self.sequence))
        print("\nParameters of %s sequence saved" %(self.main.sequence_list.getCurrentSequence()))

    def saveParametersCalibration(self):
        seq = defaultsequences[self.main.sequence_list.getCurrentSequence()]

        # Save csv with input parameters
        with open('calibration/%s_last_parameters.csv' % seq.mapVals['seqName'], 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=seq.mapKeys)
            writer.writeheader()
            map_vals = {}
            for key in seq.mapKeys:  # take only the inputs from mapVals
                map_vals[key] = seq.mapVals[key]
            writer.writerows([seq.mapNmspc, map_vals])

        print("\nParameters of %s sequence saved" %(self.main.sequence_list.getCurrentSequence()))