"""
session_controller.py
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import csv
import os
import sys
import threading
from datetime import datetime

import numpy as np
from PyQt5.QtWidgets import QAction, QFileDialog

from seq.sequences import defaultsequences
from ui.window_main import MainWindow

from configs import hw_config as hw


class MainController(MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainController, self).__init__(*args, **kwargs)

        # Add the session to all sequences
        for sequence in defaultsequences.values():
            sequence.session = self.session

        # Scanner menu
        self.menu_scanner.addAction(self.toolbar_marcos.action_start)
        self.menu_scanner.addAction(self.toolbar_marcos.action_copybitstream)
        self.menu_scanner.addAction(self.toolbar_marcos.action_server)
        self.menu_scanner.addAction(self.toolbar_marcos.action_gpa_init)

        # Protocols menu
        self.action_protocol_01 = QAction("Localizer", self)
        self.action_protocol_01.setStatusTip("Run localizer")
        self.menu_protocols.addAction(self.action_protocol_01)

        self.action_protocol_02 = QAction("RARE T1", self)
        self.action_protocol_02.setStatusTip("RARE with T1 contrast")
        self.action_protocol_02.triggered.connect(self.protocol02)
        self.menu_protocols.addAction(self.action_protocol_02)

        self.action_protocol_03 = QAction("RARE T2", self)
        self.action_protocol_03.setStatusTip("RARE with T2 contrast")
        self.menu_protocols.addAction(self.action_protocol_03)

        self.action_protocol_04 = QAction("RARE Rho", self)
        self.action_protocol_04.setStatusTip("RARE with Rho contrast")
        self.action_protocol_04.triggered.connect(self.protocol04)
        self.menu_protocols.addAction(self.action_protocol_04)

        self.action_protocol_05 = QAction("RARE knee T1", self)
        self.action_protocol_05.setStatusTip("RARE with T1 contrast for knee")
        self.menu_protocols.addAction(self.action_protocol_05)

        # Sequences menu
        self.action_load_parameters = QAction("Load parameters", self)
        self.action_load_parameters.setStatusTip("Load sequence parameters")
        self.action_load_parameters.triggered.connect(self.loadParameters)
        self.menu_sequences.addAction(self.action_load_parameters)

        self.action_save_parameters = QAction("Save parameters", self)
        self.action_save_parameters.setStatusTip("Save sequence parameters")
        self.action_save_parameters.triggered.connect(self.saveParameters)
        self.menu_sequences.addAction(self.action_save_parameters)

        self.action_save_calibration = QAction("Save as quick calibration", self)
        self.action_save_calibration.setStatusTip("Save sequence parameters for quick calibration")
        self.action_save_calibration.triggered.connect(self.saveParametersCalibration)
        self.menu_sequences.addAction(self.action_save_calibration)

        self.menu_sequences.addAction(self.toolbar_sequences.action_add_to_list)
        self.menu_sequences.addAction(self.toolbar_sequences.action_acquire)
        self.menu_sequences.addAction(self.toolbar_sequences.action_view_sequence)

        # Session menu


        thread = threading.Thread(target=self.history_list.waitingForRun)
        thread.start()

    def protocol02(self):
        # Load parameters
        defaultsequences['RARE'].loadParams(directory='protocols', file='RARE_3D_T1.csv')

        # Set larmor frequency to the value into the hw_config file
        defaultsequences['RARE'].mapVals['larmorFreq'] = hw.larmorFreq

        # Run the sequence
        self.toolbar_sequences.runToList(seq_name='RARE')

    def protocol04(self):
        # Load parameters
        defaultsequences['RARE'].loadParams(directory='protocoles', file='RARE_3D_HAND_RHO.csv')

        # Set larmor frequency to the value into the hw_config file
        defaultsequences['RARE'].mapVals['larmorFreq'] = hw.larmorFreq

        # Run the sequence
        self.toolbar_sequences.runToList(seq_name='RARE')

    def loadParameters(self):

        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Parameters File', "experiments/parameterisations/")

        seq = defaultsequences[self.sequence_list.getCurrentSequence()]
        mapValsOld = seq.mapVals
        with open(file_name, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for l in reader:
                mapValsNew = l

        seq.mapVals = {}

        # Get key for corresponding modified parameter
        for key in seq.mapKeys:
            dataLen = seq.mapLen[key]
            valOld = mapValsOld[key]
            valNew = mapValsNew[key]
            valNew = valNew.replace('[', '')
            valNew = valNew.replace(']', '')
            valNew = valNew.split(',')
            if type(valOld) == str:
                valOld = [valOld]
            elif dataLen == 1:
                valOld = [valOld]
            dataType = type(valOld[0])

            inputNum = []
            for ii in range(dataLen):
                if dataType == float or dataType == np.float64:
                    try:
                        inputNum.append(float(valNew[ii]))
                    except:
                        inputNum.append(float(valOld[ii]))
                elif dataType == int:
                    try:
                        inputNum.append(int(valNew[ii]))
                    except:
                        inputNum.append(int(valOld[ii]))
                else:
                    try:
                        inputNum.append(str(valNew[0]))
                        break
                    except:
                        inputNum.append(str(valOld[0]))
                        break
            if dataType == str:
                seq.mapVals[key] = inputNum[0]
            else:
                if dataLen == 1:  # Save value into mapVals
                    seq.mapVals[key] = inputNum[0]
                else:
                    seq.mapVals[key] = inputNum

        self.sequence_list.updateSequence()
        print("\nParameters of %s sequence loaded" % (self.sequence_list.getCurrentSequence()))

    def saveParameters(self):
        dt = datetime.now()
        dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3]
        seq = defaultsequences[self.sequence_list.getCurrentSequence()]

        # Save csv with input parameters
        if not os.path.exists('experiments/parameterisations'):
            os.makedirs('experiments/parameterisations')
        with open('experiments/parameterisations/%s.%s.csv' % (seq.mapNmspc['seqName'], dt_string), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=seq.mapKeys)
            writer.writeheader()
            mapVals = {}
            for key in seq.mapKeys:  # take only the inputs from mapVals
                mapVals[key] = seq.mapVals[key]
            writer.writerows([seq.mapNmspc, mapVals])

        # self.messages("Parameters of %s sequence saved" %(self.sequence))
        print("\nParameters of %s sequence saved" %(self.sequence_list.getCurrentSequence()))

    def saveParametersCalibration(self):
        seq = defaultsequences[self.sequence_list.getCurrentSequence()]

        # Save csv with input parameters
        with open('calibration/%s_last_parameters.csv' % seq.mapVals['seqName'], 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=seq.mapKeys)
            writer.writeheader()
            mapVals = {}
            for key in seq.mapKeys:  # take only the inputs from mapVals
                mapVals[key] = seq.mapVals[key]
            writer.writerows([seq.mapNmspc, mapVals])

        print("\nParameters of %s sequence saved" %(self.sequence_list.getCurrentSequence()))

    def closeEvent(self, event):
        """Shuts down application on close."""
        self.app_open = False
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        if not self.demo:
            os.system('ssh root@192.168.1.101 "killall marcos_server"') # Kill marcos server
        print('\nGUI closed successfully!')
        super().closeEvent(event)
