"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import copy
import csv
import os
import threading
import time
from datetime import datetime

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QFileDialog

from configs import hw_config
from controller.controller_plot3d import Plot3DController as Spectrum3DPlot
from controller.controller_plot1d import Plot1DController as SpectrumPlot
from seq.sequences import defaultsequences
from widgets.widget_toolbar_sequences import SequenceToolBar


class SequenceController(SequenceToolBar):
    def __init__(self, *args, **kwargs):
        super(SequenceController, self).__init__(*args, **kwargs)

        # Set the action_iterate button checkable
        self.new_out = None
        self.old_out = None
        self.plots = None
        self.label = None
        self.win = None
        self.new_run = None
        self.old_seq_name = None
        self.seq_name = "RARE"
        self.iterative_run = None
        self.action_iterate.setCheckable(True)
        self.serverConnected()

        # Connect the action buttons to the slots
        self.action_autocalibration.triggered.connect(self.autocalibration)
        self.action_acquire.triggered.connect(self.startAcquisition)
        self.action_add_to_list.triggered.connect(self.runToList)
        self.action_view_sequence.triggered.connect(self.startSequencePlot)
        self.action_localizer.triggered.connect(self.startLocalizer)
        self.action_iterate.triggered.connect(self.iterate)
        self.action_bender.triggered.connect(self.bender)
        self.action_save_parameters.triggered.connect(self.saveParameters)
        self.action_load_parameters.triggered.connect(self.loadParameters)
        self.action_save_parameters_cal.triggered.connect(self.saveParametersCalibration)
        self.main.toolbar_marcos.action_server.triggered.connect(self.serverConnected)

    def bender(self):
        items = [self.main.protocol_inputs.item(index) for index in range(self.main.protocol_inputs.count())]
        for item in items:
            self.main.protocol_inputs.sequenceDoubleClicked(item)
            time.sleep(0.1)

    def autocalibration(self):
        # Include here the sequences to run on autocalibration
        seq_names = [
            'Larmor',
            'Noise',
            'RabiFlops',
            'Shimming'
        ]

        for seq_name in seq_names:
            # Execute the sequence
            self.runToList(seq_name=seq_name)

        # Update the inputs of the sequences
        self.main.sequence_list.updateSequence()

    def startAcquisition(self, seq_name=None):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: run selected sequence
        """
        # Load sequence name
        if seq_name is None or seq_name is False:
            self.seq_name = self.main.sequence_list.getCurrentSequence()
        else:
            self.seq_name = seq_name

        # Delete ouput if sequence is different from previous one
        if hasattr(self, "oldSeqName"):
            if self.seq_name != self.old_seq_name:
                self.new_run = True
                defaultsequences[self.seq_name].deleteOutput()
        self.old_seq_name = copy.copy(self.seq_name)

        if not hasattr(defaultsequences[self.seq_name], 'out'):
            self.new_run = True

        # Save sequence list into the current sequence, just in case you need to do sweep
        defaultsequences[self.seq_name].sequenceList = defaultsequences

        # Save input parameters
        defaultsequences[self.seq_name].saveParams()

        # if not hasattr(defaultsequences[self.seq_name], 'out'):  # If it is the first execution
        if self.new_run:
            self.new_run = False

            # Delete previous plots
            self.main.figures_layout.clearFiguresLayout()

            # Create label with rawdata name
            self.label = QLabel()
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            self.label.setStyleSheet("background-color: black;color: white")
            self.main.figures_layout.addWidget(self.label, row=0, col=0, colspan=2)

            # Update possible rotation, fov and dfov before the sequence is executed in parallel thread
            defaultsequences[self.seq_name].sequenceAtributes()

            # Create and execute selected sequence
            defaultsequences[self.seq_name].sequenceRun(0)

            # Do sequence analysis and acquire de plots
            self.old_out = defaultsequences[self.seq_name].sequenceAnalysis()

            # Update parameters, just in case something changed
            self.main.sequence_list.updateSequence()

            # Set name to the label
            file_name = defaultsequences[self.seq_name].mapVals['fileName']
            self.label.setText(file_name)

            # Add item to the history list
            self.main.history_list.current_output = str(datetime.now())[11:23]
            name = self.main.history_list.current_output + " | " + file_name
            self.main.history_list.addItem(name)

            # Clear inputs
            defaultsequences[self.seq_name].resetMapVals()

            # Save results into the history
            self.main.history_list.outputs[self.main.history_list.current_output] = self.old_out
            self.main.history_list.inputs[self.main.history_list.current_output] = \
                [list(defaultsequences[self.seq_name].mapNmspc.values()),
                 list(defaultsequences[self.seq_name].mapVals.values()),
                 False]

            # Save the rotation and shifts to the history list
            self.main.history_list.rotations[self.main.history_list.current_output] = \
                defaultsequences[self.seq_name].rotations.copy()
            self.main.history_list.shifts[self.main.history_list.current_output] = \
                defaultsequences[self.seq_name].dfovs.copy()
            self.main.history_list.fovs[self.main.history_list.current_output] = \
                defaultsequences[self.seq_name].fovs.copy()

            # Add plots to the plotview_layout
            self.plots = []
            for item in self.old_out:
                if item['widget'] == 'image':
                    image = Spectrum3DPlot(main=self.main,
                                           data=item['data'],
                                           x_label=item['xLabel'],
                                           y_label=item['yLabel'],
                                           title=item['title'])
                    self.main.figures_layout.addWidget(image, row=item['row'] + 1, col=item['col'])
                    defaultsequences[self.seq_name].deleteOutput()
                elif item['widget'] == 'curve':
                    self.plots.append(SpectrumPlot(x_data=item['xData'],
                                                   y_data=item['yData'],
                                                   legend=item['legend'],
                                                   x_label=item['xLabel'],
                                                   y_label=item['yLabel'],
                                                   title=item['title']))
                    self.main.figures_layout.addWidget(self.plots[-1], row=item['row'] + 1, col=item['col'])

            # Iterate in parallel thread (only for 1d plots)
            if self.action_iterate.isChecked() and hasattr(defaultsequences[self.seq_name], 'out'):
                thread = threading.Thread(target=self.repeatAcquisition)
                thread.start()

            # Deactivate the iterative buttom if sequence is not iterable (2d and 3d plots)
            if not hasattr(defaultsequences[self.seq_name], 'out') and self.action_iterate.isChecked():
                self.action_iterate.toggle()

        else:
            thread = threading.Thread(target=self.repeatAcquisition)
            thread.start()

    def runToList(self, seq_name=None):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: add new run to the waiting list
        """
        # Load sequence name
        if seq_name is None or seq_name is False:
            seq_name = self.main.sequence_list.getCurrentSequence()

        # Add item to the history list
        name = str(datetime.now())[11:23] + " | " + seq_name
        self.main.history_list.addItem(name)

        # Save results into the history
        self.main.history_list.inputs[name[0:12]] = [list(defaultsequences[seq_name].mapNmspc.values()),
                                                     list(defaultsequences[seq_name].mapVals.values()),
                                                     True]

        # Set to zero the dfov and angle for next figures
        for sequence in defaultsequences.values():
            if 'dfov' in sequence.mapKeys:
                sequence.mapVals['dfov'] = [0.0, 0.0, 0.0]   # mm
            if 'angle' in sequence.mapKeys:
                sequence.mapVals['angle'] = 0.0
        self.main.sequence_list.updateSequence()

    def startSequencePlot(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: plot sequence instructions
        """
        if self.main.demo:
            print("\nIt is not possible to plot a sequence in demo mode.")
            return

        # Delete previous plots
        self.main.figures_layout.clearFiguresLayout()

        # Load sequence name
        self.seq_name = self.main.sequence_list.getCurrentSequence()

        # Create sequence to plot
        print('Plot sequence')
        defaultsequences[self.seq_name].sequenceAtributes()
        defaultsequences[self.seq_name].sequenceRun(1)  # Run sequence only for plot

        # Get sequence to plot
        out = defaultsequences[self.seq_name].sequencePlot()  # Plot results

        # Create plots
        n = 0
        plot = []
        for item in out[0:3]:
            plot.append(SpectrumPlot(item[0], item[1], item[2], 'Time (ms)', 'Amplitude (a.u.)', item[3]))
            if n > 0: plot[n].plot_item.setXLink(plot[0].plot_item)
            n += 1
        for n in range(3):
            self.main.figures_layout.addWidget(plot[n], n, 0)

    def startLocalizer(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: run localizer
        """

        print('Start localizer')

        # Load sequence name
        seq_name = 'Localizer'

        defaultsequences[seq_name].loadParams(directory="calibration")

        # Sagittal localizer
        if defaultsequences[seq_name].mapVals['planes'][0]:
            defaultsequences[seq_name].mapVals['axesOrientation'] = [0, 1, 2]
            self.runToList(seq_name=seq_name)
            time.sleep(0.1)

        # Transversal localizer
        if defaultsequences[seq_name].mapVals['planes'][1]:
            defaultsequences[seq_name].mapVals['axesOrientation'] = [1, 2, 0]
            self.runToList(seq_name=seq_name)
            time.sleep(0.1)

        # Coronal localizer
        if defaultsequences[seq_name].mapVals['planes'][2]:
            defaultsequences[seq_name].mapVals['axesOrientation'] = [2, 0, 1]
            self.runToList(seq_name=seq_name)

    def iterate(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: swtich the iterative mode
        """
        if self.action_iterate.isChecked():
            self.action_iterate.setToolTip('Switch to single run')
            self.action_iterate.setStatusTip("Switch to single run")
        else:
            self.action_iterate.setToolTip('Switch to iterative run')
            self.action_iterate.setStatusTip("Switch to iterative run")
    
    def repeatAcquisition(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: executed when you repeat some calibration sequences
        """
        single_repetition = not self.action_iterate.isChecked()

        # Acquire while iterativeRun is True
        if not self.action_iterate.isChecked():
            # Create and execute selected sequence
            defaultsequences[self.seq_name].sequenceRun(0)

            # Do sequence analysis and acquire de plots
            self.new_out = defaultsequences[self.seq_name].sequenceAnalysis()

            # Set name to the label
            file_name = defaultsequences[self.seq_name].mapVals['fileName']
            self.label.setText(file_name)

            # Add item to the history list
            self.main.history_list.current_output = str(datetime.now())[11:23]
            name = self.main.history_list.current_output + " | " + file_name
            self.main.history_list.addItem(name)

            for plot_index in range(len(self.new_out)):
                old_curves = self.plots[plot_index].plot_item.listDataItems()
                for curveIndex in range(len(self.new_out[plot_index]['yData'])):
                    x = self.new_out[plot_index]['xData']
                    y = self.new_out[plot_index]['yData'][curveIndex]
                    old_curves[curveIndex].setData(x, y)

            # Clear inputs
            defaultsequences[self.seq_name].resetMapVals()

            # Save results into the history
            self.main.history_list.outputs[self.main.history_list.current_output] = self.new_out
            self.main.history_list.inputs[self.main.history_list.current_output] = \
                [list(defaultsequences[self.seq_name].mapNmspc.values()),
                 list(defaultsequences[self.seq_name].mapVals.values()),
                 False]

            # Save the rotation and shifts to the history list
            self.main.history_list.rotations[self.main.history_list.current_output] = \
                defaultsequences[self.seq_name].rotations.copy()
            self.main.history_list.shifts[self.main.history_list.current_output] = \
                defaultsequences[self.seq_name].dfovs.copy()
            self.main.history_list.fovs[self.main.history_list.current_output] = \
                defaultsequences[self.seq_name].fovs.copy()
        else:
            while self.action_iterate.isChecked():
                # Create and execute selected sequence
                defaultsequences[self.seq_name].sequenceRun(0)

                # Do sequence analysis and acquire de plots
                self.new_out = defaultsequences[self.seq_name].sequenceAnalysis()

                # Set name to the label
                file_name = defaultsequences[self.seq_name].mapVals['fileName']
                self.label.setText(file_name)

                # Add item to the history list
                self.main.history_list.current_output = str(datetime.now())[11:23]
                name = self.main.history_list.current_output + " | " + file_name
                self.main.history_list.addItem(name)

                for plot_index in range(len(self.new_out)):
                    old_curves = self.plots[plot_index].plot_item.listDataItems()
                    for curveIndex in range(len(self.new_out[plot_index]['yData'])):
                        x = self.new_out[plot_index]['xData']
                        y = self.new_out[plot_index]['yData'][curveIndex]
                        old_curves[curveIndex].setData(x, y)

                # Clear inputs
                defaultsequences[self.seq_name].resetMapVals()

                # Save results into the history
                self.main.history_list.outputs[self.main.history_list.current_output] = self.new_out
                self.main.history_list.inputs[self.main.history_list.current_output] = \
                    [list(defaultsequences[self.seq_name].mapNmspc.values()),
                     list(defaultsequences[self.seq_name].mapVals.values()),
                     False]

                # Save the rotation and shifts to the history list
                self.main.history_list.rotations[self.main.history_list.current_output] = \
                    defaultsequences[self.seq_name].rotations.copy()
                self.main.history_list.shifts[self.main.history_list.current_output] = \
                    defaultsequences[self.seq_name].dfovs.copy()
                self.main.history_list.fovs[self.main.history_list.current_output] = \
                    defaultsequences[self.seq_name].fovs.copy()

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
        with open('experiments/parameterization/%s.%s.csv' % (seq.mapNmspc['seqName'], dt_string), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=seq.mapKeys)
            writer.writeheader()
            map_vals = {}
            for key in seq.mapKeys:  # take only the inputs from mapVals
                map_vals[key] = seq.mapVals[key]
            writer.writerows([seq.mapNmspc, map_vals])

        # self.messages("Parameters of %s sequence saved" %(self.sequence))
        print("\nParameters of %s sequence saved in 'experiments/parameterization'" %(self.main.sequence_list.getCurrentSequence()))

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

        print("\nParameters of %s sequence saved in 'calibration'" % (self.main.sequence_list.getCurrentSequence()))

    def serverConnected(self):
        if self.main.toolbar_marcos.action_server.isChecked():
            self.action_acquire.setDisabled(False)
            self.action_localizer.setDisabled(False)
            self.action_autocalibration.setDisabled(False)
            self.action_bender.setDisabled(False)
            self.action_view_sequence.setDisabled(False)
            self.action_add_to_list.setDisabled(False)
        else:
            self.action_acquire.setDisabled(True)
            self.action_localizer.setDisabled(True)
            self.action_autocalibration.setDisabled(True)
            self.action_bender.setDisabled(True)
            self.action_view_sequence.setDisabled(True)
            self.action_add_to_list.setDisabled(True)
