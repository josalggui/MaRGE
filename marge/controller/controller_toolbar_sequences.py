"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import copy
import csv
import threading
import time
from datetime import datetime

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QFileDialog

from marge.controller.controller_plot3d import Plot3DController as Spectrum3DPlot
from marge.controller.controller_plot1d import Plot1DController as SpectrumPlot
from marge.seq.sequences import defaultsequences
from marge.widgets.widget_toolbar_sequences import SequenceToolBar
import marge.configs.hw_config as hw


class SequenceController(SequenceToolBar):
    """
    A class that controls the sequence and provides methods for interacting with it.

    Inherits from `SequenceToolBar`.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the SequenceController object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
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
        """
        Simulates a bender action.

        Iterates through the `protocol_inputs` items in the main window and triggers a sequence double-click action for each item.
        Adds a small delay of 0.1 seconds between each action to simulate the bender action.
        """
        items = [self.main.protocol_inputs.item(index) for index in range(self.main.protocol_inputs.count())]
        for item in items:
            self.main.protocol_inputs.sequenceDoubleClicked(item)
            hw.dfov = [0.0, 0.0, 0.0]
            time.sleep(0.1)

    def autocalibration(self):
        """
        Executes autocalibration sequences.

        Runs a predefined set of sequences for autocalibration purposes.
        The `seq_names` list contains the names of the sequences to be executed.
        For the sequence 'RabiFlops', custom modifications are made to the sequence parameters before execution.
        Finally, the `runToList` method is called for each sequence in `seq_names`.
        Updates the inputs of the sequences in the main window after execution.
        """
        # Include here the sequences to run on autocalibration
        seq_names = [
            'Larmor',
            'AutoTuning',
            'Noise',
            'Shimming',
            'RabiFlops',
            'Larmor',
        ]

        for seq_name in seq_names:
            # Get sequence parameters
            seq = defaultsequences[seq_name]
            seq.loadParams(directory='calibration', file=seq_name)

            # Specific tasks for RabiFlops
            if seq_name == 'RabiFlops':
                # Fix rf amplitude
                rf_amp = np.pi/(hw.b1Efficiency*hw.reference_time)
                seq.mapVals['rfExAmp'] = rf_amp
                seq.mapVals['rfReAmp'] = rf_amp

            self.runToList(seq_name=seq_name, item_name="Calibration_"+seq_name)

        # Update the inputs of the sequences
        self.main.sequence_list.updateSequence()

    def startAcquisition(self, seq_name=None):
        """
        Run the selected sequence and perform data acquisition.

        Args:
            seq_name (str, optional): Name of the sequence to run. If not provided or False, the current sequence in the sequence list is used.

        Returns:
            int: Return 0 if the sequence run fails.

        Summary:
            This method executes the selected sequence and handles the data acquisition process.
            It performs various operations such as loading the sequence name, deleting output if the sequence has changed,
            saving input parameters, updating sequence attributes, creating and executing the sequence, analyzing the
            sequence output, updating parameters, displaying the output label, saving results to history, adding plots to
            the plot view, and optionally iterating the acquisition in a separate thread.
        """
        # Load sequence name
        if seq_name is None or seq_name is False:
            self.seq_name = self.main.sequence_list.getCurrentSequence()
        else:
            self.seq_name = seq_name

        # Delete ouput if sequence is different from previous one
        if hasattr(self, "old_seq_name"):
            if self.seq_name != self.old_seq_name:
                self.new_run = True
                defaultsequences[self.seq_name].deleteOutput()
        self.old_seq_name = copy.copy(self.seq_name)

        if not hasattr(defaultsequences[self.seq_name], 'output'):
            self.new_run = True

        # Save sequence list into the current sequence, just in case you need to do sweep
        defaultsequences[self.seq_name].sequence_list = defaultsequences

        # Add sequence name for metadata
        defaultsequences[self.seq_name].raw_data_name = self.seq_name

        # Save input parameters
        defaultsequences[self.seq_name].saveParams()

        if self.new_run:
            self.new_run = False

            # Update possible rotation, fov and dfov before the sequence is executed in parallel thread
            defaultsequences[self.seq_name].sequenceAtributes()

            # Create and execute selected sequence
            if defaultsequences[self.seq_name].sequenceRun(0, self.main.demo):
                # Delete previous plots
                self.main.figures_layout.clearFiguresLayout()

                # Create label with rawdata name
                self.label = QLabel()
                self.label.setAlignment(QtCore.Qt.AlignCenter)
                self.label.setStyleSheet("background-color: black;color: white")
                self.main.figures_layout.addWidget(self.label, row=0, col=0, colspan=2)
            else:
                return 0

            # Do sequence analysis and acquire de plots
            self.old_out = defaultsequences[self.seq_name].sequenceAnalysis()

            # Update parameters, just in case something changed
            self.main.sequence_list.updateSequence()

            # Set name to the label
            file_name = defaultsequences[self.seq_name].mapVals['fileName']
            self.label.setText(file_name)

            # Add item to the history list
            self.main.history_list.current_output = str(datetime.now())[11:23] + " | " + file_name.split('.')[0]
            item_name = str(datetime.now())[11:23] + " | " + file_name
            self.main.history_list.addItem(item_name)

            # Clear inputs
            defaultsequences[self.seq_name].resetMapVals()

            # Save results into the history
            self.main.history_list.outputs[self.main.history_list.current_output] = self.old_out
            self.main.history_list.inputs[self.main.history_list.current_output] = \
                [list(defaultsequences[self.seq_name].mapNmspc.values()),
                 list(defaultsequences[self.seq_name].mapVals.values())]

            # Save the rotation and shifts to the history list
            self.main.history_list.rotations[self.main.history_list.current_output] = \
                defaultsequences[self.seq_name].rotations.copy()
            self.main.history_list.shifts[self.main.history_list.current_output] = \
                defaultsequences[self.seq_name].dfovs.copy()
            self.main.history_list.fovs[self.main.history_list.current_output] = \
                defaultsequences[self.seq_name].fovs.copy()

            # Add plots to the plotview_layout
            self.plots = []
            n_columns = 1
            for item in self.old_out:
                if item['col']+1 > n_columns:
                    n_columns = item['col']+1
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
            self.main.figures_layout.addWidget(self.label, row=0, col=0, colspan=n_columns)

            # Iterate in parallel thread (only for 1d plots)
            if self.action_iterate.isChecked() and hasattr(defaultsequences[self.seq_name], 'output'):
                thread = threading.Thread(target=self.repeatAcquisition)
                thread.start()

            # Deactivate the iterative buttom if sequence is not iterable (2d and 3d plots)
            if not hasattr(defaultsequences[self.seq_name], 'output') and self.action_iterate.isChecked():
                self.action_iterate.toggle()

        else:
            thread = threading.Thread(target=self.repeatAcquisition)
            thread.start()

    def runToList(self, seq_name=None, item_name=None, map_nmspc=None, map_vals=None):
        """
        Add a new run to the waiting list.

        Args:
            seq_name (str, optional): Name of the sequence. If not provided or False, the current sequence in the sequence list is used.
            item_name (str, optional): Name of the item to add to the history list. If not provided, a timestamped name is used.

        Summary:
            This method adds a new run to the waiting list. It retrieves the sequence name, adds the item to the history list,
            saves the results into the history, and sets the dfov and angle values to zero for the next figures.
        """
        # Load sequence name
        if seq_name is None or seq_name is False:
            seq_name = self.main.sequence_list.getCurrentSequence()

        # Add item to the history list
        if item_name is None:
            name = str(datetime.now())[11:23] + " | " + seq_name
        else:
            name = str(datetime.now())[11:23] + " | " + item_name
        self.main.history_list.addItem(name)

        sequence = copy.deepcopy(defaultsequences[seq_name])
        if map_nmspc is None and map_vals is None:
            map_nmspc = list(sequence.mapNmspc.values())
            map_vals = list(sequence.mapVals.values())

        # Save results into the history
        self.main.history_list.inputs[name] = [map_nmspc, map_vals]
        self.main.history_list.pending_inputs[name] = [map_nmspc, map_vals]

        # Set to zero the dfov and angle for next figures
        hw.dfov = [0.0, 0.0, 0.0]
        for sequence in defaultsequences.values():
            if 'dfov' in sequence.mapKeys:
                sequence.mapVals['dfov'] = [0.0, 0.0, 0.0]   # mm
            if 'angle' in sequence.mapKeys:
                sequence.mapVals['angle'] = 0.0

        self.main.sequence_list.updateSequence()

    def startSequencePlot(self):
        """
        Plot the sequence instructions.

        Returns:
            int: Return 0 if plotting fails.

        Summary:
            This method plots the instructions of the current sequence. It retrieves the sequence name, creates the sequence
            to plot, runs the sequence in demo mode, and creates the necessary plots based on the sequence output.
        """

        # Load sequence name
        self.seq_name = self.main.sequence_list.getCurrentSequence()

        # Create sequence to plot
        print('Plot sequence')
        defaultsequences[self.seq_name].sequenceAtributes()
        if defaultsequences[self.seq_name].sequenceRun(1, demo=self.main.demo):
            # Delete previous plots
            self.main.figures_layout.clearFiguresLayout()
        else:
            return 0

        # Get sequence to plot
        out = defaultsequences[self.seq_name].sequencePlot()  # Plot results

        # Create plots
        n = 0
        plot = []
        for item in out[0:3]:
            plot.append(SpectrumPlot(item[0], item[1], item[2], 'Time (ms)', 'Amplitude (a.u.)', item[3]))
        for n in range(3):
            self.main.figures_layout.addWidget(plot[n], n, 0)
        plot[0].plot_item.setXLink(plot[1].plot_item)
        plot[2].plot_item.setXLink(plot[1].plot_item)

    def startLocalizer(self):
        """
        Run the localizer sequence.

        Summary:
            This method runs the localizer sequence. It loads the sequence parameters, sets the shimming values from the 'RARE'
            sequence, and runs the sagittal, transversal, and coronal localizers based on the specified planes.
        """

        print('Start localizer')

        # Load sequence name
        seq_name = 'Localizer'

        defaultsequences[seq_name].loadParams(directory="calibration")
        defaultsequences[seq_name].mapVals['shimming'] = defaultsequences['RARE'].mapVals['shimming']

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
        Switch the iterative mode.

        Summary:
            This method switches the iterative mode based on the state of the 'iterate' action. If the action is checked, the
            tooltip and status tip are updated to indicate that switching to single run is possible. If the action is unchecked,
            the tooltip and status tip are updated to indicate that switching to iterative run is possible.
        """
        if self.action_iterate.isChecked():
            self.action_iterate.setToolTip('Switch to single run')
            self.action_iterate.setStatusTip("Switch to single run")
        else:
            self.action_iterate.setToolTip('Switch to iterative run')
            self.action_iterate.setStatusTip("Switch to iterative run")
    
    def repeatAcquisition(self):
        """
        Executed when you repeat some calibration sequences.

        Summary:
            This method is executed when you repeat some calibration sequences. If the iterative run mode is not enabled, it
            creates and executes the selected sequence. It performs sequence analysis, updates the plots, adds the item to the
            history list, and saves the results. If the iterative run mode is enabled, it repeatedly creates and executes the
            selected sequence until the iterative run action is unchecked.
        """
        # Acquire while iterativeRun is True
        if not self.action_iterate.isChecked():
            # Generate atributes according to inputs
            defaultsequences[self.seq_name].sequenceAtributes()

            # Create and execute selected sequence
            defaultsequences[self.seq_name].sequenceRun(0, self.main.demo)

            # Do sequence analysis and acquire de plots
            self.new_out = defaultsequences[self.seq_name].sequenceAnalysis()

            # Set name to the label
            file_name = defaultsequences[self.seq_name].mapVals['fileName']
            self.label.setText(file_name)

            # Add item to the history list
            self.main.history_list.current_output = str(datetime.now())[11:23] + " | " + file_name.split('.')[0]
            item_name = str(datetime.now())[11:23] + " | " + file_name
            self.main.history_list.addItem(item_name)

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
                 list(defaultsequences[self.seq_name].mapVals.values())]

            # Save the rotation and shifts to the history list
            self.main.history_list.rotations[self.main.history_list.current_output] = \
                defaultsequences[self.seq_name].rotations.copy()
            self.main.history_list.shifts[self.main.history_list.current_output] = \
                defaultsequences[self.seq_name].dfovs.copy()
            self.main.history_list.fovs[self.main.history_list.current_output] = \
                defaultsequences[self.seq_name].fovs.copy()
        else:
            while self.action_iterate.isChecked():
                # Generate atributes according to inputs
                defaultsequences[self.seq_name].sequenceAtributes()

                # Create and execute selected sequence
                defaultsequences[self.seq_name].sequenceRun(0, self.main.demo)

                # Do sequence analysis and acquire de plots
                self.new_out = defaultsequences[self.seq_name].sequenceAnalysis()

                # Set name to the label
                file_name = defaultsequences[self.seq_name].mapVals['fileName']
                self.label.setText(file_name)

                # Add item to the history list
                self.main.history_list.current_output = str(datetime.now())[11:23] + " | " + file_name.split('.')[0]
                item_name = str(datetime.now())[11:23] + " | " + file_name
                self.main.history_list.addItem(item_name)

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
                     list(defaultsequences[self.seq_name].mapVals.values())]

                # Save the rotation and shifts to the history list
                self.main.history_list.rotations[self.main.history_list.current_output] = \
                    defaultsequences[self.seq_name].rotations.copy()
                self.main.history_list.shifts[self.main.history_list.current_output] = \
                    defaultsequences[self.seq_name].dfovs.copy()
                self.main.history_list.fovs[self.main.history_list.current_output] = \
                    defaultsequences[self.seq_name].fovs.copy()

    def loadParameters(self):
        """
        Opens a file dialog to load parameters from a CSV file and updates the sequence's mapVals with the new values.

        Summary:
            This method opens a file dialog to allow the user to select a CSV file containing parameter values. The selected
            file is read, and the mapVals of the current sequence are updated with the new parameter values. The CSV file is
            expected to have the same fieldnames as the mapKeys of the sequence. After updating the mapVals, the sequence list
            is updated to reflect the changes.
        """
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
            try:
                val_new = map_vals_new[key]
            except:
                val_new = str(val_old) * 1
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
        print("Parameters of %s sequence loaded" % (self.main.sequence_list.getCurrentSequence()))

    def saveParameters(self):
        """
        Saves the current sequence's parameter values to a CSV file in the 'experiments/parameterization/' directory.

        Summary:
            This method saves the parameter values of the current sequence to a CSV file in the 'experiments/parameterization/'
            directory. The file is named using the sequence's name and the current timestamp. The mapKeys and mapVals of the
            sequence are used to construct the CSV file. The sequence's mapNmspc is written as the first row of the CSV file,
            followed by the mapVals.
        """
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
        print("Parameters of %s sequence saved in 'experiments/parameterization'" %(self.main.sequence_list.getCurrentSequence()))

    def saveParametersCalibration(self):
        """
        Saves the current sequence's parameter values to a CSV file in the 'calibration/' directory.

        Summary:
            This method saves the parameter values of the current sequence to a CSV file in the 'calibration/' directory.
            The file is named using the sequence's name appended with '_last_parameters.csv'. The mapKeys and mapVals of the
            sequence are used to construct the CSV file. The sequence's mapNmspc is written as the first row of the CSV file,
            followed by the mapVals.
"""
        seq = defaultsequences[self.main.sequence_list.getCurrentSequence()]

        # Save csv with input parameters
        with open('calibration/%s_last_parameters.csv' % seq.mapVals['seqName'], 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=seq.mapKeys)
            writer.writeheader()
            map_vals = {}
            for key in seq.mapKeys:  # take only the inputs from mapVals
                map_vals[key] = seq.mapVals[key]
            writer.writerows([seq.mapNmspc, map_vals])

        print("Parameters of %s sequence saved in 'calibration'" % (self.main.sequence_list.getCurrentSequence()))

    def serverConnected(self):
        """
        Enables or disables certain actions based on the server connection status.

        Summary:
            This method is called when the server connection status changes. If the server is connected (checked), it enables
            certain actions including 'action_acquire', 'action_localizer', 'action_autocalibration', 'action_bender',
            'action_view_sequence', and 'action_add_to_list'. If the server is not connected (unchecked), it disables these actions.
        """
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
