"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import copy
import threading
import time
from datetime import datetime

from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel

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

        # Connect the action buttons to the slots
        self.action_autocalibration.triggered.connect(self.autocalibration)
        self.action_acquire.triggered.connect(self.startAcquisition)
        self.action_add_to_list.triggered.connect(self.runToList)
        self.action_view_sequence.triggered.connect(self.startSequencePlot)
        self.action_localizer.triggered.connect(self.startLocalizer)
        self.action_iterate.triggered.connect(self.iterate)
        self.action_bender.triggered.connect(self.bender)

    def bender(self):
        """
        @author:    José Miguel Algarín
        @email:     josalggui@i3m.upv.es
        @affiliation:MRILab, i3M, CSIC, Valencia, Spain
        # Summary: it runs a full protocol with a single click
        """

        # Larmor calibration
        defaultsequences['Larmor'].loadParams(directory="calibration", file="Larmor_last_parameters.csv")
        self.runToList('Larmor')
        time.sleep(0.1)

        # Noise measurement
        defaultsequences['Noise'].loadParams(directory="calibration", file="Noise_last_parameters.csv")
        self.runToList('Noise')
        time.sleep(0.1)

        # Rabi flops
        defaultsequences['RabiFlops'].loadParams(directory="calibration", file="RabiFlops_last_parameters.csv")
        self.runToList('RabiFlops')
        time.sleep(0.1)

        # Shimming
        defaultsequences['Shimming'].loadParams(directory="calibration", file="ShimmingSweep_last_parameters.csv")
        self.runToList('Shimming')
        time.sleep(0.1)

        # Larmor calibration
        defaultsequences['Larmor'].loadParams(directory="calibration", file="Larmor_last_parameters.csv")
        self.runToList('Larmor')
        time.sleep(0.1)

        # First image sequence
        defaultsequences['RARE'].loadParams(directory="automatic/bender", file="RARE01.csv")
        self.runToList('RARE')
        time.sleep(0.1)
    def autocalibration(self):
        self.main.figures_layout.clearFiguresLayout()

        # Include here the sequences to run on autocalibration
        seq_names = [
            'Larmor',
            'Noise',
            'RabiFlops',
            'Shimming'
        ]

        # Add plots to the plotview_layout
        self.plots = []

        for seq_name in seq_names:
            # Execute the sequence
            sequence = defaultsequences[seq_name]
            sequence.sequenceRun()
            output = sequence.sequenceAnalysis(obj='autocalibration')
            delattr(sequence, 'out')

            # Add item to the history list
            file_name = sequence.mapVals['fileName']
            name = str(datetime.now())[11:23] + " | " + file_name
            self.main.history_list.addItem(name)

            # Save results into the history
            self.main.history_list.outputs[name[0:12]] = output
            self.main.history_list.inputs[name[0:12]] = [list(sequence.mapNmspc.values()),
                                                         list(sequence.mapVals.values()),
                                                         False]

            # Specific for larmor
            if seq_name == 'Larmor':
                for seq in defaultsequences.values():
                    seq.mapVals['larmorFreq'] = hw_config.larmorFreq

            # Specific for noise
            if seq_name == 'Noise':
                # Create label with rawdata name
                self.label = QLabel()
                self.label.setAlignment(QtCore.Qt.AlignCenter)
                self.label.setStyleSheet("background-color: black;color: white")
                self.label.setText(sequence.mapVals['fileName'])
                self.main.figures_layout.addWidget(self.label, row=0, col=0, colspan=2)

                # Noise spectrum
                item = output[1]
                self.plots.append(SpectrumPlot(x_data=item['xData'],
                                               y_data=item['yData'],
                                               legend=item['legend'],
                                               x_label=item['xLabel'],
                                               y_label=item['yLabel'],
                                               title=item['title']))
                self.main.figures_layout.addWidget(self.plots[-1], row=1, col=0)

            # Specific for rabi
            if seq_name == 'RabiFlops':
                item = output[0]
                self.plots.append(SpectrumPlot(x_data=item['xData'],
                                               y_data=item['yData'],
                                               legend=item['legend'],
                                               x_label=item['xLabel'],
                                               y_label=item['yLabel'],
                                               title=item['title']))
                self.main.figures_layout.addWidget(self.plots[-1], row=2, col=0)

            # Specific for shimming
            if seq_name == 'Shimming':
                for seq in defaultsequences.values():
                    seq.mapVals['shimming'] = output[1]
                item = output[0]
                self.plots.append(SpectrumPlot(x_data=item['xData'],
                                               y_data=item['yData'],
                                               legend=item['legend'],
                                               x_label=item['xLabel'],
                                               y_label=item['yLabel'],
                                               title=item['title']))
                self.main.figures_layout.addWidget(self.plots[-1], row=3, col=0)

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

        # Save the rotation and shift to the history list
        self.main.history_list.rotations[name[0:12]] = defaultsequences[seq_name].rotations.copy()
        self.main.history_list.shifts[name[0:12]] = defaultsequences[seq_name].dfovs.copy()
        self.main.history_list.fovs[name[0:12]] = defaultsequences[seq_name].fovs.copy()

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
