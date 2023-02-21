"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import copy
import time

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QSize, QSizeF
from PyQt5.QtWidgets import QTableWidgetItem, QLabel, QMenu, QAction

from controller.controller_plot3d import Plot3DController as Spectrum3DPlot
from controller.controller_plot1d import Plot1DController as SpectrumPlot
from seq.sequences import defaultsequences
from widgets.widget_history_list import HistoryListWidget


class HistoryListController(HistoryListWidget):
    def __init__(self, *args, **kwargs):
        super(HistoryListController, self).__init__(*args, **kwargs)
        self.clicked_item = None
        self.fovs = {}
        self.shifts = {}
        self.rotations = {}
        self.outputs = {}
        self.inputs = {}
        self.current_output = None
        self.figures = []
        self.labels = []

        # Connect methods to item click
        self.itemDoubleClicked.connect(self.updateHistoryFigure)
        self.itemClicked.connect(self.updateHistoryTable)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)

    def showContextMenu(self, point):
        self.clicked_item = self.itemAt(point)
        if self.clicked_item is not None:
            menu = QMenu(self)
            action1 = QAction("Add new image", self)
            action1.triggered.connect(self.addNewFigure)
            menu.addAction(action1)
            action1 = QAction("Add another image", self)
            action1.triggered.connect(self.addFigure)
            menu.addAction(action1)

            selected_action = menu.exec_(self.mapToGlobal(point))

    def addNewFigure(self):
        self.figures = []
        self.labels = []
        self.addFigure()

    def addFigure(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: add another figure from the history list
        """
        self.main.figures_layout.clearFiguresLayout()
        if len(self.figures) > 3:
            print("You can add only 4 figures to the layout")
            return

        # Get file self.current_output
        file_name = self.clicked_item.text()[15::]
        self.current_output = self.clicked_item.text()[0:12]

        # Get the widget from history
        output = self.outputs[self.current_output]

        # Get rotations and shifts from history
        rotations = self.rotations[self.current_output]
        shifts = self.shifts[self.current_output]
        fovs = self.fovs[self.current_output]
        for sequence in defaultsequences.values():
            sequence.rotations = rotations.copy()
            sequence.dfovs = shifts.copy()
            sequence.fovs = fovs.copy()

        # Create label and figure
        # Create image widget
        item = output[0]
        if item['widget'] == 'image':
            self.figures.append(item)
            self.labels.append(file_name)
        else:
            print("The selected raw data does not contain an image")

        # Create the new layout
        n = 0
        sub_label = QLabel('Multiplot')
        sub_label.setAlignment(QtCore.Qt.AlignCenter)
        sub_label.setStyleSheet("background-color: black;color: white")
        if len(self.figures) > 1:
            self.main.figures_layout.addWidget(sub_label, row=0, col=0, colspan=2)
        for row in range(2):
            for col in range(2):
                try:
                    image = Spectrum3DPlot(main=self.main,
                                           data=self.figures[n]['data'],
                                           x_label=self.figures[n]['xLabel'],
                                           y_label=self.figures[n]['yLabel'],
                                           title=self.figures[n]['title'])
                    label = QLabel()
                    label.setAlignment(QtCore.Qt.AlignCenter)
                    label.setStyleSheet("background-color: black;color: white")
                    label.setText(self.labels[n])
                    self.main.figures_layout.addWidget(label, row=2 * row + 1, col=col)
                    self.main.figures_layout.addWidget(image, row=2 * row + 2, col=col)
                except:
                    pass
                n += 1

    def updateHistoryTable(self, item):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: update the table when new element is clicked in the history list
        """
        # Get file name
        name = item.text()[0:12]

        # Get the input data from history
        input_data = self.inputs[name]

        # Extract items from the input_data
        input_info = list(input_data[0])
        input_vals = list(input_data[1])

        # Set number of rows
        self.main.input_table.setColumnCount(1)
        self.main.input_table.setRowCount(len(input_info))

        # Input items into the table
        self.main.input_table.setVerticalHeaderLabels(input_info)
        self.main.input_table.setHorizontalHeaderLabels(['Values'])
        for m, item in enumerate(input_vals):
            new_item = QTableWidgetItem(str(item))
            self.main.input_table.setItem(m, 0, new_item)

    def updateHistoryFigure(self, item):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: update the shown figure when new element is double clicked in the history list
        """
        # Get file self.current_output
        file_name = item.text()[15::]
        self.current_output = item.text()[0:12]

        # Get the widget from history
        output = self.outputs[self.current_output]

        # Get rotations and shifts from history
        rotations = self.rotations[self.current_output]
        shifts = self.shifts[self.current_output]
        fovs = self.fovs[self.current_output]
        for sequence in defaultsequences.values():
            sequence.rotations = rotations.copy()
            sequence.dfovs = shifts.copy()
            sequence.fovs = fovs.copy()

        # Clear the plotview
        self.main.figures_layout.clearFiguresLayout()

        # Add label to show rawData self.current_output
        label = QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("background-color: black;color: white")
        self.main.figures_layout.addWidget(label, row=0, col=0, colspan=2)
        label.setText(file_name)

        for item in output:
            if item['widget'] == 'image':
                image = Spectrum3DPlot(main=self.main,
                                       data=item['data'],
                                       x_label=item['xLabel'],
                                       y_label=item['yLabel'],
                                       title=item['title'])
                self.main.figures_layout.addWidget(image, row=item['row'] + 1, col=item['col'])
            elif item['widget'] == 'curve':
                plot = SpectrumPlot(x_data=item['xData'],
                                    y_data=item['yData'],
                                    legend=item['legend'],
                                    x_label=item['xLabel'],
                                    y_label=item['yLabel'],
                                    title=item['title'])
                self.main.figures_layout.addWidget(plot, row=item['row'] + 1, col=item['col'])

    def waitingForRun(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: this method is continuously waiting for running new sequences in the history_list
        """
        while self.main.app_open:
            keys = list(self.inputs.keys())  # List of elements in the sequence history list
            element = 0
            for key in keys:
                if self.inputs[key][2]:
                    # Disable acquire button
                    self.main.toolbar_sequences.action_acquire.setEnabled(False)

                    # Get the sequence to run
                    seq_name = self.inputs[key][1][0]
                    sequence = copy.copy(defaultsequences[seq_name])
                    # Modify input parameters of the sequence
                    n = 0
                    input_list = list(sequence.mapVals.keys())
                    for keyParam in input_list:
                        sequence.mapVals[keyParam] = self.inputs[key][1][n]
                        n += 1
                    # Run the sequence
                    output = self.runSequenceInlist(sequence=sequence)
                    # Add item to the history list
                    file_name = sequence.mapVals['fileName']
                    self.item(element).setText(key + " | " + file_name)
                    # self.history_list.addItem()
                    # Save results into the history
                    self.outputs[key] = output
                    self.inputs[key] = [list(defaultsequences[seq_name].mapNmspc.values()),
                                        list(defaultsequences[seq_name].mapVals.values()),
                                        False]
                    # Delete outputs from the sequence
                    sequence.resetMapVals()
                    print(key + " Done!")
                else:
                    # Enable acquire button
                    self.main.toolbar_sequences.action_acquire.setEnabled(True)
                element += 1
            time.sleep(1)

        return 0

    @staticmethod
    def runSequenceInlist(sequence=None):
        # Save sequence list into the current sequence, just in case you need to do sweep
        sequence.sequenceList = defaultsequences

        # Save input parameters
        sequence.saveParams()

        # Create and execute selected sequence
        sequence.sequenceRun(0)

        time.sleep(1)

        # Do sequence analysis and get results
        return sequence.sequenceAnalysis()
