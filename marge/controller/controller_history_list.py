"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import copy
import os
import time
import datetime as dt

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QTableWidgetItem, QLabel, QMenu, QAction

from controller.controller_plot3d import Plot3DController as Spectrum3DPlot
from marge.controller.controller_plot1d import Plot1DController as SpectrumPlot
from controller.controller_smith_chart import PlotSmithChartController as SmithChart
try:
    from marge.seq.sequences import defaultsequences
except:
    pass
from marge.widgets.widget_history_list import HistoryListWidget
from marge.manager.dicommanager import DICOMImage
from marge.marge_utils import utils
import numpy as np
import marge.configs.hw_config as hw
import nibabel as nib

class HistoryListController(HistoryListWidget):
    """
    Controller for the history list.
    """

    sequence_ready_signal = QtCore.pyqtSignal(object)
    figure_ready_signal = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        """
        Controller for the history list.

        It has a dictionary with the input and outputs of executed and pending sequences.
        """
        super(HistoryListController, self).__init__(*args, **kwargs)
        self.clicked_item = None
        self.fovs = {}
        self.shifts = {}
        self.rotations = {}
        self.outputs = {}
        self.inputs = {}
        self.pending_inputs = {}
        self.current_output = None
        self.figures = []
        self.labels = []

        # Connect methods to item click
        self.itemDoubleClicked.connect(self.updateHistoryFigure)
        self.itemClicked.connect(self.updateHistoryTable)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.itemChanged.connect(self.main.sequence_list.updateSequence)
        self.itemEntered.connect(self.main.sequence_list.updateSequence)

    def delete_items(self):
        while self.count() > 0:
            self.deleteTask(item_number=0)

    def showContextMenu(self, point):
        """
        Displays a context menu at the given point.

        :param point: The position where the context menu should be displayed.
        """

        self.clicked_item = self.itemAt(point)
        if self.clicked_item is not None:
            menu = QMenu(self)
            action1 = QAction("Add new image", self)
            action1.triggered.connect(self.addNewFigure)
            menu.addAction(action1)
            action2 = QAction("Add another image", self)
            action2.triggered.connect(self.addFigure)
            menu.addAction(action2)
            action3 = QAction("Delete task", self)
            action3.triggered.connect(self.deleteTask)
            menu.addAction(action3)
            action4 = QAction("Post-processing", self)
            action4.triggered.connect(self.openPostGui)
            menu.addAction(action4)

            menu.exec_(self.mapToGlobal(point))

    def openPostGui(self):
        # Get the corresponding key to get access to the history dictionary
        item_name = self.clicked_item.text().split(' | ')[1]
        path = self.main.session['directory']
        self.main.post_gui.showMaximized()
        self.main.post_gui.toolbar_image.rawDataLoading(file_path=path + "/mat/", file_name=item_name)

    def deleteTask(self, item_number=None):
        """
        Deletes the currently selected task from the list.

        This method removes the currently selected task item from the list widget.
        """
        if item_number is None or item_number is False:
            item = self.currentItem()
        else:
            item = self.item(item_number)
        text = item.text()
        text = text.split(".", maxsplit=2)
        key = text[0] + "." + text[1]
        self.takeItem(self.row(item))
        if key in self.pending_inputs.keys():
            self.pending_inputs.pop(text)
        if key in self.inputs.keys():
            self.inputs.pop(key)

    def addNewFigure(self):
        """
        Adds a new figure and initializes the figures and labels lists.

        This method adds a new figure and initializes the `figures` and `labels` lists to empty lists.
        It then calls the `addFigure` method to add the new figure to the list.
        """
        self.figures = []
        self.labels = []
        self.addFigure()

    def addFigure(self):
        """
        Adds a figure to the layout and updates the figures and labels lists.

        This method clears the figures layout, checks the number of existing figures, and returns early if the maximum
        limit of 4 figures is reached.
        It retrieves information from the clicked item, such as the time and name, and assigns it to
        `self.current_output`.
        The method then fetches the relevant data and configurations from the history based on `self.current_output`.
        It creates the label and figure for the image widget, adds them to the figures layout, and updates the figures
        and labels lists.

        Note: The figures layout is assumed to be available as `self.main.figures_layout`.

        If the selected raw data does not contain an image, a message is printed.

        Raises:
            - Exception: An exception may be raised if there is an error creating a Spectrum3DPlot.

        """
        self.main.figures_layout.clearFiguresLayout()
        if len(self.figures) > 3:
            print("You can add only 4 figures to the layout")
            return

        # Get clicked item self.current_output
        item_time = self.clicked_item.text().split(' | ')[0]
        item_name = self.clicked_item.text().split(' | ')[1].split('.')[0]
        self.current_output = item_time + " | " + item_name

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
            self.labels.append(item_name)
        else:
            print("The selected raw data does not contain an image")

        # Create the new layout
        n = 0
        sub_label = QLabel('Multiplot')
        sub_label.setAlignment(QtCore.Qt.AlignCenter)
        sub_label.setStyleSheet("background-color: black;color: white")
        # if len(self.figures) > 1:
        #     self.main.figures_layout.addWidget(sub_label, row=0, col=0, colspan=2)
        for col in range(4):
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
                self.main.figures_layout.addWidget(label, row=1, col=col)
                self.main.figures_layout.addWidget(image, row=2, col=col)
                self.main.figures_layout.addWidget(sub_label, row=0, col=0, colspan=col + 1)
            except:
                pass
            n += 1

    def updateHistoryTable(self, item):
        """
        Updates the history table with input data corresponding to the selected item.

        This method takes an item as input, retrieves the corresponding key from the item's text, and accesses the
        history dictionary using the key.
        It extracts the input data from the history and separates it into input_info and input_vals.
        The method sets the number of rows in the main input_table and populates it with the input_info and input_vals.
        The input_info is used as the vertical header labels, and 'Values' is set as the horizontal header label.

        Note: The main input_table is assumed to be available as self.main.input_table.

        :param item: The selected item from which to retrieve the corresponding input data.
        """

        # Get the corresponding key to get access to the history dictionary
        item_time = item.text().split(' | ')[0]
        item_name = item.text().split(' | ')[1].split('.')[0]
        name = item_time + " | " + item_name

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

    def updateHistoryFigure(self, item=None):
        """
        Updates the history figure based on the selected item.

        This method takes an item as input, retrieves the corresponding key from the item's text, and assigns it to `self.current_output`.
        It accesses the history dictionary using `self.current_output` to retrieve the output widget information.
        If available, it also retrieves the rotations, shifts, and field of views (fovs) from the history.
        The method then clears the plot view by calling `self.main.figures_layout.clearFiguresLayout()`.
        It adds a label to show the rawData corresponding to the selected item at the top of the figures layout.
        Finally, it iterates through the output items, adds either a Spectrum3DPlot or SpectrumPlot widget to the figures layout based on the item's widget type and populates it with the relevant data and configurations.

        Note: The figures layout is assumed to be available as `self.main.figures_layout`.

        :param item: The selected item from which to retrieve the corresponding output information.
        """

        # Get the corresponding key to get access to the history dictionary
        if item is None:
            item = self.item(self.count() - 1)
        item_time = item.text().split(' | ')[0]
        item_name = item.text().split(' | ')[1].split('.')[0]
        self.current_output = item_time + " | " + item_name

        # Get the widget from history
        output = self.outputs[self.current_output]

        # Get rotations and shifts from history
        try:
            rotations = self.rotations[self.current_output]
            shifts = self.shifts[self.current_output]
            fovs = self.fovs[self.current_output]
            for sequence in defaultsequences.values():
                sequence.rotations = rotations.copy()
                sequence.dfovs = shifts.copy()
                sequence.fovs = fovs.copy()
        except:
            pass

        # Clear the plotview
        self.main.figures_layout.clearFiguresLayout()

        # Add label to show rawData self.current_output
        label = QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("background-color: black;color: white")
        self.main.figures_layout.addWidget(label, row=0, col=0, colspan=2)
        label.setText(item.text().split(' | ')[1])

        # Add plots to the plotview_layout
        n_columns = 1
        for item in output:
            if item['col'] + 1 > n_columns:
                n_columns = item['col'] + 1
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
            elif item['widget'] == 'smith':
                plot = SmithChart(x_data=item['xData'],
                                  y_data=item['yData'],
                                  legend=item['legend'],
                                  x_label=item['xLabel'],
                                  y_label=item['yLabel'],
                                  title=item['title'])
                self.main.figures_layout.addWidget(plot, row=item['row'] + 1, col=item['col'])
        self.main.figures_layout.addWidget(label, row=0, col=0, colspan=n_columns)

    def updateHistoryFigure2(self, item=None):
        """
        Updates the history figure based on the selected item.

        This method takes an item as input, retrieves the corresponding key from the item's text, and assigns it to `self.current_output`.
        It accesses the history dictionary using `self.current_output` to retrieve the output widget information.
        If available, it also retrieves the rotations, shifts, and field of views (fovs) from the history.
        The method then clears the plot view by calling `self.main.figures_layout.clearFiguresLayout()`.
        It adds a label to show the rawData corresponding to the selected item at the top of the figures layout.
        Finally, it iterates through the output items, adds either a Spectrum3DPlot or SpectrumPlot widget to the figures layout based on the item's widget type and populates it with the relevant data and configurations.

        Note: The figures layout is assumed to be available as `self.main.figures_layout`.

        :param item: The selected item from which to retrieve the corresponding output information.
        """

        # Get the corresponding key to get access to the history dictionary
        if item is None:
            item = self.item(self.count() - 1)
        item_time = item.text().split(' | ')[0]
        item_name = item.text().split(' | ')[1].split('.')[0]
        self.current_output = item_time + " | " + item_name

        # Get the widget from history
        output = self.outputs[self.current_output]

        # Get rotations and shifts from history
        try:
            rotations = self.rotations[self.current_output]
            shifts = self.shifts[self.current_output]
            fovs = self.fovs[self.current_output]
            for sequence in defaultsequences.values():
                sequence.rotations = rotations.copy()
                sequence.dfovs = shifts.copy()
                sequence.fovs = fovs.copy()
        except:
            pass

        # Clear the plotview
        self.main.figures_layout.clearFiguresLayout()

        # Add label to show rawData self.current_output
        label = QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("background-color: black;color: white")
        self.main.figures_layout.addWidget(label, row=0, col=0, colspan=2)
        label.setText(item.text().split(' | ')[1])

        # Add plots to the plotview_layout
        n_columns = 1
        for item in output:
            if item['col'] + 1 > n_columns:
                n_columns = item['col'] + 1
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
            elif item['widget'] == 'smith':
                plot = SmithChart(x_data=item['xData'],
                                  y_data=item['yData'],
                                  legend=item['legend'],
                                  x_label=item['xLabel'],
                                  y_label=item['yLabel'],
                                  title=item['title'])
                self.main.figures_layout.addWidget(plot, row=item['row'] + 1, col=item['col'])
        self.main.figures_layout.addWidget(label, row=0, col=0, colspan=n_columns)

        if 'Calibration' in item_name:
            QTimer.singleShot(0, self.figure_ready_signal.emit)

    def waitingForRun(self):
        """
        Wait for the run to start.

        This method waits until the main application is open and the server action is checked in the toolbar.
        It then iterates over the pending inputs and runs the corresponding sequence for each input. After running the
        sequence, it handles the output and updates the history list accordingly.

        This method is executed in a parallel thread.

        Returns:
            int: The value 0 indicating the completion of the method.
        """
        while self.main.app_open:
            if self.main.toolbar_marcos.action_server.isChecked():
                pending_keys = list(self.pending_inputs.keys())  # List of elements in the pending sequence list
                keys = list(self.inputs.keys())  # List of elements in the sequence history list
                for key in pending_keys:
                    # Disable acquire button
                    self.main.toolbar_sequences.action_acquire.setEnabled(False)

                    # Get the sequence to run
                    seq_name = self.pending_inputs[key][1][0]
                    sequence = copy.deepcopy(defaultsequences[seq_name])

                    # Modify input parameters of the sequence according to current item
                    n = 0
                    for keyParam in sequence.mapKeys:
                        sequence.mapVals[keyParam] = self.pending_inputs[key][1][n]
                        n += 1

                    # Specific tasks for calibration
                    if "Calibration" in key:
                        if seq_name == 'Larmor':
                            sequence.mapVals['larmorFreq'] = hw.larmorFreq
                            try:
                                sequence.mapVals['shimming'] = defaultsequences['Shimming'].mapVals['shimming']
                            except:
                                pass
                        if seq_name == 'RabiFlops':
                            sequence.mapVals['shimming'] = defaultsequences['Shimming'].mapVals['shimming']

                    # Run the sequence
                    key_index = keys.index(key)
                    raw_data_name = key.split('|')[1].split(' ')[1]
                    output = self.runSequenceInlist(sequence=sequence, key=key, raw_data_name=raw_data_name)
                    if output == 0:
                        # There is an error
                        del self.pending_inputs[key]
                        print("ERROR: " + key + " sequence finished abruptly with error.\n")
                    else:
                        # Add item to the history list
                        file_name = sequence.mapVals['fileName']
                        date = ".".join(file_name.split('.')[1::])
                        self.item(key_index).setText(self.item(key_index).text() + "." + date)
                        # Save results into the history
                        self.outputs[key] = output
                        del self.pending_inputs[key]
                        # Delete outputs from the sequence
                        sequence.resetMapVals()
                        # self.main.sequence_list.updateSequence()
                        print("READY: " + key + "\n")
                        self.sequence_ready_signal.emit(self.item(key_index))
                    time.sleep(0.5)
                # Enable acquire button
                if self.main.toolbar_marcos.action_server.isChecked():
                    self.main.toolbar_sequences.action_acquire.setEnabled(True)
            time.sleep(0.1)
        return 0

    def runSequenceInlist(self, sequence=None, key=None, raw_data_name=""):
        """
        Run a sequence in the list.

        This method executes a given sequence in the list. It saves the sequence list, input parameters, and updates
        the rotation, field of view (FOV), and dynamic field of view (dFOV) values. The sequence is then executed, and
        afterwards, sequence analysis is performed to retrieve the results.

        Args:
            sequence (object): The sequence object to be executed.
            key (str): The key associated with the sequence to get previous rotations, shifts and fovs.
            raw_data_name (str): The name of the raw data to be included in the file name.

        Returns:
            object: The result of the sequence analysis.
        """
        # Save sequence list into the current sequence, just in case you need to do sweep
        sequence.sequence_list = defaultsequences
        sequence.raw_data_name = raw_data_name

        # Save input parameters
        sequence.saveParams()

        # Update possible rotation, fov and dfov before the sequence is executed in parallel thread
        sequence.sequenceAtributes()
        self.rotations[key] = sequence.rotations.copy()
        self.shifts[key] = sequence.dfovs.copy()
        self.fovs[key] = sequence.fovs.copy()

        # Create and execute selected sequence
        try:
            if sequence.sequenceRun(0, self.main.demo):
                pass
            else:
                return 0
        except Exception as e:
            print(f"An error occurred in sequenceRun method: {e}")
            return 0

        # Do sequence analysis and get results
        try:
            return sequence.sequenceAnalysis()
        except Exception as e:
            print(f"An error ocurred in sequenceAnalysis method: {e}")
            return 0


class HistoryListControllerPos(HistoryListWidget):
    """
    @AUTHOR: D. Comlan, MRILab, CSIC.
    @AUTHOR: J.M. Algarín, MRILab, CSIC

    Controller class for the history list widget.

    Inherits from HistoryListWidget.

    Attributes:
        image_hist: Dictionary to store images.
        operations_hist: Dictionary to store operations' history.
        image_key: Information about the matrix.
        image_view: Reference to the ImageViewWidget.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the HistoryListController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super(HistoryListControllerPos, self).__init__(*args, **kwargs)
        self.labels = None
        self.figures = None
        self.orientations = None
        self.image_hist = {}  # Dictionary to store historical images
        self.image_orientation = {}
        self.operations_hist = {}  # Dictionary to store operations history
        self.space = {}  # Dictionary to retrieve if matrix is in k-space or image-space
        self.image_key = None
        self.image_view = None

        # Connect methods to item click
        self.itemDoubleClicked.connect(self.updateHistoryFigure)
        self.itemClicked.connect(self.updateHistoryTable)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)

    def showContextMenu(self, point):
        """
        Displays a context menu at the given point.

        :param point: The position where the context menu should be displayed.
        """
        self.clicked_item = self.itemAt(point)
        if self.clicked_item is not None:
            menu = QMenu(self)
            action1 = QAction("Add new image", self)
            action1.triggered.connect(self.addNewFigure)
            menu.addAction(action1)
            action2 = QAction("Add another image", self)
            action2.triggered.connect(self.addFigure)
            menu.addAction(action2)
            action3 = QAction("Delete image", self)
            action3.triggered.connect(self.deleteSelectedItem)
            menu.addAction(action3)
            action4 = QAction("Save figure", self)
            action4.triggered.connect(self.saveImage)
            menu.addAction(action4)

            menu.exec_(self.mapToGlobal(point))

    def addNewFigure(self):
        """
        Adds a new figure and initializes the figures and labels lists.

        This method adds a new figure and initializes the `figures` and `labels` lists to empty lists.
        It then calls the `addFigure` method to add the new figure to the list.
        """
        self.figures = []
        self.orientations = []
        self.labels = []
        self.addFigure()

    def addFigure(self):
        """
        Adds a figure to the layout and updates the figures and labels lists.

        This method clears the figures layout, checks the number of existing figures, and returns early if the maximum
        limit of 4 figures is reached.
        It retrieves information from the clicked item, such as the time and name, and assigns it to
        `self.current_output`.
        The method then fetches the relevant data and configurations from the history based on `self.current_output`.
        It creates the label and figure for the image widget, adds them to the figures layout, and updates the figures
        and labels lists.

        Note: The figures layout is assumed to be available as `self.main.figures_layout`.

        If the selected raw data does not contain an image, a message is printed.

        Raises:
            - Exception: An exception may be raised if there is an error creating a Spectrum3DPlot.

        """
        self.main.image_view_widget.clearFiguresLayout()
        if len(self.figures) > 7:
            print("You can add only 8 figures to the layout")
            return

        # Get clicked item self.current_output
        selected_items = self.selectedItems()
        if selected_items:
            selected_item = selected_items[0]
            image_key = selected_item.text()
            if image_key in self.image_hist:
                self.orientations.append(self.image_orientation.get(image_key))
                if self.space[image_key] == 'k':
                    image = np.log10(np.abs(self.image_hist.get(image_key)))
                    image[image == -np.inf] = np.inf
                    val = np.min(image[:])
                    image[image == np.inf] = val
                else:
                    image = np.abs(self.image_hist[image_key])

                # Add image and label to the list
                self.figures.append(image)
                self.labels.append(image_key)

                # self.main.image_view_widget.addWidget(image, row=0, col=0)

        # Create the new layout
        n = 0
        sub_label = QLabel('Multiplot')
        sub_label.setAlignment(QtCore.Qt.AlignCenter)
        sub_label.setStyleSheet("background-color: black;color: white")
        ncol = 0
        for idx in range(8):
            try:
                # Label
                label = QLabel(self.labels[n])
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setStyleSheet("background-color: black;color: white")
                self.main.image_view_widget.addWidget(label, row=2 * (idx // 4) + 1, col=idx % 4)

                # Figure
                image2show, x_label, y_label, title = self.main.toolbar_image.fixImage(self.figures[n],
                                                                                       orientation=self.orientations[n])
                image = Spectrum3DPlot(main=self.main,
                                       data=image2show,
                                       x_label=x_label,
                                       y_label=y_label,
                                       title=title)
                self.main.image_view_widget.addWidget(image, row=2 * (idx // 4) + 2, col=idx % 4)

                ncol = np.max([ncol, idx % 4 + 1])
                self.main.image_view_widget.addWidget(sub_label, row=0, col=0, colspan=ncol)
            except:
                pass
            n += 1

    def addNewItem(self, image_key=None, stamp=None, image=None, orientation=None, operation=None, space=None):
        # Generate the image key
        current_time = dt.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self.image_key = f"{current_time} - {stamp}"

        # Add the item to the history list
        self.addItem(self.image_key)

        # Update the history dictionary with the new main matrix
        self.image_hist[self.image_key] = image
        self.image_orientation[self.image_key] = orientation

        # Update the operations history
        if len(self.operations_hist) == 0 or image_key is None:
            self.operations_hist[self.image_key] = [operation]
        else:
            operations = self.operations_hist[image_key]
            operations = operations.copy()
            operations.append(operation)
            self.operations_hist[self.image_key] = operations
        self.main.image_view_widget.image_key = self.image_key

        # Update the space dictionary
        self.space[self.image_key] = space

        return 0

    def updateHistoryFigure(self, item):
        """
        Update the displayed image based on the selected item in the history list.

        Args:
            item (QListWidgetItem): The selected item in the history list.
        """

        image_key = item.text()
        if image_key in self.image_hist.keys():
            self.main.image_view_widget.main_matrix = self.image_hist[image_key]
            self.main.image_view_widget.image_key = image_key
            orientation = self.image_orientation[image_key]
            if self.space[image_key] == 'k':
                image = np.log10(np.abs(self.main.image_view_widget.main_matrix))
                image[image == -np.inf] = np.inf
                val = np.min(image[:])
                image[image == np.inf] = val
            else:
                image = np.abs(self.main.image_view_widget.main_matrix)

            # Delete all widgets from image_view_widget
            self.main.image_view_widget.clearFiguresLayout()

            # Create label widget
            label = QLabel()
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet("background-color: black;color: white")
            self.main.image_view_widget.addWidget(label, row=0, col=0, colspan=2)
            label.setText(image_key)

            # Create image_widget
            image2show, x_label, y_label, title = self.main.toolbar_image.fixImage(image, orientation=orientation)
            image = Spectrum3DPlot(main=self.main,
                                   data=image2show,
                                   x_label=x_label,
                                   y_label=y_label,
                                   title=title)

            # Add widgets to the figure layout
            self.main.image_view_widget.addWidget(label, row=0, col=0)
            self.main.image_view_widget.addWidget(image, row=1, col=0)

    def updateOperationsHist(self, image_key, text, new=False):
        """
        Update the operations history dictionary with the given information.

        Args:
            image_key (str): Information for the operations' history.
            text (str): Text to be added to the operations' history.
            new (bool): True is new loaded image
        """
        if len(self.operations_hist) == 0 or new is True:
            self.operations_hist[image_key] = [text]
        else:
            list1 = list(self.operations_hist.values())
            new_value = list1[-1].copy()
            new_value.append(text)
            self.operations_hist[image_key] = new_value

    def updateHistoryTable(self, item):
        """
        Update the operations history table based on the selected item in the history list.

        Args:
            item (QListWidgetItem): The selected item in the history list.
        """

        # Clear the methods_list table
        self.main.methods_list.setText('')

        # Get the values to show in the methods_list table
        selected_text = item.text()
        values = self.operations_hist.get(selected_text, [])

        # Print the methods
        for value in values:
            self.main.methods_list.append(value)

    def moveKeyAndValuesToEnd(self, dictionary, key):
        """
        Move the given key and its associated values to the end of the dictionary.

        Args:
            dictionary (dict): The dictionary containing the key and values.
            key (str): The key to be moved to the end of the dictionary.
        """
        if key in dictionary:
            values = dictionary[key]
            del dictionary[key]
            dictionary[key] = values

    def saveImage(self):
        # Path to the DICOM file
        path = self.main.session['directory'] + "/dcm/" + self.main.file_name[0:-4]
        if not os.path.exists(self.main.session['directory'] + "/dcm/"):
            os.makedirs(self.main.session['directory'] + "/dcm/")

        # Load the DICOM file
        dicom_image = DICOMImage(path=path + ".dcm")

        # Get image to save into dicom
        image = self.main.image_view_widget.main_matrix
        imageDICOM = np.transpose(image, (0, 2, 1))
        slices, rows, columns = imageDICOM.shape
        dicom_image.meta_data["Columns"] = columns
        dicom_image.meta_data["Rows"] = rows
        dicom_image.meta_data["NumberOfSlices"] = slices
        dicom_image.meta_data["NumberOfFrames"] = slices
        imgFullAbs = np.abs(imageDICOM) * (2 ** 15 - 1) / np.amax(np.abs(imageDICOM))
        imgFullInt = np.int16(np.abs(imgFullAbs))
        imgFullInt = np.reshape(imgFullInt, (slices, rows, columns))
        dicom_image.meta_data["PixelData"] = imgFullInt.tobytes()

        # Save meta_data dictionary into dicom object metadata (Standard DICOM 3.0)
        dicom_image.image2Dicom()

        # Generate date to add to the name
        name = dt.datetime.now()
        name_string = name.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3]

        # Save dicom file
        dicom_image.save(path + "_" + name_string + ".dcm")

        print("Dicom image saved")

        # Save nifti
        path = self.main.session['directory'] + "/nii/" + self.main.file_name[0:-4]
        if not os.path.exists(self.main.session['directory'] + "/nii/"):
            os.makedirs(self.main.session['directory'] + "/nii/")
        utils.save_nifti(axes_orientation = self.main.toolbar_image.mat_data['axesOrientation'][0],
                         n_points = list(image.shape)[::-1],
                         fov = self.main.toolbar_image.mat_data['fov'][0],
                         dfov = self.main.toolbar_image.mat_data['dfov'][0],
                         image = image,
                         file_path = path + "_" + name_string + ".nii")
        print("Nifti image saved")

        return 0

    def deleteSelectedItem(self):
        """
        Delete the selected item from the history list.
        """
        selected_items = self.selectedItems()
        if selected_items:
            selected_item = selected_items[0]
            self.takeItem(self.row(selected_item))

        if selected_item.text() in self.image_hist:
            del self.image_hist[selected_item.text()]

        if selected_item.text() in self.operations_hist:
            del self.operations_hist[selected_item.text()]

    # def plotPhase(self):
    #     selected_items = self.selectedItems()
    #     if selected_items:
    #         selected_item = selected_items[0]
    #         text = selected_item.text()
    #         if text in self.image_hist:
    #             image = self.image_hist.get(text)
    #             if self.image_view is None:
    #                 self.image_view = ImageViewWidget(parent=self.main)
    #                 self.main.image_view_splitter.addWidget(self.image_view)
    #         self.image_view.setImage(np.angle(image))
