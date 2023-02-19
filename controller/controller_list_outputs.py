"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5 import QtCore
from PyQt5.QtWidgets import QTableWidgetItem, QLabel

from plotview.spectrumplot import Spectrum3DPlot, SpectrumPlot
from seq.sequences import defaultsequences
from widgets.widget_list_outputs import OutputListWidget


class OutputListController(OutputListWidget):
    def __init__(self, *args, **kwargs):
        super(OutputListController, self).__init__(*args, **kwargs)
        self.history_list_fovs = {}
        self.history_list_shifts = {}
        self.history_list_rotations = {}
        self.history_list_outputs = {}
        self.history_list_inputs = {}
        self.current_output = None
        
        self.itemDoubleClicked.connect(self.updateHistoryFigure)
        self.itemClicked.connect(self.updateHistoryTable)

    def updateHistoryTable(self, item):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: update the table when new element is clicked in the history list
        """
        # Get file name
        name = item.text()[0:12]

        # Get the input data from history
        input_data = self.history_list_inputs[name]

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
        output = self.history_list_outputs[self.current_output]

        # Get rotations and shifts from history
        rotations = self.history_list_rotations[self.current_output]
        shifts = self.history_list_shifts[self.current_output]
        fovs = self.history_list_fovs[self.current_output]
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
                image = Spectrum3DPlot(data=item['data'],
                                       xLabel=item['xLabel'],
                                       yLabel=item['yLabel'],
                                       title=item['title'])
                image.parent = self
                self.main.figures_layout.addWidget(image.getImageWidget(), row=item['row'] + 1, col=item['col'])
            elif item['widget'] == 'curve':
                plot = SpectrumPlot(xData=item['xData'],
                                    yData=item['yData'],
                                    legend=item['legend'],
                                    xLabel=item['xLabel'],
                                    yLabel=item['yLabel'],
                                    title=item['title'])
                self.main.figures_layout.addWidget(plot, row=item['row'] + 1, col=item['col'])
        self.newRun = True