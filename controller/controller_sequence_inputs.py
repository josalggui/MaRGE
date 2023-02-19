"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
from widgets.widget_tab_sequence_inputs import SequenceInputsWidget
from seq.sequences import defaultsequences
import numpy as np


class SequenceInputsController(SequenceInputsWidget):
    def __init__(self, *args, **kwargs):
        super(SequenceInputsController, self).__init__(*args, **kwargs)
        self.seq_name = self.main.sequence_list.seq_name
        self.displayInputParameters()

    def displayInputParameters(self):
        self.seq_name = self.main.sequence_list.seq_name
        if hasattr(defaultsequences[self.seq_name], 'IMproperties'):
            im_prop = defaultsequences[self.seq_name].IMproperties
            input_widgets_1 = self.createTab(im_prop)
            self.addTab(input_widgets_1, "Image")

        if hasattr(defaultsequences[self.seq_name], 'RFproperties'):
            im_prop = defaultsequences[self.seq_name].RFproperties
            input_widgets_2 = self.createTab(im_prop)
            self.addTab(input_widgets_2, "RF")

        if hasattr(defaultsequences[self.seq_name], 'SEQproperties'):
            im_prop = defaultsequences[self.seq_name].SEQproperties
            input_widgets_3 = self.createTab(im_prop)
            self.addTab(input_widgets_3, "Sequence")

        if hasattr(defaultsequences[self.seq_name], 'OTHproperties'):
            im_prop = defaultsequences[self.seq_name].OTHproperties
            input_widgets_4 = self.createTab(im_prop)
            self.addTab(input_widgets_4, "Others")

    def removeTabs(self):
        self.removeTab(0)
        self.removeTab(0)
        self.removeTab(0)
        self.removeTab(0)

    def createTab(self, inputs):
        # Create widget
        widget = QWidget()

        # Inputs are distributed in vertical layout
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Add inputs. Label and Values are distributed in horizontal layout
        for key in inputs.keys():
            input_layout = SequenceParameter([key, str(inputs[key][0])], self.seq_name)
            layout.addLayout(input_layout)

        # Add spacer to place all the items on the top
        layout.addStretch()

        return widget


class SequenceParameter(QHBoxLayout):
    def __init__(self, values, seq_name, *args, **kwargs):
        super(SequenceParameter, self).__init__(*args, **kwargs)
        self.seq_name = seq_name
        self.param_text = values[0]
        self.param_value = values[1]

        # QLabel
        self.input_label = QLabel(values[0])
        self.input_label.setMinimumWidth(150)
        self.addWidget(self.input_label)

        # QLineEdit
        self.input_value = QLineEdit(values[1])
        self.addWidget(self.input_value)
        self.input_value.textChanged.connect(self.getValue)

    def getValue(self):
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Here is where input values obtained from the gui are input into the sequence property mapVals
        """
        sequence = defaultsequences[self.seq_name]

        # Get key for corresponding modified parameter
        names = sequence.mapNmspc  # Map with GUI names
        mod_name = self.input_label.text()  # GUI name of the modified value
        key = [k for k, v in names.items() if v == mod_name][0]  # Get corresponding key of the modified value
        valOld = sequence.mapVals[key]  # Current value to be saved again in case of error
        dataLen = sequence.mapLen[key]
        valNew = self.input_value.text()
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
            sequence.mapVals[key] = inputNum[0]
        else:
            if dataLen == 1:  # Save value into mapVals
                sequence.mapVals[key] = inputNum[0]
            else:
                sequence.mapVals[key] = inputNum

        # Print value into the console
        seqTime = sequence.sequenceTime()
        if hasattr(self, 'oldSeqTime'):
            if seqTime != self.oldSeqTime:
                print('Sequence time %0.1f minutes' % seqTime)
                self.oldSeqTime = seqTime
        else:
            print('Sequence time %0.1f minutes' % seqTime)
            self.oldSeqTime = seqTime

        defaultsequences[self.seq_name] = sequence
