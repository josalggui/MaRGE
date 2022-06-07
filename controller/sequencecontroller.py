"""
Operations Controller

@author:    Yolanda Vives
@author:    J.M. Algarín, josalggui@i3m.upv.es
@version:   2.0 (Beta)
@change:    25/05/2022
@change     06/06/2022 (J.M. Algarín): adapted to the new structure of the sequences

@summary:   TBD

@status:    Under development
@todo:      Extend construction of parameter section (headers, more categories, etc. )

"""
from PyQt5.QtWidgets import QSizePolicy, QLabel,  QComboBox, QTabWidget, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QRegExp
# from sequencemodes import defaultsequences
from sequencesnamespace import Namespace as nmspc
from sequencesnamespace import Tooltip_label as tlt_l
from sequencesnamespace import Tooltip_inValue as tlt_inV
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QRegExpValidator

# Test from J.M. Algarín
from seq.sequences import defaultsequences

Parameter_Form, Parameter_Base = loadUiType('ui/inputparameter.ui')
#Parameter_FormG, Parameter_BaseG = loadUIType('ui/gradients.ui')


#class SequenceList(QListWidget):
class SequenceList(QComboBox):
    """
    Sequence List Class
    """
    def __init__(self, parent=None):
        """
        Initialization
        @param parent:  Mainviewcontroller (access to parameter layout)
        """
        super(SequenceList, self).__init__(parent)

        # Add sequences to sequences list
        self.addItems(list(defaultsequences.keys()))
        
        if hasattr(parent, 'onSequenceChanged'):
            parent.onSequenceChanged.connect(self.triggeredSequenceChanged)
            # Make parent reachable from outside __init__
            self.parent = parent
            self._currentSequence = "RARE"
            self.setParametersUI(self._currentSequence)
        else:
            self._currentSequence=parent.sequence
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    
    def triggeredSequenceChanged(self, sequence: str = None) -> None:
        # TODO: set sequence only once right here or on changed signal
        # packet = Com.constructSequencePacket(operation)
        # Com.sendPacket(packet)
        self._currentSequence = sequence
        self.setParametersUI(sequence)
        defaultsequences[sequence].sequenceInfo()

    def getCurrentSequence(self) -> str:
        return self._currentSequence

    def setParametersUI(self, sequence: str = None) -> None:
        """
        Set input parameters from sequence object
        @param sequence:  Sequence object
        @return:    None
        """
        # Reset row layout for input parameters
        for i in reversed(range(self.parent.layout_parameters.count())):
            self.parent.layout_parameters.itemAt(i).widget().setParent(None)

        # Add input parameters to row layout
        inputwidgets1: list = []
        inputwidgets2: list = []
        inputwidgets3: list = []
        inputwidgets4: list = []

        self.tabwidget = QTabWidget()

        if hasattr(defaultsequences[sequence], 'RFproperties'):
            rf_prop = defaultsequences[sequence].RFproperties
            inputwidgets1 += self.generateWidgetsFromDict(rf_prop, sequence)
            self.tab = self.generateTab(inputwidgets1)
            self.tabwidget.addTab(self.tab,"RF")

        if hasattr(defaultsequences[sequence], 'IMproperties'):
            im_prop = defaultsequences[sequence].IMproperties
            inputwidgets2 += self.generateWidgetsFromDict(im_prop, sequence)
            self.tab = self.generateTab(inputwidgets2)
            self.tabwidget.addTab(self.tab,"Image")

        if hasattr(defaultsequences[sequence], 'SEQproperties'):
            seq_prop = defaultsequences[sequence].SEQproperties
            inputwidgets3 += self.generateWidgetsFromDict(seq_prop, sequence)
            self.tab = self.generateTab(inputwidgets3)
            self.tabwidget.addTab(self.tab,"Sequence")

        if hasattr(defaultsequences[sequence], 'OTHproperties'):
            oth_prop = defaultsequences[sequence].OTHproperties
            inputwidgets4 += self.generateWidgetsFromDict(oth_prop, sequence)
            self.tab = self.generateTab(inputwidgets4)
            self.tabwidget.addTab(self.tab,"Others")
     
        self.parent.layout_parameters.addWidget(self.tabwidget)

    @staticmethod
    def generateWidgetsFromDict(obj: dict = None, sequence: str = None) -> list:
        widgetlist: list = []
        for key in obj:
            widget = SequenceParameter(key, obj[key], sequence)
            widgetlist.append(widget)
        return widgetlist

    @staticmethod
    def generateLabelItem(text):
        label = QLabel()
        label.setText(text)
        label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        return label
        
    @staticmethod
    def generateTab(inputwidgets):
        tab = QWidget()
        tab.layout = QVBoxLayout()
        for item in inputwidgets:
            tab.layout.addWidget(item)

        tab.layout.addStretch()
        tab.setLayout(tab.layout)
        return tab


#    def generateTickItem(text):
#        Qx = QCheckBox("Qx")
#        Qx.setChecked(True)
        
#        Qy = QRadioButton('Qy')
#        Qy.setChecked(True)
#        Qz = QRadioButton('Qz')
#        return Qx

    def get_items(self, struct: dict = None, sequence:str = None) -> list:
        itemlist: list = []
        for key in list(struct.keys()):
            if type(struct[key]) == dict:
                itemlist.append(self.generateLabelItem(key))
                itemlist += self.get_items(struct[key])
            else:
                item = SequenceParameter(key, struct[key], sequence)
                itemlist.append(item)

        return itemlist


class SequenceParameter(Parameter_Base, Parameter_Form):
    """
    Operation Parameter Widget-Class
    """
    # Get reference to position in operation object
    def __init__(self, name, parameter, sequence):
        super(SequenceParameter, self).__init__()
        self.setupUi(self)

        # Set input parameter's label and value
        self.sequence = sequence
        self.parameter = parameter
        self.label_name.setText(name)
        self.input_value.setText(str(parameter[0]))

        # Connect text changed signal to getValue function
        self.input_value.textChanged.connect(self.get_value)
        
    def get_value(self) -> None:
        """"
        @author: J.M. Algarin, MRILab, i3M, CSIC, Valencia, Spain
        @email: josalggui@i3m.upv.es
        Here is where input values obtained from the gui are input into the sequence property mapVals
        """
        temp = defaultsequences[self.sequence].mapVals

        # Get key for corresponding modified parameter
        names = defaultsequences[self.sequence].mapNmspc            # Map with GUI names
        modName = self.label_name.text()                            # GUI name of the modified value
        key = [k for k, v in names.items() if v == modName][0]      # Get corresponding key of the modified value
        valOld = defaultsequences[self.sequence].mapVals[key]       # Current value to be saved again in case of error
        dataLen = defaultsequences[self.sequence].mapLen[key]
        if dataLen == 1: valOld = [valOld]
        dataType = type(valOld[0])

        # Modify the corresponding value into the sequence
        inputStr = self.input_value.text()                          # Input value (gui allways gives strings)
        inputStr = inputStr.replace('[','')
        inputStr = inputStr.replace(']','')
        inputStr = inputStr.split(',')
        inputNum = []
        for ii in range(dataLen):
            if dataType==float:
                try: inputNum.append(float(inputStr[ii]))
                except: inputNum.append(float(valOld[ii]))
            elif dataType==int:
                try: inputNum.append(int(inputStr[ii]))
                except: inputNum.append(int(valOld[ii]))
            else:
                try: inputNum.append(str(inputStr[ii]))
                except: inputNum.append(str(valOld[ii]))
        if dataLen==1:                                                # Save value into mapVals
            defaultsequences[self.sequence].mapVals[key] = inputNum[0]
        else:
            defaultsequences[self.sequence].mapVals[key] = inputNum

        # Print value into the console (I have to ask how to show into the GUI)
        seqTime = defaultsequences[self.sequence].sequenceTime()
        print('Sequence time %1.1d minutes' % seqTime)
                
    def validate_input(self):
        reg_ex = QRegExp('^(?:0*(?:\.\d+)?|1(\.0*)?)$')
        input_validator = QRegExpValidator(reg_ex, self.input_value)
        self.input_value.setValidator(input_validator)
        state = input_validator.validate(self.input_value.text(), 0)
        if state[0] == QRegExpValidator.Acceptable:
            return 1
        else:
            return 0
        
    def set_value(self, key, value: str) -> None:
        print("{}: {}".format(self.label_name.text(), self.input_value.text()))
        
