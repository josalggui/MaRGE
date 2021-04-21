"""
Operations Controller

@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   2.0 (Beta)
@change:    13/06/2020

@summary:   TBD

@status:    Under development
@todo:      Extend construction of parameter section (headers, more categories, etc. )

"""

from PyQt5.QtWidgets import QListWidget, QSizePolicy, QLabel
from PyQt5.QtCore import Qt
from sequencemodes import defaultsequences
from sequencesnamespace import Namespace as nmspc
from PyQt5.uic import loadUiType

Parameter_Form, Parameter_Base = loadUiType('ui/inputparameter.ui')


class SequenceList(QListWidget):
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
        parent.onSequenceChanged.connect(self.triggeredSequenceChanged)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        # Make parent reachable from outside __init__
        self.parent = parent
        self._currentSequence = None

    def triggeredSequenceChanged(self, sequence: str = None) -> None:
        # TODO: set sequence only once right here or on changed signal
        # packet = Com.constructSequencePacket(operation)
        # Com.sendPacket(packet)
        self._currentSequence = sequence
        self.setParametersUI(sequence)

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
        inputwidgets: list = []

        if hasattr(defaultsequences[sequence], 'systemproperties'):
            sys_prop = defaultsequences[sequence].systemproperties
            inputwidgets += [self.generateLabelItem(nmspc.systemproperties)]
            inputwidgets += self.generateWidgetsFromDict(sys_prop, sequence)
            
        if hasattr(defaultsequences[sequence], 'sqncproperties'):
            seqs_prop = defaultsequences[sequence].sqncproperties
            inputwidgets += [self.generateLabelItem(nmspc.sqncproperties)]
            inputwidgets += self.generateWidgetsFromDict(seqs_prop, sequence)

        if hasattr(defaultsequences[sequence], 'gradientshims'):
            shims = defaultsequences[sequence].gradientshims
            inputwidgets += [(self.generateLabelItem(nmspc.shim))]
            inputwidgets += (self.generateWidgetsFromDict(shims))
            
#        print(self.get_items(sys_prop, sequence))

        for item in inputwidgets:
            self.parent.layout_parameters.addWidget(item)

    @staticmethod
    def generateWidgetsFromDict(obj: dict = None, sequence: str = None) -> list:
        widgetlist: list = []
        for key in obj:
            print(key)
            print(obj[key])
            widget = SequenceParameter(key, obj[key], sequence)
            widgetlist.append(widget)
        return widgetlist

    @staticmethod
    def generateLabelItem(text):
        label = QLabel()
        label.setText(text)
        label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        return label

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
        print("{}: {}".format(self.label_name.text(), self.input_value.text()))
        # TODO: Setup validator to numbers only (float)
        
        # Connect text changed signal to getValue function
        self.input_value.textChanged.connect(self.get_value)
        
    def get_value(self) -> None:
        print("{}: {}".format(self.label_name.text(), self.input_value.text()))

        temp = vars(defaultsequences[self.sequence])
        for item in temp:
            lab = 'nmspc.%s' %(item)
            res=eval(lab)
            if (res == self.label_name.text()):
                t = type(getattr(defaultsequences[self.sequence], item))        
                if (t is float): 
                    value: float = float(self.input_value.text())
                elif (t is int): 
                    value: int = int(self.input_value.text())
        
                setattr(defaultsequences[self.sequence], item, value)
        
    def set_value(self, value: str) -> None:
        print("{}: {}".format(self.label_name.text(), self.input_value.text()))
        self.input_value.setText(value)
