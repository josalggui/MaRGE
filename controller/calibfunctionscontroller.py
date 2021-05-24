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
from PyQt5.QtCore import Qt, QRegExp
#from sequencemodes import defaultsequences
from calibfunctionmodes import defaultcalibfunctions
from calibfunctionsnamespace import Namespace as cfnmspc
from sequencesnamespace import Tooltip_label as tlt_l
from sequencesnamespace import Tooltip_inValue as tlt_inV
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QRegExpValidator

Parameter_Form, Parameter_Base = loadUiType('ui/inputparameter.ui')


class CalibFunctionsList(QListWidget):
    """
    Sequence List Class
    """
    def __init__(self, parent=None):
        """
        Initialization
        @param parent:  Mainviewcontroller (access to parameter layout)
        """
        super(CalibFunctionsList, self).__init__(parent)

        # Add calibfunctions to calibfunctionslist
        self.addItems(list(defaultcalibfunctions.keys()))
        parent.onSequenceChanged.connect(self.triggeredCalibfunctionChanged)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        # Make parent reachable from outside __init__
        self.parent = parent
#        self._currentSequence = None
        self._currentCalibfunction = "Larmor Frequency"
        self.setParametersUI("Larmor Frequency")
    
    def triggeredCalibfunctionChanged(self, calibfunction: str = None) -> None:
        # TODO: set calibfunction only once right here or on changed signal
        # packet = Com.constructSequencePacket(operation)
        # Com.sendPacket(packet)
        self._currentCalibfunction = calibfunction
        self.setParametersUI(calibfunction)

    def getCurrentCalibfunction(self) -> str:
        return self._currentCalibfunction

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

        if hasattr(defaultcalibfunctions[self.calibfunction], 'systemproperties'):
            sys_prop = defaultcalibfunctions[self.calibfunction].systemproperties
            inputwidgets += [self.generateLabelItem(nmspc.systemproperties)]
            inputwidgets += self.generateWidgetsFromDict(sys_prop, sequence)
            
        if hasattr(defaultcalibfunctions[self.calibfunction], 'sqncproperties'):
            seqs_prop = defaultcalibfunctions[self.calibfunction].sqncproperties
            inputwidgets += [self.generateLabelItem(nmspc.sqncproperties)]
            inputwidgets += self.generateWidgetsFromDict(seqs_prop, sequence)

        if hasattr(defaultcalibfunctions[self.calibfunction], 'gradientshims'):
            shims = defaultcalibfunctions[self.calibfunction].gradientshims
            inputwidgets += [(self.generateLabelItem(nmspc.gradientshims))]
            inputwidgets += (self.generateWidgetsFromDict(shims, sequence))
            
        for item in inputwidgets:
            self.parent.layout_parameters.addWidget(item)
        
       
    @staticmethod
    def generateWidgetsFromDict(obj: dict = None, sequence: str = None) -> list:
        widgetlist: list = []
        for key in obj:
            widget = CalibfunctionParameter(key, obj[key], sequence)
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
                item = CalibfunctionParameter(key, struct[key], sequence)
                itemlist.append(item)

        return itemlist


class CalibfunctionParameter(Parameter_Base, Parameter_Form):
    """
    Operation Parameter Widget-Class
    """
    # Get reference to position in operation object
    def __init__(self, name, parameter, sequence):
        super(CalibfunctionParameter, self).__init__()
        self.setupUi(self)

        # Set input parameter's label and value
        self.calibfunction = calibfunction
        self.parameter = parameter
        self.label_name.setText(name)
        
        # Tooltips
        temp = vars(defaultcalibfunctions[self.calibfunction])
        for item in temp:
            label = 'nmspc.%s' %item
            res=eval(label)
            if (res == name):
                lab = 'tlt_l.%s' %(item)
                if (hasattr(tlt_l, item)):
                    res2=eval(lab)
                    self.label_name.setToolTip(res2)
                inV = 'tlt_inV.%s' %(item)
                if (hasattr(tlt_inV, item)):
                    res3 = eval(inV)
                    self.input_value.setToolTip(res3)    
                    
        self.input_value.setText(str(parameter[0]))
        
        
        # Connect text changed signal to getValue function
        self.input_value.textChanged.connect(self.get_value)
        
    def get_value(self) -> None:

        temp = vars(defaultcalibfunctions[self.calibfunction])
        for item in temp:
            lab = 'nmspc.%s' %(item)
            res=eval(lab)
            if (res == self.label_name.text()):
                t = type(getattr(defaultcalibfunctions[self.calibfunction], item))     
                inV = 'tlt_inV.%s' %(item)
                if (hasattr(tlt_inV, item)):
                    res3 = eval(inV)
                    if res3 == 'Value between 0 and 1':  
                        val=self.validate_input()
                        if val == 1:           
                            if (t is float): 
                                value: float = float(self.input_value.text())
                                setattr(defaultcalibfunctions[self.calibfunction], item, value)
                            elif (t is int): 
                                value: int = int(self.input_value.text())
                                setattr(defaultcalibfunctions[self.calibfunction], item, value)  
                else:
                    if (t is float): 
                        value: float = float(self.input_value.text())
                        setattr(defaultcalibfunctions[self.calibfunction], item, value)
                    elif (t is int): 
                        value: int = int(self.input_value.text())
                        setattr(defaultcalibfunctions[self.calibfunction], item, value)

                
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
        
