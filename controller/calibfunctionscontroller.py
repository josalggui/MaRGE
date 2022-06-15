"""
Calibration Function Controller

@author:    Yolanda Vives
@author:    J.M. Algarín, josalggui@i3m.upv.es
@version:   2.0 (Beta)
"""
from PyQt5.QtWidgets import QListWidget, QSizePolicy, QLabel
from PyQt5.QtCore import Qt, QRegExp
# from calibfunctionsmodes import defaultCalibFunctions
from seq.sequencesCalibration import defaultCalibFunctions
from calibfunctionsnamespace import Namespace as cfnmspc
from calibfunctionsnamespace import Tooltip_label as tlt_l
from calibfunctionsnamespace import Tooltip_inValue as tlt_inV
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
        self.addItems(list(defaultCalibFunctions.keys()))
        parent.onCalibFunctionChanged.connect(self.triggeredCalibfunctionChanged)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        # Make parent reachable from outside __init__
        self.parent = parent
#        self._currentSequence = None
        self._currentCalibfunction = "RabiFlops"
        self.setParametersUI("RabiFlops")
    
    def triggeredCalibfunctionChanged(self, calibfunction: str = None) -> None:
        # TODO: set calibfunction only once right here or on changed signal
        # packet = Com.constructSequencePacket(operation)
        # Com.sendPacket(packet)
        self._currentCalibfunction = calibfunction
        self.setParametersUI(calibfunction)

    def getCurrentCalibfunction(self) -> str:
        return self._currentCalibfunction

    def setParametersUI(self, calibfunction: str = None) -> None:
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
        self.calibfunction = calibfunction

        if hasattr(defaultCalibFunctions[self.calibfunction], 'OTHproperties'):
            sys_prop = defaultCalibFunctions[self.calibfunction].OTHproperties
            inputwidgets += [self.generateLabelItem(cfnmspc.systemproperties)]
            inputwidgets += self.generateWidgetsFromDict(sys_prop, calibfunction)
            
        for item in inputwidgets:
            self.parent.layout_parameters.addWidget(item)

       
    @staticmethod
    def generateWidgetsFromDict(obj: dict = None, calibfunction: str = None) -> list:
        widgetlist: list = []
        for key in obj:
            widget = CalibfunctionParameter(key, obj[key], calibfunction)
            widgetlist.append(widget)
        return widgetlist

    @staticmethod
    def generateLabelItem(text):
        label = QLabel()
        label.setText(text)
        label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        return label

    def get_items(self, struct: dict = None, calibfunction:str = None) -> list:
        itemlist: list = []
        for key in list(struct.keys()):
            if type(struct[key]) == dict:
                itemlist.append(self.generateLabelItem(key))
                itemlist += self.get_items(struct[key])
            else:
                item = CalibfunctionParameter(key, struct[key], calibfunction)
                itemlist.append(item)

        return itemlist


class CalibfunctionParameter(Parameter_Base, Parameter_Form):
    """
    Operation Parameter Widget-Class
    """
    # Get reference to position in operation object
    def __init__(self, name, parameter, calibfunction):
        super(CalibfunctionParameter, self).__init__()
        self.setupUi(self)

        # Set input parameter's label and value
        self.calibfunction = calibfunction
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
        temp = defaultCalibFunctions[self.calibfunction].mapVals

        # Get key for corresponding modified parameter
        names = defaultCalibFunctions[self.calibfunction].mapNmspc  # Map with GUI names
        modName = self.label_name.text()  # GUI name of the modified value
        key = [k for k, v in names.items() if v == modName][0]  # Get corresponding key of the modified value
        valOld = defaultCalibFunctions[self.calibfunction].mapVals[key]  # Current value to be saved again in case of error
        dataLen = defaultCalibFunctions[self.calibfunction].mapLen[key]
        if dataLen == 1: valOld = [valOld]
        dataType = type(valOld[0])

        # Modify the corresponding value into the sequence
        inputStr = self.input_value.text()  # Input value (gui allways gives strings)
        inputStr = inputStr.replace('[', '')
        inputStr = inputStr.replace(']', '')
        inputStr = inputStr.split(',')
        inputNum = []
        for ii in range(dataLen):
            if dataType == float:
                try:
                    inputNum.append(float(inputStr[ii]))
                except:
                    inputNum.append(float(valOld[ii]))
            elif dataType == int:
                try:
                    inputNum.append(int(inputStr[ii]))
                except:
                    inputNum.append(int(valOld[ii]))
            else:
                try:
                    inputNum.append(str(inputStr[ii]))
                except:
                    inputNum.append(str(valOld[ii]))
        if dataLen == 1:  # Save value into mapVals
            defaultCalibFunctions[self.calibfunction].mapVals[key] = inputNum[0]
        else:
            defaultCalibFunctions[self.calibfunction].mapVals[key] = inputNum


                
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
        
