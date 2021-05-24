"""
Acquisition Manager
@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   2.0 (Beta)
@change:    19/06/2020

@summary:   Class for controlling the acquisition

@status:    Under development

"""


from PyQt5.uic import loadUiType


CalibrationController_Form, CalibrationController_Base = loadUiType('ui/calibrationViewer.ui')

class CalibrationController(CalibrationController_Base, CalibrationController_Form):

    def __init__(self, parent=None, calibfunctionslist=None):
        super(CalibrationController, self).__init__(parent)

        self.setupUi(self)
        self.parent = parent
        self.calibfunctionslist = calibfunctionslist
        
        self.action_close.triggered.connect(self.close) 

    

        
    


