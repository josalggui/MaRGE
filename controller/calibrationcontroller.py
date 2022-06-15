"""
Acquisition Manager
@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   2.0 (Beta)
@change:    19/06/2020

@summary:   Class for controlling the acquisition

@status:    Under development

"""
from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.uic import loadUiType,  loadUi
from PyQt5.QtCore import pyqtSignal,  pyqtSlot
from controller.calibfunctionscontroller import CalibFunctionsList
from controller.calibrationAcqcontroller import CalibrationAcqController
from PyQt5 import QtGui
from seq.sequencesCalibration import defaultCalibFunctions
from controller.sweepcontroller import SweepController

CalibrationController_Form, CalibrationController_Base = loadUiType('ui/calibrationViewer.ui')

class CalibrationController(CalibrationController_Base, CalibrationController_Form):

    onCalibFunctionChanged = pyqtSignal(str)

    def __init__(self, parent=None, calibfunctionslist=None):
        super(CalibrationController, self).__init__(parent)
        self.ui = loadUi('ui/calibrationViewer.ui')
        self.setupUi(self)
        
        # Initialisation of calibfunctions list
        self.calibfunctionslist = CalibFunctionsList(self)
        self.calibfunctionslist.itemClicked.connect(self.calibfunctionChangedSlot)
        self.layout_operations.addWidget(self.calibfunctionslist)
        
        
        calibAcqCtrl = CalibrationAcqController(self, self.calibfunctionslist)
        self.action_acquire.triggered.connect(calibAcqCtrl.startCalibAcq)
        self.action_sweep.triggered.connect(self.sweep_system)
        self.action_close.triggered.connect(self.close)
        self.action_viewsequence.triggered.connect(calibAcqCtrl.startSequencePlot)
       
    
    @pyqtSlot(QListWidgetItem)
    def calibfunctionChangedSlot(self, item: QListWidgetItem = None) -> None:
        """
        Operation changed slot function
        @param item:    Selected Operation Item
        @return:        None
        """
        self.calibfunction = item.text()
        self.onCalibFunctionChanged.emit(self.calibfunction)
        self.action_acquire.setEnabled(True)

        self.clearPlotviewLayout()
        
    def onUpdateText(self, text):
        cursor = self.cons.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.cons.setTextCursor(cursor)
        self.cons.ensureCursorVisible()

    def clearPlotviewLayout(self) -> None:
        """
        Clear the plot layout
        @return:    None
        """
        
        for i in reversed(range(self.plotview_layout.count())):
            if self.plotview_layout.itemAt(i).layout():
                self.plotview_layout.itemAt(i).layout().setParent(None)
            else:
                self.plotview_layout.itemAt(i).widget().setParent(None)

    def sweep_system(self):
        sweep = SweepController(self, self.calibfunctionslist, defaultCalibFunctions)
        sweep.show()
