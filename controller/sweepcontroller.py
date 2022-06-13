"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: All sequences on the GUI must be here
"""
from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.uic import loadUiType,  loadUi
from PyQt5.QtCore import pyqtSignal,  pyqtSlot
from controller.sweepfunctionscontroller import FunctionsList
from controller.sweepacqcontroller import AcqController
from PyQt5 import QtGui

SweepController_Form, SweepController_Base = loadUiType('ui/sweepViewer.ui')

class SweepController(SweepController_Base, SweepController_Form):

    onFunctionChanged = pyqtSignal(str)

    def __init__(self, parent=None, sequencelist=None, defaultsequences=None):
        super(SweepController, self).__init__(parent)
        self.ui = loadUi('ui/sweepViewer.ui')
        self.setupUi(self)
        
        # Initialisation of sweepfunctions list
        self.defaultsequences = defaultsequences
        self.functionslist = FunctionsList(self)
        self.functionslist.itemClicked.connect(self.functionChangedSlot)
        self.layout_operations.addWidget(self.functionslist)
        
        
        acqCtrl = AcqController(self, self.functionslist)
        self.action_acquire.triggered.connect(acqCtrl.startAcquisition)
        self.action_close.triggered.connect(self.close) 
       
    
    @pyqtSlot(QListWidgetItem)
    def functionChangedSlot(self, item: QListWidgetItem = None) -> None:
        """
        Operation changed slot function
        @param item:    Selected Operation Item
        @return:        None
        """
        self.calibfunction = item.text()
        self.onFunctionChanged.emit(self.calibfunction)
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
