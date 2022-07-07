"""
Calibration Controller

@author:    Yolanda Vives
@author:    J.M. Algarín, josalggui@i3m.upv.es
@version:   2.0 (Beta)
"""
from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.uic import loadUiType,  loadUi
from PyQt5.QtCore import pyqtSignal,  pyqtSlot
from controller.calibfunctionscontroller import CalibFunctionsList
from controller.calibrationAcqcontroller import CalibrationAcqController
from PyQt5 import QtGui
from seq.sequencesCalibration import defaultCalibFunctions
from plotview.spectrumplot import SpectrumPlotSeq

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
        self.action_acquire.triggered.connect(calibAcqCtrl.startAcquisition)
        self.action_close.triggered.connect(self.close)
        self.action_viewsequence.triggered.connect(self.startSequencePlot)

    def startSequencePlot(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: rare sequence class
        """
        # Delete previous plots
        self.clearPlotviewLayout()

        self.calibfunction = defaultCalibFunctions[self.calibfunctionslist.getCurrentCalibfunction()]
        self.funName = self.calibfunction.mapVals['seqName']

        # Create selected sequence
        print('Plot sequence')
        defaultCalibFunctions[self.funName].sequenceRun(1)  # Run sequence

        # Get sequence instructions plot
        out = defaultCalibFunctions[self.funName].sequencePlot()  # Plot results

        # Add plots to layout
        n = 0
        plot = []
        for item in out:
            plot.append(SpectrumPlotSeq(item[0], item[1], item[2], 'Time (ms)', 'Amplitude (a.u.)', item[3]))
            if n > 0: plot[n].plotitem.setXLink(plot[0].plotitem)
            n += 1
        for n in range(4):
            self.plotview_layout.addWidget(plot[n])
    
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

        # self.clearPlotviewLayout()
        
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
            # if self.plotview_layout.itemAt(i).layout():
            #     self.plotview_layout.itemAt(i).layout().setParent(None)
            # else:
            #     self.plotview_layout.itemAt(i).widget().setParent(None)
            self.plotview_layout.itemAt(i).widget().setParent(None)
