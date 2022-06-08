"""

@summary:   Class for controlling the acquisition of the calibration

@status:    Under development

"""

from PyQt5.QtWidgets import QTextEdit, QCheckBox, QHBoxLayout
from plotview.spectrumplot import SpectrumPlot
from seq.sequencesCalibration import defaultCalibFunctions
from PyQt5.QtCore import QObject,  pyqtSlot
from manager.datamanager import DataManager
from seq.shimming import shimming
from seq.rabiFlops import rabi_flops
from seq.larmor import larmorFreq
from seq.flipAngle import flipAngle
# from seq.inversionRecovery import inversionRecovery
#from scipy.io import savemat
import numpy as np

class CalibrationAcqController(QObject):
    def __init__(self, parent=None, calibfunctionslist=None):
        super(CalibrationAcqController, self).__init__(parent)

        self.parent = parent
        self.calibfunctionslist = calibfunctionslist
        self.acquisitionData = None
        
        self.layout = QHBoxLayout()
        self.b1 = QCheckBox("Plot Shim x")
        self.b1.setGeometry(200, 150, 100, 30)   #            setting geometry of check box
        self.b1.setStyleSheet("QCheckBox::indicator"
                               "{"
                               "background-color : white;"
                               "selection-color: black;"
                               "}")     #    adding background color to indicator
        self.b1.toggled.connect(lambda:self.btnstate(self.b1))
        self.layout.addWidget(self.b1)
    
        self.b2 = QCheckBox("Plot Shim y")
        self.b2.setGeometry(200, 150, 100, 30)   #            setting geometry of check box
        self.b2.setStyleSheet("QCheckBox::indicator"
                               "{"
                               "background-color : white;"
                               "selection-color: black;"
                               "}")     #    adding background color to indicator
        self.b2.toggled.connect(lambda:self.btnstate(self.b2))
        self.layout.addWidget(self.b2)
        
        self.b3 = QCheckBox("Plot Shim z")
        self.b3.setGeometry(200, 150, 100, 30)   #            setting geometry of check box
        self.b3.setStyleSheet("QCheckBox::indicator"
                               "{"
                               "background-color : white;"
                               "selection-color: black;"
                               "}")     #    adding background color to indicator        
        self.b3.toggled.connect(lambda:self.btnstate(self.b3))
        self.layout.addWidget(self.b3)

    @pyqtSlot(bool)
    def startCalibAcq(self):
    
        self.layout.setParent(None)
        self.parent.clearPlotviewLayout()

        self.calibfunction = defaultCalibFunctions[self.calibfunctionslist.getCurrentCalibfunction()]
        self.funName = self.calibfunction.mapVals['seqName']

        # Execute selected sequence
        print('Start sequence')
        defaultCalibFunctions[self.funName].sequenceRun(0)  # Run sequence
        defaultCalibFunctions[self.funName].sequenceAnalysisGUI(self)   # Plot results
        print('End sequence')

    def plot_shim(self, axis):
        
        shim_values = np.linspace(self.calibfunction.shim_initial, self.calibfunction.shim_final, self.calibfunction.N)
        
        if axis == 'x':
            plotview1 = SpectrumPlot(shim_values, self.peakValsf_x, [],[],"Shim value x", "Peak value", "%s x Peak value" %(self.calibfunction.cfn))
            plotview2 = SpectrumPlot(shim_values, self.fwhmf_x, [],[],"Shim value x ", "FHWM", "%s x FHWM" %(self.calibfunction.cfn))

        elif axis == 'y':
            plotview1 = SpectrumPlot(shim_values,  self.peakValsf_y, [],[],"Shim value", "Peak value", "%s y Peak value" %(self.calibfunction.cfn))
            plotview2 = SpectrumPlot(shim_values, self.fwhmf_y, [],[],"Shim value", "FHWM", "%s y FWHM" %(self.calibfunction.cfn))

        elif axis == 'z':
            plotview1 = SpectrumPlot(shim_values, self.peakValsf_z, [],[],"Shim value", "Peak value", "%s z Peak value" %(self.calibfunction.cfn))
            plotview2 = SpectrumPlot(shim_values, self.fwhmf_z, [],[],"Shim value", "FHWM", "%s z FWHM" %(self.calibfunction.cfn))

        self.layout.setParent(None)
        self.parent.clearPlotviewLayout()  
        self.parent.plotview_layout.addLayout(self.layout)
        self.parent.plotview_layout.addWidget(plotview1)
        self.parent.plotview_layout.addWidget(plotview2)
            
        max_x=self.peakValsf_x.index(round(np.max(self.peakValsf_x), 4))
        max_y=self.peakValsf_y.index(round(np.max(self.peakValsf_y), 4))
        max_z=self.peakValsf_z.index(round(np.max(self.peakValsf_z), 4))
        
        shim_x=shim_values[max_x]
        shim_y=shim_values[max_y]
        shim_z=shim_values[max_z]
        
        self.textEdit = QTextEdit()
        self.textEdit.setPlainText('Shim_x=%0.5f,       Shim_y=%0.5f,       Shim_z=%0.5f'%(shim_x, shim_y, shim_z))
        self.parent.plotview_layout.addWidget(self.textEdit)
        

    def btnstate(self,b):

        if b.text() == 'Plot Shim x':
            if b.isChecked() == True:
                self.plot_shim(axis='x')
                self.b2.setChecked(False)
                self.b3.setChecked(False)  
            
        if b.text() == 'Plot Shim y':
            if b.isChecked() == True:
                self.plot_shim(axis='y')
                self.b1.setChecked(False)
                self.b3.setChecked(False)  

        if b.text() == 'Plot Shim z':
            if b.isChecked() == True:
                self.plot_shim(axis='z')
                self.b1.setChecked(False)
                self.b2.setChecked(False)  

    


