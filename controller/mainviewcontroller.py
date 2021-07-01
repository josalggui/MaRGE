"""
Main View Controller

@author:    Yolanda Vives

@status:    Sets up the main view, its views and controllers
@todo:

"""
from PyQt5.QtWidgets import QListWidgetItem,  QMessageBox,  QFileDialog,  QTextEdit
from PyQt5.QtCore import QFile, QTextStream,  pyqtSignal, pyqtSlot
from PyQt5.uic import loadUiType, loadUi
from PyQt5 import QtGui
from controller.acquisitioncontroller import AcquisitionController
from controller.calibrationcontroller import CalibrationController
import pyqtgraph.exporters
import os
import ast
import sys
sys.path.append('../marcos_client')
import experiment as ex
from scipy.io import savemat
from controller.sequencecontroller import SequenceList
from seq.gradEcho import grad_echo 
from seq.radial import radial
from seq.turboSpinEcho import turbo_spin_echo
from seq.fid import fid
from seq.spinEcho import spin_echo
from seq.spinEcho1D import spin_echo1D
from seq.spinEcho2D import spin_echo2D
from seq.spinEcho3D import spin_echo3D
#from plotview.sequenceViewer import SequenceViewer
from sequencemodes import defaultsequences
from manager.datamanager import DataManager
from datetime import date,  datetime 
from globalvars import StyleSheets as style
from stream import EmittingStream
sys.path.append('../marcos_client')
from local_config import ip_address


import cgitb 
cgitb.enable(format = 'text')
import pdb
st = pdb.set_trace

MainWindow_Form, MainWindow_Base = loadUiType('ui/mainview.ui')


class MainViewController(MainWindow_Form, MainWindow_Base):
    """
    MainViewController Class
    """
    onSequenceChanged = pyqtSignal(str)
    
    def __init__(self):
        super(MainViewController, self).__init__()
        self.ui = loadUi('ui/mainview.ui')
        self.setupUi(self)
        self.styleSheet = style.breezeLight
        self.setupStylesheet(self.styleSheet)
  
        # Initialisation of sequence list
        self.sequencelist = SequenceList(self)
        self.sequencelist.setCurrentIndex(1)
#        self.sequencelist.itemClicked.connect(self.sequenceChangedSlot)
        self.sequencelist.currentIndexChanged.connect(self.selectionChanged)
        self.layout_operations.addWidget(self.sequencelist)
        
        # Console
        self.cons = self.generateConsole('')
        self.layout_output.addWidget(self.cons)
        sys.stdout = EmittingStream(textWritten=self.onUpdateText)
        sys.stderr = EmittingStream(textWritten=self.onUpdateText)        
        
        # Initialisation of acquisition controller
        acqCtrl = AcquisitionController(self, self.sequencelist)
        
        # Connection to the server
        self.ip = ip_address
        
        # Init gpa
        
        
        # Toolbar Actions
        self.action_gpaInit.triggered.connect(self.initgpa)
        self.action_calibration.triggered.connect(self.calibrate)
        self.action_changeappearance.triggered.connect(self.changeAppearanceSlot)
        self.action_acquire.triggered.connect(acqCtrl.startAcquisition)
        self.action_loadparams.triggered.connect(self.load_parameters)
        self.action_saveparams.triggered.connect(self.save_parameters)
        self.action_close.triggered.connect(self.close)    
        self.action_savedata.triggered.connect(self.save_data)
        self.action_exportfigure.triggered.connect(self.export_figure)
        self.action_viewsequence.triggered.connect(self.plot_sequence)
        
    def lines_that_start_with(self, str, f):
        return [line for line in f if line.startswith(str)]
    
    @staticmethod
    def generateConsole(text):
        con = QTextEdit()
        con.setText(text)
        return con
    
    def onUpdateText(self, text):
        cursor = self.cons.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.cons.setTextCursor(cursor)
        self.cons.ensureCursorVisible()
    
    def __del__(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)
    
    @pyqtSlot(bool)
    def changeAppearanceSlot(self) -> None:
        """
        Slot function to switch application appearance
        @return:
        """
        if self.styleSheet is style.breezeDark:
            self.setupStylesheet(style.breezeLight)
        else:
            self.setupStylesheet(style.breezeDark)
       
    def close(self):
        sys.exit()   

    def setupStylesheet(self, style) -> None:
        """
        Setup application stylesheet
        @param style:   Stylesheet to be set
        @return:        None
        """
        self.styleSheet = style
        file = QFile(style)
        file.open(QFile.ReadOnly | QFile.Text)
        stream = QTextStream(file)
        stylesheet = stream.readAll()
        self.setStyleSheet(stylesheet)  
        
#    @pyqtSlot(QListWidgetItem)
#    def sequenceChangedSlot(self, item: QListWidgetItem = None) -> None:
#        """
#        Operation changed slot function
#        @param item:    Selected Operation Item
#        @return:        None
#        """
#        self.sequence = item.text()
#        self.onSequenceChanged.emit(self.sequence)
#        self.action_acquire.setEnabled(True)
#        self.clearPlotviewLayout()
        
    def selectionChanged(self,item):
        self.sequence = self.sequencelist.currentText()
        self.onSequenceChanged.emit(self.sequence)
        self.action_acquire.setEnabled(True)
        self.clearPlotviewLayout()
    
    def clearPlotviewLayout(self) -> None:
        """
        Clear the plot layout
        @return:    None
        """
        for i in reversed(range(self.plotview_layout.count())):
            self.plotview_layout.itemAt(i).widget().setParent(None)
          
    
    def save_data(self):
        
        dataobject: DataManager = DataManager(self.rxd, self.lo_freq, len(self.rxd),  self.sequence.n, self.sequence.BW)
        dict = vars(defaultsequences[self.sequence])
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M")
        dt2 = date.today()
        dt2_string = dt2.strftime("%d-%m-%Y")
        dict["rawdata"] = self.rxd
        dict["fft"] = dataobject.f_fftData
        if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('experiments/acquisitions/%s' % (dt2_string))
            
        if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
            
        savemat("experiments/acquisitions/%s/%s/%s.mat" % (dt2_string, dt_string, self.sequence), dict)
        
        self.messages("Data saved")

    def export_figure(self):
        
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M")
        dt2 = date.today()
        dt2_string = dt2.strftime("%d-%m-%Y")

        if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('experiments/acquisitions/%s' % (dt2_string))    
        if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
                    
        exporter1 = pyqtgraph.exporters.ImageExporter(self.f_plotview.scene())
        exporter1.export("experiments/acquisitions/%s/%s/Freq%s.png" % (dt2_string, dt_string, self.sequence))
        exporter2 = pyqtgraph.exporters.ImageExporter(self.t_plotview.scene())
        exporter2.export("experiments/acquisitions/%s/%s/Temp%s.png" % (dt2_string, dt_string, self.sequence))
        
        self.messages("Figures saved")

    def load_parameters(self):
    
        self.clearPlotviewLayout()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Parameters File', "experiments/parameterisations/")
        
        if file_name:
            f = open(file_name,"r")
            contents=f.read()
            new_dict=ast. literal_eval(contents)
            f.close()

        lab = 'nmspc.%s' %(new_dict['seq'])
        item=eval(lab)

        del new_dict['seq']     
        for key in new_dict:       
            setattr(defaultsequences[self.sequence], key, new_dict[key])
        
        self.sequence = item
        self.onSequenceChanged.emit(self.sequence)

        self.messages("Parameters of %s sequence loaded" %(self.sequence))

    def save_parameters(self):
        
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M")
        dict = vars(defaultsequences[self.sequence]) 
    
        f = open("experiments/parameterisations/%s_params_%s.txt" % (self.sequence, dt_string),"w")
        f.write( str(dict) )
        f.close()
  
        self.messages("Parameters of %s sequence saved" %(self.sequence))
        
    def plot_sequence(self):
        
        plotSeq=1
        self.sequence = defaultsequences[self.sequencelist.getCurrentSequence()]
        
        if self.sequence.seq == 'FID':
            fid(self.sequence, plotSeq)
        if self.sequence.seq=='SE':
            spin_echo(self.sequence, plotSeq) 
        if self.sequence.seq=='SE1D':
            spin_echo1D(self.sequence, plotSeq)
        if self.sequence.seq=='SE2D':
            spin_echo2D(self.sequence, plotSeq)
        if self.sequence.seq=='SE3D':
            spin_echo3D(self.sequence, plotSeq)
        if self.sequence.seq == 'R':
            radial(self.sequence, plotSeq)    
        elif self.sequence.seq == 'GE':
            grad_echo(self.sequence, plotSeq)   
        elif self.sequence.seq == 'TSE':
            turbo_spin_echo(self.sequence, plotSeq)    
#        seqViewer = SequenceViewer(self, self.sequencelist)
#        seqViewer.plotSequence()
#        seqViewer.show()
  
        
    def messages(self, text):
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.exec();
        
    def calibrate(self):
        seqCalib = CalibrationController(self, self.sequencelist)
        seqCalib.show()
        
    def initgpa(self):
        expt = ex.Experiment(init_gpa=True)
        expt.run()
