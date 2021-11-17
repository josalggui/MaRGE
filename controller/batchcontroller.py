"""
@author:    Yolanda Vives
@contact:   yolanda.vives@physiomri.com
@version:   2.0 (Beta)
@change:    19/10/2021

@summary:   Class controlling the batch 

@status:    Under development

"""
from PyQt5.QtWidgets import QFileDialog, QListWidget,  QMessageBox, QTextEdit
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon
from PyQt5.uic import loadUiType,  loadUi
from PyQt5.QtCore import QFileInfo
from controller.acquisitioncontroller import AcquisitionController
from sequencemodes import defaultsequences
from controller.sequencecontroller import SequenceList
from sequencesnamespace import Namespace as nmspc
#from PyQt5 import QtGui
import ast
import sys
from stream import EmittingStream

BatchController_Form, BatchController_Base = loadUiType('ui/batchViewer.ui')

class BatchController(BatchController_Base, BatchController_Form):

#    onBatchFunctionChanged = pyqtSignal(str)
#    onSequenceChanged = pyqtSignal(str)

    def __init__(self, parent=None, batchfunctionslist=None):
        super(BatchController, self).__init__(parent)
        self.ui = loadUi('ui/batchViewer.ui')
        self.setupUi(self)
        
        # Initialisation of batchfunctions list
        self.batchfunctionslist = QListWidget()
        self.layout_operations.addWidget(self.batchfunctionslist)
        
        # Toolbar Actions
        self.action_acquire.triggered.connect(self.startAcquisition)
        self.action_addProcess.triggered.connect(self.load_process)
        self.action_removeProcess.triggered.connect(self.remove_process)
        self.action_close.triggered.connect(self.close)  
        self.action_XNATupload.triggered.connect(self.xnat)  
        
        # XNAT upload
        self.xnat_active = 'FALSE'
        self.batch = 1
        self.jobs = []
        
        #Console
        self.cons = self.generateConsole('')
        self.console_batch.addWidget(self.cons)
        sys.stdout = EmittingStream(textWritten=self.onUpdateText)
        sys.stderr = EmittingStream(textWritten=self.onUpdateText)        
        
    def load_process(self):
    
        filenamePath, _ = QFileDialog.getOpenFileName(self, 'Open Parameters File', "experiments/parameterisations/")
        filename = QFileInfo(filenamePath).fileName()
        self.batchfunctionslist.addItem(filename)
        self.jobs.append(filenamePath)

    def remove_process(self):   
    
        idx = self.batchfunctionslist.currentRow()  
        del self.jobs[idx]        
        listItems=self.batchfunctionslist.selectedItems()
        for item in listItems:
            self.batchfunctionslist.takeItem(self.batchfunctionslist.row(item))

    def startAcquisition(self):
        
        num_jobs = len(self.jobs)

        for j in range(num_jobs):
            
            pathJob=self.jobs[j]
            new_dict = {}
            # Reading file of the parameters
            f = open(pathJob,"r")
            contents=f.read()
            new_dict=ast. literal_eval(contents)
            f.close()
            
            # Charging the parameters
            lab = 'nmspc.%s' %(new_dict['seq'])
            item=eval(lab)
            self.sequence=item
            del new_dict['seq']     
            for key in new_dict:       
                setattr(defaultsequences[self.sequence], key, new_dict[key])

            self.sequencelist = SequenceList(self)
            

           # Initialisation of acquisition controller
            acqCtrl = AcquisitionController(self, self.sequencelist)
            acqCtrl.startAcquisition()
            print('Job %s finished' %pathJob)
        
        self.messages("Done")
        
    def messages(self, text):
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.setWindowTitle('Job finished')
        msg.exec();

    def xnat(self):
        
        if self.xnat_active == 'TRUE':
            self.xnat_active = 'FALSE'
            self.action_XNATupload.setIcon(QIcon('/home/physioMRI/git_repos/PhysioMRI_GUI/resources/icons/upload-outline.svg') )
            self.action_XNATupload.setToolTip('Activate XNAT upload')
        else:
            self.xnat_active = 'TRUE'
            self.action_XNATupload.setIcon(QIcon('/home/physioMRI/git_repos/PhysioMRI_GUI/resources/icons/upload.svg') )
            self.action_XNATupload.setToolTip('Deactivate XNAT upload')
        
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
