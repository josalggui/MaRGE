"""
@author:    Yolanda Vives
@contact:   yolanda.vives@physiomri.com
@version:   2.0 (Beta)
@change:    19/10/2021

@summary:   Class controlling the batch 

@status:    Under development

"""
from PyQt5.QtWidgets import QFileDialog, QListWidget
from PyQt5.uic import loadUiType,  loadUi
from PyQt5.QtCore import pyqtSignal, QFileInfo
from controller.acquisitioncontroller import AcquisitionController
from sequencemodes import defaultsequences
#from PyQt5 import QtGui
import ast

BatchController_Form, BatchController_Base = loadUiType('ui/batchViewer.ui')

class BatchController(BatchController_Base, BatchController_Form):

    onCalibFunctionChanged = pyqtSignal(str)

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
        
        self.jobs = []
        
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
            
            # Reading file of the parameters
            f = open(pathJob,"r")
            contents=f.read()
            new_dict=ast. literal_eval(contents)
            f.close()
            
            # Charging the parameters
            lab = 'nmspc.%s' %(new_dict['seq'])
            item=eval(lab)

            del new_dict['seq']     
            for key in new_dict:       
                setattr(defaultsequences[self.sequence], key, new_dict[key])
        
            self.sequence = item
            self.onSequenceChanged.emit(self.sequence)

           # Initialisation of acquisition controller
            acqCtrl = AcquisitionController(self, self.sequencelist)
            acqCtrl.startAcquisition()
        
