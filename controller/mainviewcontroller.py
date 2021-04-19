"""
Main View Controller

@author:    Yolanda Vives

@status:    Sets up the main view, its views and controllers
@todo:

"""
from PyQt5.QtWidgets import QListWidgetItem
from PyQt5.QtCore import QFile, QTextStream
from PyQt5.uic import loadUiType, loadUi
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5 import QtGui
from controller.acquisitioncontroller import AcquisitionController
from scipy.io import savemat
from controller.sequencecontroller import SequenceList
from controller.connectiondialog import ConnectionDialog
from controller.outputparametercontroller import Output
from sequencemodes import defaultsequences
from manager.datamanager import DataManager
from datetime import date,  datetime 
from globalvars import StyleSheets as style

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
        sequencelist = SequenceList(self)
        sequencelist.itemClicked.connect(self.sequenceChangedSlot)
        self.layout_operations.addWidget(sequencelist)

        # Initialisation of output section
        outputsection = Output(self)
        
        # Initialisation of acquisition controller
        acqCtrl = AcquisitionController(self, outputsection, sequencelist)
        
        # Toolbar Actions
        self.action_connect.triggered.connect(self.marcos_server)
        self.action_changeappearance.triggered.connect(self.changeAppearanceSlot)
        self.action_acquire.triggered.connect(acqCtrl.startAcquisition)
        self.action_close.triggered.connect(self.close)    
        self.action_savedata.triggered.connect(self.save_data)
        
    def marcos_server(self):
        self.con = ConnectionDialog(self)
        self.con.show()
    
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
        quit()    

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
        

        
    @pyqtSlot(QListWidgetItem)
    def sequenceChangedSlot(self, item: QListWidgetItem = None) -> None:
        """
        Operation changed slot function
        @param item:    Selected Operation Item
        @return:        None
        """
        self.sequence = item.text()
        self.onSequenceChanged.emit(self.sequence)
        self.action_acquire.setEnabled(True)
    
    def clearPlotviewLayout(self) -> None:
        """
        Clear the plot layout
        @return:    None
        """
        for i in reversed(range(self.plotview_layout.count())):
            self.plotview_layout.itemAt(i).widget().setParent(None)
          
    
    def save_data(self):
        
#        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File', 'default.mat')
#        file = open(name[0],'w')
#        text = self.textEdit.toPlainText()
#        file.write(text)
#        file.close()
        
        dataobject: DataManager = DataManager(self.rxd, self.lo_freq, len(self.rxd))
        dict = vars(defaultsequences[self.sequence])
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M")
#        t =datetime.now()
#        print(dt)
#        dict["date"] = dt.strftime
#        dict["time"] = t.strftime
        dict["rawdata"] = self.rxd
        dict["fft"] = dataobject.f_fftData
        
        savemat("experiments/%s.mat" % dt_string, dict)
        


    

