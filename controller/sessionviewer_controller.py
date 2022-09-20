from PyQt5.uic import loadUiType, loadUi
import sys
from controller.mainviewcontroller import MainViewController
from controller.sessioncontroller import SessionList
from sessionmodes import defaultsessions
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import os

MainWindow_Form, MainWindow_Base = loadUiType('ui/sessionViewer.ui')


class SessionViewerController(MainWindow_Form, MainWindow_Base):
    
    onSessionChanged = pyqtSignal(str)
    
    def __init__(self, session, parent=None, pyfirmata=None):
        super(SessionViewerController, self).__init__(parent)
        self.ui = loadUi('ui/sessionViewer.ui')
        self.setupUi(self)
        
        self.session = session
        self.pyfirmata = pyfirmata
        self.lab = QLabel()
        self.lab.setText("New Session")
        self.lab.setAlignment(Qt.AlignCenter)
        self.layout_operations.addWidget(self.lab)
        
        self.sessionlist = SessionList(self)
        # Initialisation of session list
        if self.session != '':
            self.session = self.sessionlist.currentText() 
            self.onSessionChanged.emit(self.session)  

        else:
            self.sessionlist.setCurrentIndex(0)
            self.session = self.sessionlist.currentText() 
            
        self.sessionlist.currentIndexChanged.connect(self.selectionChanged)
        self.layout_operations.addWidget(self.sessionlist)

        self.sessionlist.currentIndexChanged.connect(self.selectionChanged)
        
        # Toolbar Actions
        self.action_close.triggered.connect(self.close) 
        self.action_gui.triggered.connect(self.mainGUI)
        
    def selectionChanged(self,item):
        self.session = self.sessionlist.currentText()
        self.onSessionChanged.emit(self.session)    
        
    def closeEvent(self, *args, **kwargs):
        os.system('ssh root@192.168.1.101 "killall marcos_server"')
        print('GUI closed successfully!')

    def close(self):
        os.system('ssh root@192.168.1.101 "killall marcos_server"')
        print('GUI closed successfully!')
        sys.exit() 
        
    def mainGUI(self):
        self.session = self.sessionlist.currentText()
        mGUI = MainViewController(self.session, pyfirmata=self.pyfirmata)
        mGUI.show()
        self.hide()

        
