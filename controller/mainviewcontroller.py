"""
Main View Controller

@author:    Yolanda Vives

@status:    Sets up the main view, its views and controllers
@todo:

"""

#from PyQt5.QtCore import QFile, QTextStream
#from PyQt5.QtWidgets import QPushButton,  QComboBox,  QGraphicsView
from PyQt5.uic import loadUiType, loadUi
#from PyQt5.QtCore import pyqtSignal, pyqtSlot
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas)
import matplotlib.pyplot as plt

from controller.connectiondialog import ConnectionDialog
from sec.radial import radial
from sec.gradEcho import grad_echo

#from server.communicationmanager import Com

MainWindow_Form, MainWindow_Base = loadUiType('ui/MainWindow.ui')


class MainViewController(MainWindow_Form, MainWindow_Base):
    """
    MainViewController Class
    """
    #onOperationChanged = pyqtSignal(str)

    def __init__(self):
        super(MainViewController, self).__init__()
        self.ui = loadUi('ui/MainWindow.ui')
        self.setupUi(self)
        self.setWindowTitle('PhysioMRI')
        self.initCombo()
        
        # Set figures        
        self.fig1 = plt.figure()
        self.ax1f1 = self.fig1.add_subplot(111)
        self.ax1f1.set_title('K space')
        self.canvas1 = FigureCanvas(self.fig1)
        self.verticalLayout1.addWidget(self.canvas1)
        
        self.fig2 = plt.figure()
        self.ax1f2 = self.fig2.add_subplot(111)
        self.ax1f2.set_title('Image space')
        self.canvas2 = FigureCanvas(self.fig2)
        self.verticalLayout2.addWidget(self.canvas2)
        
        # Widgets actions
        self.PushButton_connect.clicked.connect(self.marcos_server)
        self.comboBox_seq.activated[str].connect(self.onActivated)
        self.PushButton_run.clicked.connect(self.run)
       
    def marcos_server(self):
        self.con = ConnectionDialog(self)
        self.con.show()
        
    def run(self):
        content = self.comboBox_seq.currentText()
        #if content == 'Turbo Spin Echo':
            
        if content == 'Gradient Echo':
            
            # Retrieve values from GUI
            self.lo_freq = float(self.lineEdit_loFreq.text())
            self.rf_amp = float(self.textEdit_rfAmp.toPlainText())
            self.trs = int(self.textEdit_trs.toPlainText())
            self.rf_tstart = int(self.textEdit_rfTstart.toPlainText())
            self.rx_period = int(self.textEdit_rxPeriod.toPlainText())
            self.tx_gate_pre = int(self.textEdit_txGpre.toPlainText())
            self.tx_gate_post = int(self.textEdit_txGpost.toPlainText())            
            self.slice_amp = float(self.textEdit_sliceAmp.toPlainText())       
            self.phase_amp = float(self.textEdit_phAmp.toPlainText())     
            self.readout_amp = float(self.textEdit_rdAmp.toPlainText()) 
            self.rf_duration = int(self.textEdit_rfDur.toPlainText()) 
            self.trap_ramp_duration = int(self.textEdit_trapRampDur.toPlainText()) 
            self.phase_delay = int(self.textEdit_phDelay.toPlainText()) 
            self.phase_duration = int(self.textEdit_phDur.toPlainText()) 

            self.dbg_sc = 1
            self.plot_rx = True
            self.init_gpa = False
            self.rxd1, self.rxd2 = grad_echo(self)   
         
            if self.plot_rx == True:
                self.plot_data()
   
        if content == 'Radial':
            
            # Retrieve values from GUI
            self.lo_freq = float(self.lineEdit_loFreq.text())
            self.rf_amp = float(self.textEdit_rfAmp.toPlainText())
            self.G = float(self.textEdit_Gamp.toPlainText())
            self.trs = int(self.textEdit_trs.toPlainText())
            self.grad_tstart = int(self.textEdit_gradTstart.toPlainText())
            self.tr_total_time = int(self.textEdit_TR.toPlainText())
            self.rf_tstart = int(self.textEdit_rfTstart.toPlainText())
            self.rf_tend = int(self.textEdit_rfTend.toPlainText())
            self.rx_tstart = int(self.textEdit_rxTstart.toPlainText())
            self.rx_tend = int(self.textEdit_rxTend.toPlainText())
            self.rx_period = int(self.textEdit_rxPeriod.toPlainText())
            self.tx_gate_pre = int(self.textEdit_txGpre.toPlainText())
            self.tx_gate_post = int(self.textEdit_txGpost.toPlainText())
            
            self.plot_rx = True
            self.init_gpa = False
            self.rxd1, self.rxd2 = radial(self)
        
            if self.plot_rx == True:
                self.plot_data()
    
    def initCombo(self):
        self.comboBox_seq.addItem('Select Sequences ...')
        self.comboBox_seq.addItem('Turbo Spin Echo')
        self.comboBox_seq.addItem('Gradient Echo')
        self.comboBox_seq.addItem('Radial')
        self.comboBox_seq.addItem('Inversion Recovery')

    def onActivated(self, text):
        
        if text == 'Turbo Spin Echo':
            self.listWidget_seq.insertItem(0, text)
            
        if text == 'Gradient Echo':
            
            self.lineEdit_loFreq.setText("0.1")  # MHz 
            self.textEdit_rfAmp.setPlainText("0.1")  # 1 = full-scale
            self.textEdit_Gamp.setDisabled(True)            
            self.textEdit_trs.setPlainText("2")
            self.textEdit_gradTstart.setDisabled(True)
            self.textEdit_TR.setDisabled(True)
            self.textEdit_rfTend.setDisabled(True)
            self.textEdit_rfTstart.setPlainText("100")  # us
            self.textEdit_rxPeriod.setPlainText("3")  # us
            self.textEdit_rxTstart.setDisabled(True)
            self.textEdit_rxTend.setDisabled(True)       
            self.textEdit_txGpre.setDisabled(False)
            self.textEdit_txGpre.setPlainText("2")  # us, time to start the TX gate before the RF pulse begins
            self.textEdit_txGpost.setDisabled(False)
            self.textEdit_txGpost.setPlainText("1")  # us, time to keep the TX gate on after the RF pulse ends
            self.textEdit_sliceAmp.setDisabled(False)
            self.textEdit_sliceAmp.setPlainText("0.4") # 1 = gradient full-scale
            self.textEdit_phAmp.setDisabled(False)
            self.textEdit_phAmp.setPlainText("0.3") # 1 = gradient full-scale
            self.textEdit_rdAmp.setDisabled(False)
            self.textEdit_rdAmp.setPlainText("0.8") # 1 = gradient full-scale
            self.textEdit_rfDur.setDisabled(False)
            self.textEdit_rfDur.setPlainText("50")
            self.textEdit_trapRampDur.setDisabled(False) 
            self.textEdit_trapRampDur.setPlainText("50")  # us, ramp-up/down time
            self.textEdit_phDelay.setDisabled(False)
            self.textEdit_phDelay.setPlainText("100") # how long after RF end before starting phase ramp-up
            self.textEdit_phDur.setDisabled(False)
            self.textEdit_phDur.setPlainText("200") # length of phase plateau
 
        
        if text == 'Radial':
            # Default values
             ## All times are relative to a single TR, starting at time 0
            self.lineEdit_loFreq.setText("0.2")  # MHz 
            self.textEdit_rfAmp.setPlainText("0.2")  # 1 = full-scale
            self.textEdit_Gamp.setDisabled(False)
            self.textEdit_Gamp.setPlainText("0.5")  # Gx = G cos (t), Gy = G sin (t)
            self.textEdit_trs.setPlainText("36")
            self.textEdit_gradTstart.setDisabled(False)
            self.textEdit_gradTstart.setPlainText("0")  # us
            self.textEdit_TR.setDisabled(False)
            self.textEdit_TR.setPlainText("220")  # start-finish TR time
            self.textEdit_rfTstart.setPlainText("5")  # us
            self.textEdit_rfTend.setDisabled(False)
            self.textEdit_rfTend.setPlainText("50")  # us
            self.textEdit_rxTstart.setDisabled(False)
            self.textEdit_rxTstart.setPlainText("70")  # us
            self.textEdit_rxTend.setDisabled(False) 
            self.textEdit_rxTend.setPlainText("180")  # us
            self.textEdit_rxPeriod.setPlainText("3")  # us
            self.textEdit_txGpre.setPlainText("2")  # us, time to start the TX gate before the RF pulse begins
            self.textEdit_txGpost.setPlainText("1")  # us, time to keep the TX gate on after the RF pulse ends
            self.textEdit_sliceAmp.setDisabled(True)
            self.textEdit_phAmp.setDisabled(True)
            self.textEdit_rdAmp.setDisabled(True)        
            self.textEdit_rfDur.setDisabled(True)          
            self.textEdit_trapRampDur.setDisabled(True)  
            self.textEdit_phDelay.setDisabled(True)  
            self.textEdit_phDur.setDisabled(True)            
    
    def plot_data(self):
        
        self.ax1f1.cla()
        self.ax1f2.cla()
        self.ax1f1.plot(self.rxd1)
        self.ax1f1.set_title('K space')
        self.ax1f2.plot(self.rxd2)
        self.ax1f2.set_title('Image space')
                
        self.canvas1.draw()
        self.canvas2.draw()
