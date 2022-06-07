"""
Main View Controller

@author:    Yolanda Vives

@status:    Sets up the main view, its views and controllers
@todo:

"""
from PyQt5.QtWidgets import  QMessageBox,  QFileDialog,  QTextEdit
from PyQt5.QtCore import QFile, QTextStream,  pyqtSignal, pyqtSlot, QThread
from PyQt5.uic import loadUiType, loadUi
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon
from controller.acquisitioncontroller import AcquisitionController
from controller.calibrationcontroller import CalibrationController
from controller.batchcontroller import BatchController
import pyqtgraph.exporters
from functools import partial
import os
import ast
import sys
sys.path.append('/media/physiomri/TOSHIBA\ EXT/')
import experiment as ex
from scipy.io import savemat
from controller.sequencecontroller import SequenceList
from sequencesnamespace import Namespace as nmspc
from sessionmodes import defaultsessions
from manager.datamanager import DataManager
from datetime import date,  datetime 
from globalvars import StyleSheets as style
from stream import EmittingStream
from local_config import ip_address
import mrilabMethods.mrilabMethods as mri
from seq.sequences import defaultsequences
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
    
    def __init__(self, session, parent=None):
        super(MainViewController, self).__init__(parent)
        self.ui = loadUi('ui/mainview.ui')
        self.setupUi(self)
        self.styleSheet = style.breezeLight
        self.setupStylesheet(self.styleSheet)
        
        # Initialisation of sequence list
        self.session = session
        dict = vars(defaultsessions[self.session])
        self.sequencelist = SequenceList(self)
        self.sequencelist.setCurrentIndex(0)
        self.sequencelist.currentIndexChanged.connect(self.selectionChanged)
        self.layout_operations.addWidget(self.sequencelist)
        self.sequence = self.sequencelist.currentText()
        self.session_label.setText(dict["name_code"])
                
        # Console
        self.cons = self.generateConsole('')
        self.layout_output.addWidget(self.cons)
        sys.stdout = EmittingStream(textWritten=self.onUpdateText)
        sys.stderr = EmittingStream(textWritten=self.onUpdateText)        
        
        # Initialisation of acquisition controller
        acqCtrl = AcquisitionController(self, self.session, self.sequencelist)
        
        # Connection to the server
        self.ip = ip_address
        
        # XNAT upload
        self.xnat_active = 'FALSE'
        
        # Toolbar Actions
        self.action_gpaInit.triggered.connect(self.initgpa)
        self.action_calibration.triggered.connect(self.calibrate)
        self.action_changeappearance.triggered.connect(self.changeAppearanceSlot)
        self.action_acquire.triggered.connect(acqCtrl.startAcquisition)
        self.action_loadparams.triggered.connect(self.load_parameters)
        self.action_saveparams.triggered.connect(self.save_parameters)
        self.action_close.triggered.connect(self.close)    
#        self.action_savedata.triggered.connect(self.save_data)
        self.action_exportfigure.triggered.connect(self.export_figure)
        self.action_viewsequence.triggered.connect(self.plot_sequence)
        self.action_batch.triggered.connect(self.batch_system)
        self.action_XNATupload.triggered.connect(self.xnat)
        self.action_session.triggered.connect(self.change_session)

        self.seqName = defaultsequences[self.sequencelist.getCurrentSequence()].mapVals['seqName']
        defaultsequences[self.seqName].sequenceInfo()

    def lines_that_start_with(self, str, f):
        return [line for line in f if line.startswith(str)]
    
    # @staticmethod
    def generateConsole(self, text):
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
        
        dataobject: DataManager = DataManager(self.data_avg, self.sequence.lo_freq, len(self.data_avg), [self.sequence.n_rd, self.sequence.n_ph, self.sequence.n_sl], self.sequence.BW)
        dict1=vars(defaultsessions[self.session])
        dict2 = vars(self.sequence)
        dict = self.merge_two_dicts(dict1, dict2)
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
            
        if not os.path.exists('/media/physiomri/TOSHIBA\ EXT/experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('/media/physiomri/TOSHIBA\ EXT/%s'% (dt2_string) )
            
        if not os.path.exists('/media/physiomri/TOSHIBA\ EXT/experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('/media/physiomri/TOSHIBA\ EXT/%s/%s'% (dt2_string, dt_string) )   
            
        savemat("experiments/acquisitions/%s/%s/%s.mat" % (dt2_string, dt_string, self.sequence), dict)
        savemat("/media/physiomri/TOSHIBA\ EXT/%s/%s/%s.mat" % (dt2_string, dt_string, self.sequence), dict)
        
        self.messages("Data saved")
        
#        if hasattr(self.dataobject, 'f_fft2Magnitude'):
#            nifti_file=nib.Nifti1Image(self.dataobject.f_fft2Magnitude, affine=np.eye(4))
#            nib.save(nifti_file, 'experiments/acquisitions/%s/%s/%s.%s.nii'% (dt2_string, dt_string, dict["seq"],dt_string))

        if hasattr(self.parent, 'f_plotview'):
            exporter1 = pyqtgraph.exporters.ImageExporter(self.f_plotview.scene())
            exporter1.export("experiments/acquisitions/%s/%s/Freq%s.png" % (dt2_string, dt_string, self.sequence))
        if hasattr(self.parent, 't_plotview'):
            exporter2 = pyqtgraph.exporters.ImageExporter(self.t_plotview.scene())
            exporter2.export("experiments/acquisitions/%s/%s/Temp%s.png" % (dt2_string, dt_string, self.sequence))

        from controller.WorkerXNAT import Worker
        
        if self.xnat_active == 'TRUE':
            # Step 2: Create a QThread object
            self.thread = QThread()
            # Step 3: Create a worker object
            self.worker = Worker()
            # Step 4: Move worker to the thread
            self.worker.moveToThread(self.thread)
            # Step 5: Connect signals and slots
            self.thread.started.connect(partial(self.worker.run, 'experiments/acquisitions/%s/%s' % (dt2_string, dt_string)))
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            
            # Step 6: Start the thread
            self.thread.start()

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
   
    def merge_two_dicts(self, x, y):
        z = x.copy()   # start with keys and values of x
        z.update(y)    # modifies z with keys and values of y
        return z
   
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
        dt_string = dt.strftime("%d-%m-%Y_%H_%M_%S")
        dict = vars(defaultsequences[self.sequence]) 
        dict.pop('rawdata', None)
        dict.pop('average', None)       
        
        sequ = '%s' %(self.sequence)
        sequ = sequ.replace(" ", "")
        f = open("experiments/parameterisations/%s_params_%s.txt" % (sequ, dt_string),"w")
        f.write( str(dict) )
        f.close()
  
        self.messages("Parameters of %s sequence saved" %(self.sequence))
        
    def plot_sequence(self):
        
        plotSeq=1
        self.sequence = defaultsequences[self.sequencelist.getCurrentSequence()]
        self.seqName = self.sequence.mapVals['seqName']
        defaultsequences[self.seqName].sequenceRun(plotSeq=plotSeq)
        
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
        expt.add_flodict({
            'grad_vx': (np.array([100]), np.array([0])),
        })
        expt.run()
        expt.__del__()
        print("GPA init done!")

    def batch_system(self):
        batchW = BatchController(self, self.sequencelist)
        batchW.show()

    def xnat(self):
        
        if self.xnat_active == 'TRUE':
            self.xnat_active = 'FALSE'
            self.action_XNATupload.setIcon(QIcon('/home/physioMRI/git_repos/PhysioMRI_GUI/resources/icons/upload-outline.svg') )
            self.action_XNATupload.setToolTip('Activate XNAT upload')
        else:
            self.xnat_active = 'TRUE'
            self.action_XNATupload.setIcon(QIcon('/home/physioMRI/git_repos/PhysioMRI_GUI/resources/icons/upload.svg') )
            self.action_XNATupload.setToolTip('Deactivate XNAT upload')
            
    def change_session(self):
        from controller.sessionviewer_controller import SessionViewerController
        sessionW = SessionViewerController(self.session)
        sessionW.show()
        self.hide()    
