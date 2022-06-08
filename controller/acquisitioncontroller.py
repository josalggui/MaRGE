"""
Acquisition Manager
@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   2.0 (Beta)
@change:    19/06/2020

@summary:   Class for controlling the acquisition

@status:    Under development

"""

from PyQt5.QtWidgets import QLabel, QPushButton
from plotview.spectrumplot import SpectrumPlot
from plotview.spectrumplot import Spectrum2DPlot
from plotview.spectrumplot import Spectrum3DPlot
from seq.sequences import defaultsequences
#from seq.utilities import change_axes
from PyQt5 import QtCore
from PyQt5.QtCore import QObject,  pyqtSlot,  pyqtSignal, QThread
from manager.datamanager import DataManager
from datetime import date,  datetime 
from scipy.io import savemat
import os
import pyqtgraph as pg
import numpy as np
import nibabel as nib
import pyqtgraph.exporters
from functools import partial
from sessionmodes import defaultsessions

class AcquisitionController(QObject):
    def __init__(self, parent=None, session=None, sequencelist=None):
        super(AcquisitionController, self).__init__(parent)

        self.parent = parent
        self.sequencelist = sequencelist
        self.acquisitionData = None
        self.session = session
        
        self.button = QPushButton("Change View")
        self.button.setChecked(False)
        self.button.clicked.connect(self.button_clicked)
    
    def startAcquisition(self):

        if hasattr(self.parent, 'clearPlotviewLayout'):
            self.parent.clearPlotviewLayout()

        # Load sequence and sequence name
        self.sequence = defaultsequences[self.sequencelist.getCurrentSequence()]
        self.seqName = self.sequence.mapVals['seqName']

        # Execute selected sequence
        plotSeq=0
        print('Start sequence')
        self.rxd, self.msgs, self.data_avg, self.sequence.BW = defaultsequences[self.seqName].sequenceRun(plotSeq=plotSeq)
        print('End sequence')

        [self.sequence.n_rd, self.sequence.n_ph, self.sequence.n_sl]= self.sequence.mapVals['nPoints']
        self.dataobject: DataManager = DataManager(self.data_avg, self.sequence.mapVals['larmorFreq'], len(self.data_avg), self.sequence.mapVals['nPoints'], self.sequence.BW)
        self.sequence.ns = self.sequence.mapVals['nPoints']

        if not hasattr(self.parent, 'batch'):
            if (self.sequence.n_ph ==1 and self.sequence.n_sl == 1):
                f_plotview = SpectrumPlot(self.dataobject.f_axis, self.dataobject.f_fftMagnitude,[],[],"Frequency (kHz)", "Amplitude", "%s Spectrum" %(self.sequence.mapVals['seqName']), )
                t_plotview = SpectrumPlot(self.dataobject.t_axis, self.dataobject.t_magnitude, self.dataobject.t_real,self.dataobject.t_imag,'Time (ms)', "Amplitude (mV)", "%s Raw data" %(self.sequence.mapVals['seqName']), )
                self.parent.plotview_layout.addWidget(t_plotview)
                self.parent.plotview_layout.addWidget(f_plotview)
                self.parent.f_plotview = f_plotview
                self.parent.t_plotview = t_plotview
                [f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency]=self.dataobject.get_peakparameters()
                print('Peak Value = %0.3f' %(f_signalValue))
    
            else:
                           
                self.plot_3Dresult()

        self.parent.rxd = self.rxd
        self.parent.data_avg = self.data_avg
        self.parent.sequence = self.sequence
        print(self.msgs)
        #self.parent.save_data()         
        self.save_data()
        
    def plot_3Dresult(self):
        
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M:%S")
        self.label = QLabel("%s %s" % (self.sequence.mapVals['seqName'], dt_string))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;color: white")
        if (self.sequence.n_ph !=1 & self.sequence.n_sl != 1):  #Add button to change the view only if 3D image
            self.parent.plotview_layout.addWidget(self.button)

        self.parent.plotview_layout.addWidget(self.label)

        # Plot image
        self.parent.plotview_layout.addWidget(pg.image(np.abs(self.rxd['image3D'])))

        # Plot k-space
        self.parent.plotview_layout.addWidget(pg.image(np.log10(np.abs(self.rxd['kSpace3D']))))

    @pyqtSlot()
    def button_clicked(self):
        
#        if self.button.isChecked():
        self.parent.clearPlotviewLayout()
        im = self.kspace
        im2=np.moveaxis(im, 0, -1)
        im3 = np.reshape(im2, (self.sequence.n_sl*self.sequence.n_ph*self.sequence.n_rd))    
        self.sequence.ns = self.sequence.ns[1:]+self.sequence.ns[:1]

        self.dataobject: DataManager = DataManager(im3, self.sequence.lo_freq, len(im3),self.sequence.ns, self.sequence.BW)
        
        self.button.setChecked(False)
        
        self.plot_3Dresult()
        
    def save_data(self):
        
        dict1 = vars(defaultsessions[self.session])
        #dict2 = vars(defaultsequences[self.sequencelist.getCurrentSequence()])
        dict2 = self.rxd
        
        dt = datetime.now()
        dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
        dt2 = date.today()
        dt2_string = dt2.strftime("%Y.%m.%d")

        if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('experiments/acquisitions/%s' % (dt2_string))
            
        if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
            
        if not os.path.exists('/media/physiomri/TOSHIBA EXT/experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('/media/physiomri/TOSHIBA EXT/experiments/acquisitions/%s'% (dt2_string) )
            
        if not os.path.exists('/media/physiomri/TOSHIBA EXT/experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('/media/physiomri/TOSHIBA EXT/experiments/acquisitions/%s/%s'% (dt2_string, dt_string) )
            
        dict = self.merge_two_dicts(dict1, dict2)
        #dict['rawdata'] = self.rxd
        #dict['average'] = self.data_avg
            
        savemat("experiments/acquisitions/%s/%s/%s.%s.%s.mat" % (dt2_string, dt_string, dict["name_code"], self.sequence.mapVals['seqName'], dt_string),  dict)
        savemat("/media/physiomri/TOSHIBA EXT/experiments/acquisitions/%s/%s/%s.%s.%s.mat" %(dt2_string, dt_string, dict["name_code"], self.sequence.mapVals['seqName'],dt_string), dict)

        if hasattr(self.dataobject, 'f_fft2Magnitude'):
            nifti_file=nib.Nifti1Image(self.dataobject.f_fft2Magnitude, affine=np.eye(4))
            nib.save(nifti_file, 'experiments/acquisitions/%s/%s/%s.%s.%s.nii'% (dt2_string, dt_string, dict["name_code"],self.sequence.mapVals['seqName'],dt_string))
            nib.save(nifti_file, '/media/physiomri/TOSHIBA EXT/experiments/acquisitions/%s/%s/%s.%s.%s.nii'% (dt2_string, dt_string, dict["name_code"],self.sequence.mapVals['seqName'],dt_string))

        if hasattr(self.parent, 'f_plotview'):
            exporter1 = pyqtgraph.exporters.ImageExporter(self.parent.f_plotview.scene())
            exporter1.export("experiments/acquisitions/%s/%s/Freq.%s.%s.png" % (dt2_string, dict["name_code"], dt_string, self.sequence))
        if hasattr(self.parent, 't_plotview'):
            exporter2 = pyqtgraph.exporters.ImageExporter(self.parent.t_plotview.scene())
            exporter2.export("experiments/acquisitions/%s/%s/Temp.%s.%s.png" % (dt2_string, dict["name_code"], dt_string, self.sequence))

        from controller.WorkerXNAT2 import run
        
#        if self.parent.xnat_active == 'TRUE':
#            # Step 2: Create a QThread object
#            self.parent.thread = QThread()
#            # Step 3: Create a worker object
#            self.worker = Worker()
#            # Step 4: Move worker to the thread
#            self.worker.moveToThread(self.parent.thread)
#            # Step 5: Connect signals and slots
#            self.parent.thread.started.connect(partial(self.worker.run, 'experiments/acquisitions/%s/%s' % (dt2_string, dt_string)))
#            self.worker.finished.connect(self.parent.thread.quit)
#            self.worker.finished.connect(self.worker.deleteLater)
#            self.parent.thread.finished.connect(self.parent.thread.deleteLater)
#            
#            # Step 6: Start the thread
#            self.parent.thread.start()

        if self.parent.xnat_active == 'TRUE':
            run(self,'experiments/acquisitions/%s/%s' % (dt2_string, dt_string))
    

    def merge_two_dicts(self, x, y):
        z = x.copy()   # start with keys and values of x
        z.update(y)    # modifies z with keys and values of y
        return z
   
    def plot_cpmg(self):
        
        data = self.data
        etl = self.etl
        echoSpacing = self.echoSpacing
        
        t = (np.arange(etl)*echoSpacing+echoSpacing)*1e-3
        
        # Fitting
        dataLog = np.log(data)
        fitting = np.polyfit(t, dataLog, 1)
        dataFitting = np.poly1d(fitting)
        dataFitLog = dataFitting(t)
        dataFit = np.exp(dataFitLog)
        T2 = -1/fitting[0]
        
        #    # Plot data
#    plt.plot(t, data, 'o', t, dataFit, 'r')
#    plt.ylabel('Echo amplitude (mV)')
#    plt.xlabel('Echo time (ms)')
#    plt.legend(['Experimental', 'Fitting'])
#    plt.title('CPMG, T2 = '+str(round(T2, 1))+' ms')
#    plt.show()

