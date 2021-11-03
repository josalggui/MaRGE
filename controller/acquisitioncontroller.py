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
from sequencemodes import defaultsequences
from seq.utilities import change_axes
from PyQt5 import QtCore
from PyQt5.QtCore import QObject,  pyqtSlot,  pyqtSignal
from manager.datamanager import DataManager
from seq.radial import radial
from seq.gradEcho import grad_echo
from seq.turboSpinEcho_filter import turbo_spin_echo
from seq.cpmg import cpmg
from seq.rare import rare
from seq.fid import fid
from seq.spinEcho import spin_echo
from datetime import date,  datetime 
from scipy.io import savemat
import os
import pyqtgraph as pg
import numpy as np
import scipy.signal as sig
from scipy.signal import butter, filtfilt 
import nibabel as nib
import pyqtgraph.exporters

class AcquisitionController(QObject):
    def __init__(self, parent=None, sequencelist=None):
        super(AcquisitionController, self).__init__(parent)

        self.parent = parent
        self.sequencelist = sequencelist
        self.acquisitionData = None
        
        self.button = QPushButton("Change View")
        self.button.setChecked(False)
        self.button.clicked.connect(self.button_clicked)
    
    def startAcquisition(self):

        if hasattr(self.parent, 'clearPlotviewLayout'):
            self.parent.clearPlotviewLayout()

        self.sequence = defaultsequences[self.sequencelist.getCurrentSequence()]
        self.sequence.oversampling_factor = 6

        
        if hasattr(self.sequence, 'axes') and self.sequence.seq != 'RARE':
            self.sequence.x, self.sequence.y, self.sequence.z, self.sequence.n_rd, self.sequence.n_ph, self.sequence.n_sl, self.sequence.fov_rd, self.sequence.fov_ph, self.sequence.fov_sl = change_axes(self.sequence)
        else:
            self.sequence.n_rd=1
            self.sequence.n_ph=1
            self.sequence.n_sl=1

        plotSeq=0
        if self.sequence.seq == 'SE':
            self.rxd, self.msgs, self.data_avg=spin_echo(self.sequence, plotSeq)
        elif self.sequence.seq == 'FID':
            self.rxd, self.msgs, self.data_avg=fid(self.sequence, plotSeq)
        elif self.sequence.seq == 'R':
            self.rxd, self.msgs = radial(self.sequence, plotSeq)
        elif self.sequence.seq == 'GE':
            self.rxd, self.msgs = grad_echo(self.sequence, plotSeq)
        elif self.sequence.seq == 'TSE':
            self.rxd, self.msgs, self.data_avg  = turbo_spin_echo(self.sequence, plotSeq)
        elif self.sequence.seq == 'CPMG':
            self.rxd, self.msgs, self.data_avg, self.sequence.BW = cpmg(self.sequence, plotSeq)
            self.sequence.lo_freq=self.sequence.larmorFreq
        elif  self.sequence.seq == 'RARE':
            self.rxd, self.msgs, self.data_avg, self.sequence.BW = rare(self.sequence, plotSeq)
            self.sequence.lo_freq=self.sequence.larmorFreq
            [self.sequence.n_rd, self.sequence.n_ph, self.sequence.n_sl] = self.sequence.nPoints
            
            
        self.dataobject: DataManager = DataManager(self.data_avg, self.sequence.lo_freq, len(self.data_avg), [self.sequence.n_rd, self.sequence.n_ph, self.sequence.n_sl], self.sequence.BW)
        self.sequence.ns = [self.sequence.n_rd, self.sequence.n_ph, self.sequence.n_sl]

        if not hasattr(self.parent, 'batch'):
            if (self.sequence.n_ph ==1 and self.sequence.n_sl == 1):
                f_plotview = SpectrumPlot(self.dataobject.f_axis, self.dataobject.f_fftMagnitude,[],[],"Frequency (kHz)", "Amplitude", "%s Spectrum" %(self.sequence.seq), )
                t_plotview = SpectrumPlot(self.dataobject.t_axis, self.dataobject.t_magnitude, self.dataobject.t_real,self.dataobject.t_imag,'Time (ms)', "Amplitude (mV)", "%s Raw data" %(self.sequence.seq), )
                self.parent.plotview_layout.addWidget(t_plotview)
                self.parent.plotview_layout.addWidget(f_plotview)
                self.parent.f_plotview = f_plotview
                self.parent.t_plotview = t_plotview
                [f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency]=self.dataobject.get_peakparameters()
                print('Peak Value = %0.3f' %(f_signalValue))
    #            snr=self.dataobject.get_snr()
    #            print('SNR:%0.3f' %(snr))
    
    #        elif(self.sequence.seq=='CPMG'):
    #            self.plot_cpmg()
    
            else:
                           
                self.plot_3Dresult()

        self.save_data()    
        self.parent.rxd = self.rxd
        self.parent.lo_freq = self.sequence.lo_freq
        print(self.msgs)
        
    def plot_3Dresult(self):
        
        self.kspace=self.dataobject.k_space
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M:%S")
        self.label = QLabel("%s %s" % (self.sequence.seq, dt_string))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;color: white")
        if (self.sequence.n_ph !=1 & self.sequence.n_sl != 1):  #Add button to change the view only if 3D image
            self.parent.plotview_layout.addWidget(self.button)

        self.parent.plotview_layout.addWidget(self.label)
        self.parent.plotview_layout.addWidget(pg.image( self.dataobject.f_fft2Magnitude))

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
        
        dict = vars(defaultsequences[self.sequencelist.getCurrentSequence()])
        dt = datetime.now()
        dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S")
        dt2 = date.today()
        dt2_string = dt2.strftime("%Y.%m.%d")

        if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('experiments/acquisitions/%s' % (dt2_string))
            
        if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
            
        dict2 = dict
        dict2['rawdata'] = self.rxd
        dict2['average'] = self.data_avg
            
        savemat("experiments/acquisitions/%s/%s/%s.%s.mat" % (dt2_string, dt_string, dict["seq"],dt_string),  dict2) 


        if hasattr(self.dataobject, 'f_fft2Magnitude'):
            nifti_file=nib.Nifti1Image(self.dataobject.f_fft2Magnitude, affine=np.eye(4))
            nib.save(nifti_file, 'experiments/acquisitions/%s/%s/%s.%s.nii'% (dt2_string, dt_string, dict["seq"],dt_string))

        if hasattr(self.parent, 'f_plotview'):
            exporter1 = pyqtgraph.exporters.ImageExporter(self.parent.f_plotview.scene())
            exporter1.export("experiments/acquisitions/%s/%s/Freq%s.png" % (dt2_string, dt_string, self.sequence))
        if hasattr(self.parent, 't_plotview'):
            exporter2 = pyqtgraph.exporters.ImageExporter(self.parent.t_plotview.scene())
            exporter2.export("experiments/acquisitions/%s/%s/Temp%s.png" % (dt2_string, dt_string, self.sequence))

    
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
