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
from PyQt5.QtCore import QObject,  pyqtSlot
from manager.datamanager import DataManager
from seq.radial import radial
from seq.gradEcho import grad_echo
from seq.turboSpinEcho import turbo_spin_echo
from seq.fid import fid
from seq.spinEcho import spin_echo
from seq.spinEcho1D import spin_echo1D
from seq.spinEcho2D import spin_echo2D
from seq.spinEcho3D import spin_echo3D
from datetime import date,  datetime 
from scipy.io import savemat
import os
import pyqtgraph as pg
import numpy as np

class AcquisitionController(QObject):
    def __init__(self, parent=None, sequencelist=None):
        super(AcquisitionController, self).__init__(parent)

        self.parent = parent
        self.sequencelist = sequencelist
        self.acquisitionData = None

    @pyqtSlot(bool)
    def startAcquisition(self):

        self.parent.clearPlotviewLayout()
        self.sequence = defaultsequences[self.sequencelist.getCurrentSequence()]
        
        x, y, z, self.n_rd, self.n_ph, self.n_sl = change_axes(self.sequence)

        plotSeq=0
        if self.sequence.seq == 'SE':
            self.rxd, self.msgs, self.data_avg=spin_echo(self.sequence, plotSeq)
        elif self.sequence.seq == 'SE1D':
            self.rxd, self.msgs=spin_echo1D(self.sequence, plotSeq)
        elif self.sequence.seq == 'SE2D':
            self.rxd, self.msgs=spin_echo2D(self.sequence, plotSeq)
        elif self.sequence.seq == 'SE3D':
            self.rxd, self.msgs=spin_echo3D(self.sequence, plotSeq)
        elif self.sequence.seq == 'FID':
            self.rxd, self.msgs=fid(self.sequence, plotSeq)
        elif self.sequence.seq == 'R':
            self.rxd, self.msgs = radial(self.sequence, plotSeq)
        elif self.sequence.seq == 'GE':
            self.rxd, self.msgs = grad_echo(self.sequence, plotSeq)
        elif self.sequence.seq == 'TSE':
            self.rxd, self.msgs, self.data_avg = turbo_spin_echo(self.sequence, plotSeq)
    
        self.plot_result()
        
    def plot_result(self):
        
        dataobject: DataManager = DataManager(self.data_avg, self.sequence.lo_freq, len(self.data_avg), [self.n_rd, self.n_ph, self.n_sl], self.sequence.BW)
        if (self.n_ph ==1 and self.n_sl == 1):
            f_plotview = SpectrumPlot(dataobject.f_axis, dataobject.f_fftMagnitude,[],[],"Frequency (kHz)", "Amplitude (mV)", "%s Spectrum" %(self.sequence.seq), )
            t_plotview = SpectrumPlot(dataobject.t_axis, dataobject.t_magnitude, dataobject.t_real,dataobject.t_imag,'Time (ms)', "Amplitude (mV)", "%s Raw data" %(self.sequence.seq), )
            self.parent.plotview_layout.addWidget(t_plotview)
            self.parent.plotview_layout.addWidget(f_plotview)
#            [fwhm, fwhm_hz, fwhm_ppm] = dataobject.get_fwhm()
#            print('FWHM:%0.3f'%(fwhm))
#            [f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency]=dataobject.get_peakparameters()
#            print('Max Signal Value = %0.3f' %(f_signalValue))
#            [snr]=dataobject.get_snr()
#            print('SNR:%0.3f' %(snr))

        else:
            dt = datetime.now()
            dt_string = dt.strftime("%d-%m-%Y_%H:%M:%S")
            label = QLabel("%s %s" % (self.sequence.seq, dt_string))
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet("background-color: black;color: white")
#            button = QPushButton("Change View")
#            if (self.n_ph !=1 & self.n_sl != 1):  #Add button to change the view only if 3D image
#                self.parent.plotview_layout.addWidget(button(dataobject))
            self.parent.plotview_layout.addWidget(label)
            self.parent.plotview_layout.addWidget(pg.image(dataobject.f_fft2Magnitude))
#            button.clicked.connect(self.button_clicked)
        
        self.save_data()    

        self.parent.rxd = self.rxd
        self.parent.lo_freq = self.sequence.lo_freq
        print(self.msgs)


    def button_clicked(self, dataobject):
        
        self.parent.clearPlotviewLayout()
        im = dataobject.k_space
        im2=np.moveaxis(im, 0, -1)
        im3 = np.reshape(im2, (n[2]*n[1]*n[0]))      
        
        self.plot_result()
        
        
        
    def save_data(self):
        
#        dataobject: DataManager = DataManager(self.rxd, self.sequence.lo_freq, len(self.rxd), self.sequence.n, self.sequence.BW)
        dict = vars(defaultsequences[self.sequencelist.getCurrentSequence()])
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H_%M_%S")
        dt2 = date.today()
        dt2_string = dt2.strftime("%d-%m-%Y")
#        dict["flodict"] = self.flodict
        dict["rawdata"] = self.rxd
        dict["average"] = self.data_avg
#        dict["fft2D"] = dataobject.f_fft2Data
        if not os.path.exists('/home/physiomri/share_vm/results_experiments/%s' % (dt2_string)):
            os.makedirs('/home/physiomri/share_vm/results_experiments/%s' % (dt2_string))
            
        if not os.path.exists('/home/physiomri/share_vm/results_experiments/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('/home/physiomri/share_vm/results_experiments/%s/%s' % (dt2_string, dt_string)) 
            
        savemat("/home/physiomri/share_vm/results_experiments/%s/%s/%s_%s.mat" % (dt2_string, dt_string, dict["seq"],dt_string),  dict) 
#        savemat('/home/physiomri/share_vm/results_experiments/TSE.mat', dict)



        
    


