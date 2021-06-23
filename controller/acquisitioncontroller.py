"""
Acquisition Manager
@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   2.0 (Beta)
@change:    19/06/2020

@summary:   Class for controlling the acquisition

@status:    Under development

"""

from PyQt5.QtWidgets import QLabel
from plotview.spectrumplot import SpectrumPlot
from plotview.spectrumplot import Spectrum2DPlot
from plotview.spectrumplot import Spectrum3DPlot
from sequencemodes import defaultsequences
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
        
#        self.sequence.plot_rx = True
#        self.sequence.init_gpa = True 
        
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
            
        dataobject: DataManager = DataManager(self.data_avg, self.sequence.lo_freq, len(self.data_avg),  self.sequence.n, self.sequence.BW)
        if (self.sequence.n[1] ==1 & self.sequence.n[2] == 1):
            f_plotview = SpectrumPlot(dataobject.f_axis, dataobject.f_fftMagnitude,[],[],"frequency", "signal intensity", "%s Spectrum" %(self.sequence.seq), 'Frequency (kHz)')
            t_plotview = SpectrumPlot(dataobject.t_axis, dataobject.t_magnitude, dataobject.t_real,dataobject.t_imag,"time", "signal intensity", "%s Raw data" %(self.sequence.seq), 'Time (ms)')
            self.parent.plotview_layout.addWidget(t_plotview)
            self.parent.plotview_layout.addWidget(f_plotview)
        else:
            dt = datetime.now()
            dt_string = dt.strftime("%d-%m-%Y_%H:%M:%S")
            label = QLabel("%s %s" % (self.sequence.seq, dt_string))
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet("background-color: black;color: white")
            self.parent.plotview_layout.addWidget(label)
            self.parent.plotview_layout.addWidget(pg.image(dataobject.f_fft2Magnitude))
        
        self.save_data()

        self.parent.rxd = self.rxd
        self.parent.lo_freq = self.sequence.lo_freq
        print(self.msgs)


        
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



        
    


