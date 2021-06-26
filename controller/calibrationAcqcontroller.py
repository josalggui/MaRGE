"""

@summary:   Class for controlling the acquisition of the calibration

@status:    Under development

"""

from PyQt5.QtWidgets import QLabel
from plotview.spectrumplot import SpectrumPlot
from plotview.spectrumplot import Spectrum2DPlot
from plotview.spectrumplot import Spectrum3DPlot
from calibfunctionsmodes import defaultcalibfunctions
from PyQt5 import QtCore
from PyQt5.QtCore import QObject,  pyqtSlot
from manager.datamanager import DataManager
from rabiFlops import rabi_flops
from datetime import date,  datetime 
from scipy.io import savemat
import os
import pyqtgraph as pg

class CalibrationAcqController(QObject):
    def __init__(self, parent=None, sequencelist=None):
        super(CalibrationAcqController, self).__init__(parent)

        self.parent = parent
        self.sequencelist = sequencelist
        self.acquisitionData = None

    @pyqtSlot(bool)
    def startCalibAcq(self):

        self.parent.clearPlotviewLayout()
        self.calibfunction = defaultcalibfunctions[self.calibfunctionslist.getCurrentCalibfunction()]
       
        if self.calibfunction.cfn == 'RabiFlops':
            self.rxd, self.msgs=rabi_flops(self.calibfunction)
#        elif self.calibfunction.cfn  == 'SE1D':
#            self.rxd, self.msgs=spin_echo1D(self.sequence)
                    
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
        dict = vars(defaultcalibfunctions[self.calibfunctionslist.getCurrentCalibfunction()])
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




        
    


