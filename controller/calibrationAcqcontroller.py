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
from seq.rabiFlops import rabi_flops
from datetime import date,  datetime 
from scipy.io import savemat
import os
import pyqtgraph as pg
import numpy as np

class CalibrationAcqController(QObject):
    def __init__(self, parent=None, calibfunctionslist=None):
        super(CalibrationAcqController, self).__init__(parent)

        self.parent = parent
        self.calibfunctionslist = calibfunctionslist
        self.acquisitionData = None

    @pyqtSlot(bool)
    def startCalibAcq(self):

        self.parent.clearPlotviewLayout()
        self.calibfunction = defaultcalibfunctions[self.calibfunctionslist.getCurrentCalibfunction()]
       
        if self.calibfunction.cfn == 'Rabi Flops':
            self.rxd, self.msgs=rabi_flops(self.calibfunction)
            values = self.rxd
            samples = np.int32(len(values)/self.calibfunction.N)
            i=0
            s=0
            peakValsf =[]
            peakValst = []
            while i < self.calibfunction.N:
                d_cropped = values[s:s+samples-1] 
                
                dataobject: DataManager = DataManager(d_cropped, self.calibfunction.lo_freq, len(d_cropped),  [], self.calibfunction.BW)
                peakValsf.append(round(np.max(dataobject.f_fftMagnitude), 4))
                peakValst.append(dataobject.t_magnitude[0])

                s=s+samples
                i=i+1
            
            f_plotview = SpectrumPlot(dataobject.f_axis, dataobject.f_fftMagnitude,[],[],"frequency", "signal intensity", "%s Spectrum last pulse" %(self.calibfunction.cfn), 'Frequency (kHz)')
            t_plotview = SpectrumPlot(np.linspace(self.calibfunction.rf_pi2_duration, self.calibfunction.rf_pi2_duration+self.calibfunction.N*self.calibfunction.step-self.calibfunction.step, self.calibfunction.N), peakValst, [],[],"time", "pi2 pulse duration", "%s First Value of the time signal" %(self.calibfunction.cfn), 'Excitation duration (ms)')
            self.parent.plotview_layout.addWidget(t_plotview)
            self.parent.plotview_layout.addWidget(f_plotview)
#            
#            
##        elif self.calibfunction.cfn  == 'SE1D':
##            self.rxd, self.msgs=spin_echo1D(self.sequence)
#                    
#        
#        if (self.sequence.n[1] ==1 & self.sequence.n[2] == 1):
#            f_plotview = SpectrumPlot(dataobject.f_axis, dataobject.f_fftMagnitude,[],[],"frequency", "signal intensity", "%s Spectrum" %(self.sequence.seq), 'Frequency (kHz)')
#            t_plotview = SpectrumPlot(dataobject.t_axis, dataobject.t_magnitude, dataobject.t_real,dataobject.t_imag,"time", "signal intensity", "%s Raw data" %(self.sequence.seq), 'Time (ms)')
#
#        else:
#            dt = datetime.now()
#            dt_string = dt.strftime("%d-%m-%Y_%H:%M:%S")
#            label = QLabel("%s %s" % (self.sequence.seq, dt_string))
#            label.setAlignment(QtCore.Qt.AlignCenter)
#            label.setStyleSheet("background-color: black;color: white")
#            self.parent.plotview_layout.addWidget(label)
#            self.parent.plotview_layout.addWidget(pg.image(dataobject.f_fft2Magnitude))
#        
#        self.save_data()
#
#        self.parent.rxd = self.rxd
#        self.parent.lo_freq = self.sequence.lo_freq
#        print(self.msgs)


        
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




        
    


