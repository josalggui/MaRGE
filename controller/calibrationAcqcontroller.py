"""

@summary:   Class for controlling the acquisition of the calibration

@status:    Under development

"""

from PyQt5.QtWidgets import QTextEdit, QCheckBox, QHBoxLayout
from plotview.spectrumplot import SpectrumPlot
#from plotview.spectrumplot import Spectrum2DPlot
#from plotview.spectrumplot import Spectrum3DPlot
from calibfunctionsmodes import defaultcalibfunctions
from PyQt5.QtCore import QObject,  pyqtSlot
from manager.datamanager import DataManager
from seq.shimming import shimming
from seq.rabiFlops import rabi_flops
from seq.larmor import larmorFreq
from seq.inversionRecovery import inversionRecovery
#from scipy.io import savemat
import numpy as np

class CalibrationAcqController(QObject):
    def __init__(self, parent=None, calibfunctionslist=None):
        super(CalibrationAcqController, self).__init__(parent)

        self.parent = parent
        self.calibfunctionslist = calibfunctionslist
        self.acquisitionData = None
        
        self.layout = QHBoxLayout()
        self.b1 = QCheckBox("Plot Shim x")
        self.b1.setGeometry(200, 150, 100, 30)   #            setting geometry of check box
        self.b1.setStyleSheet("QCheckBox::indicator"
                               "{"
                               "background-color : white;"
                               "selection-color: black;"
                               "}")     #    adding background color to indicator
        self.b1.toggled.connect(lambda:self.btnstate(self.b1))
        self.layout.addWidget(self.b1)
    
        self.b2 = QCheckBox("Plot Shim y")
        self.b2.setGeometry(200, 150, 100, 30)   #            setting geometry of check box
        self.b2.setStyleSheet("QCheckBox::indicator"
                               "{"
                               "background-color : white;"
                               "selection-color: black;"
                               "}")     #    adding background color to indicator
        self.b2.toggled.connect(lambda:self.btnstate(self.b2))
        self.layout.addWidget(self.b2)
        
        self.b3 = QCheckBox("Plot Shim z")
        self.b3.setGeometry(200, 150, 100, 30)   #            setting geometry of check box
        self.b3.setStyleSheet("QCheckBox::indicator"
                               "{"
                               "background-color : white;"
                               "selection-color: black;"
                               "}")     #    adding background color to indicator        
        self.b3.toggled.connect(lambda:self.btnstate(self.b3))
        self.layout.addWidget(self.b3)

    @pyqtSlot(bool)
    def startCalibAcq(self):
    
        self.layout.setParent(None)
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
                d_cropped = values[s:s+samples] 
                
                dataobject: DataManager = DataManager(d_cropped, self.calibfunction.lo_freq, len(d_cropped),  [], self.calibfunction.BW)
                peakValsf.append(round(np.max(dataobject.f_fftMagnitude), 4))
                peakValst.append(dataobject.t_magnitude[0])

                s=s+samples
                i=i+1
            
            f_plotview = SpectrumPlot(dataobject.f_axis, dataobject.f_fftMagnitude,[],[],'Frequency (kHz)', "Amplitude", "%s Spectrum (last pulse)" %(self.calibfunction.cfn) )
            t_plotview = SpectrumPlot(np.linspace(self.calibfunction.rf_pi2_duration, self.calibfunction.rf_pi2_duration+self.calibfunction.N*self.calibfunction.step-self.calibfunction.step, self.calibfunction.N), peakValst, [],[],'Excitation duration (ms)', "pi2 pulse duration", "%s" %(self.calibfunction.cfn))
            self.parent.plotview_layout.addWidget(t_plotview)
            self.parent.plotview_layout.addWidget(f_plotview)
            
        elif self.calibfunction.cfn == 'Inversion Recovery':
            self.rxd, self.msgs=inversionRecovery(self.calibfunction)
            values = self.rxd
            samples = np.int32(len(values)/self.calibfunction.N_ir)
            i=0
            s=0
            peakValsf =[]
            peakValst = []
            while i < self.calibfunction.N_ir:
                d_cropped = values[s:s+samples] 
                
                dataobject: DataManager = DataManager(d_cropped, self.calibfunction.lo_freq, len(d_cropped),  [], self.calibfunction.BW)
                peakValsf.append(round(np.max(dataobject.f_fftMagnitude), 4))
                peakValst.append(dataobject.t_magnitude[0])

                s=s+samples
                i=i+1
            
            f_plotview = SpectrumPlot(dataobject.f_axis, dataobject.f_fftMagnitude,[],[],'Frequency (kHz)', "Amplitude", "%s Spectrum (last pulse)" %(self.calibfunction.cfn) )
            t_plotview = SpectrumPlot(np.linspace(self.calibfunction.echo_duration/2-self.calibfunction.rf_duration*1e-3, self.calibfunction.echo_duration/2-self.calibfunction.rf_duration*1e-3+self.calibfunction.N_ir*self.calibfunction.step-self.calibfunction.step, self.calibfunction.N_ir), peakValst, [],[],'Time between pulses (ms)', "Amplitude", "%s" %(self.calibfunction.cfn))
            self.parent.plotview_layout.addWidget(t_plotview)
            self.parent.plotview_layout.addWidget(f_plotview)
            
        elif self.calibfunction.cfn == 'Shimming':
            self.rxd, self.msgs=shimming(self.calibfunction, 'x')
            values_x = self.rxd
            self.rxd, self.msgs=shimming(self.calibfunction, 'y')
            values_y = self.rxd
            self.rxd, self.msgs=shimming(self.calibfunction, 'z')
            values_z = self.rxd
            
            samples = np.int32(len(values_x)/self.calibfunction.N_shim)
            
            i=0
            s=0
            self.peakValsf_x =[]
            self.fwhmf_x = []
            self.peakValsf_y =[]
            self.fwhmf_y = []
            self.peakValsf_z =[]
            self.fwhmf_z = []
            while i < self.calibfunction.N_shim:
                #############################
                
                d_cropped_x = values_x[s:s+samples] 
                d_cropped_y = values_y[s:s+samples] 
                d_cropped_z = values_z[s:s+samples] 
                
                #############################

                dataobject_x: DataManager = DataManager(d_cropped_x, self.calibfunction.lo_freq, len(d_cropped_x),  [], self.calibfunction.BW)
                f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency = dataobject_x.get_peakparameters()
                self.peakValsf_x.append(f_signalValue)
                fwhm, fwhm_hz, fwhm_ppm = dataobject_x.get_fwhm()
                self.fwhmf_x.append(fwhm_hz)
                
                #############################
                
                dataobject_y: DataManager = DataManager(d_cropped_y, self.calibfunction.lo_freq, len(d_cropped_y),  [], self.calibfunction.BW)
                f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency = dataobject_y.get_peakparameters()
                self.peakValsf_y.append(f_signalValue)
                fwhm, fwhm_hz, fwhm_ppm = dataobject_y.get_fwhm()
                self.fwhmf_y.append(fwhm_hz)              
                
                #############################
                
                dataobject_z: DataManager = DataManager(d_cropped_z, self.calibfunction.lo_freq, len(d_cropped_z),  [], self.calibfunction.BW)
                f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency = dataobject_z.get_peakparameters()
                self.peakValsf_z.append(f_signalValue)
                fwhm, fwhm_hz, fwhm_ppm = dataobject_z.get_fwhm()
                self.fwhmf_z.append(fwhm_hz)                 
                
                ##############################
                
                s=s+samples
                i=i+1

            self.plot_shim(axis='x')
            
        elif self.calibfunction.cfn == 'Larmor Frequency':
            self.peakVals=larmorFreq(self.calibfunction)
            t_plotview = SpectrumPlot(np.linspace(-self.calibfunction.N_larmor/2*self.calibfunction.step, self.calibfunction.N_larmor/2*self.calibfunction.step, self.calibfunction.N_larmor), self.peakVals, [],[],'Larmor Frequency variation (KHz)', "Amplitude", "%s" %(self.calibfunction.cfn))
            self.parent.plotview_layout.addWidget(t_plotview)
            
            
    def plot_shim(self, axis):
        
        
        if axis == 'x':
            plotview1 = SpectrumPlot(np.linspace(self.calibfunction.shim_initial, self.calibfunction.shim_final, self.calibfunction.N_shim), self.peakValsf_x, [],[],"Shim value x", "Peak value", "%s x Peak value" %(self.calibfunction.cfn))
            plotview2 = SpectrumPlot(np.linspace(self.calibfunction.shim_initial, self.calibfunction.shim_final, self.calibfunction.N_shim), self.fwhmf_x, [],[],"Shim value x ", "FHWM", "%s x FHWM" %(self.calibfunction.cfn))

        elif axis == 'y':
            plotview1 = SpectrumPlot(np.linspace(self.calibfunction.shim_initial, self.calibfunction.shim_final, self.calibfunction.N_shim), self.peakValsf_y, [],[],"Shim value", "Peak value", "%s y Peak value" %(self.calibfunction.cfn))
            plotview2 = SpectrumPlot(np.linspace(self.calibfunction.shim_initial, self.calibfunction.shim_final, self.calibfunction.N_shim), self.fwhmf_y, [],[],"Shim value", "FHWM", "%s y FWHM" %(self.calibfunction.cfn))

        elif axis == 'z':
            plotview1 = SpectrumPlot(np.linspace(self.calibfunction.shim_initial, self.calibfunction.shim_final, self.calibfunction.N_shim), self.peakValsf_z, [],[],"Shim value", "Peak value", "%s z Peak value" %(self.calibfunction.cfn))
            plotview2 = SpectrumPlot(np.linspace(self.calibfunction.shim_initial, self.calibfunction.shim_final, self.calibfunction.N_shim), self.fwhmf_z, [],[],"Shim value", "FHWM", "%s z FWHM" %(self.calibfunction.cfn))

#        self.layout.setParent(None)
        self.parent.clearPlotviewLayout()  
        self.parent.plotview_layout.addLayout(self.layout)
        self.parent.plotview_layout.addWidget(plotview1)
        self.parent.plotview_layout.addWidget(plotview2)
            
        shim_x=round(np.max(self.peakValsf_x), 4)
        shim_y=round(np.max(self.peakValsf_y), 4)
        shim_z=round(np.max(self.peakValsf_z), 4)
        
        self.textEdit = QTextEdit()
        self.textEdit.setPlainText('Shim_x=%0.3f,       Shim_y=%0.3f,       Shim_z=%0.3f'%(shim_x, shim_y, shim_z))
        self.parent.plotview_layout.addWidget(self.textEdit)

    def btnstate(self,b):

        if b.text() == 'Plot Shim x':
            if b.isChecked() == True:
                self.plot_shim(axis='x')
                self.b2.setChecked(False)
                self.b3.setChecked(False)  
            
        if b.text() == 'Plot Shim y':
            if b.isChecked() == True:
                self.plot_shim(axis='y')
                self.b1.setChecked(False)
                self.b3.setChecked(False)  

        if b.text() == 'Plot Shim z':
            if b.isChecked() == True:
                self.plot_shim(axis='z')
                self.b1.setChecked(False)
                self.b2.setChecked(False)  

    


