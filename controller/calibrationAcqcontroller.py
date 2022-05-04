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
from seq.flipAngle import flipAngle
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
                f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency = dataobject.get_peakparameters()
                peakValsf.append(f_signalValue)
                peakValst.append(dataobject.t_magnitude[0])

                s=s+samples
                i=i+1
            
            f_plotview = SpectrumPlot(dataobject.f_axis, dataobject.f_fftMagnitude,[],[],'Frequency (kHz)', "Amplitude", "%s Spectrum (last pulse)" %(self.calibfunction.cfn) )
            t_plotview = SpectrumPlot(np.linspace(self.calibfunction.rf_pi2_duration0, self.calibfunction.rf_pi2_durationEnd,  self.calibfunction.N), peakValst, [],[],'Excitation duration (us)', "pi2 pulse duration", "%s" %(self.calibfunction.cfn))
            self.parent.plotview_layout.addWidget(t_plotview)
            self.parent.plotview_layout.addWidget(f_plotview)
            
        elif self.calibfunction.cfn == 'Inversion Recovery':
            self.rxd, self.msgs=inversionRecovery(self.calibfunction)
            values = self.rxd
            samples = np.int32(len(values)/self.calibfunction.N)
            i=0
            s=0
            peakValsf =[]
            peakValst = []
            while i < self.calibfunction.N:
                d_cropped = values[s:s+samples] 
                
                dataobject: DataManager = DataManager(d_cropped, self.calibfunction.lo_freq, len(d_cropped),  [], self.calibfunction.BW)
                f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency = dataobject.get_peakparameters()
                peakValsf.append(f_signalValue)
                peakValst.append(dataobject.t_magnitude[0])

                s=s+samples
                i=i+1
            
            f_plotview = SpectrumPlot(dataobject.f_axis, dataobject.f_fftMagnitude,[],[],'Frequency (kHz)', "Amplitude", "%s Spectrum (last pulse)" %(self.calibfunction.cfn) )
            t_plotview = SpectrumPlot(np.linspace(self.calibfunction.echo_duration/2-self.calibfunction.rf_duration*1e-3, self.calibfunction.echo_duration/2-self.calibfunction.rf_duration*1e-3+self.calibfunction.N*self.calibfunction.step-self.calibfunction.step, self.calibfunction.N), peakValst, [],[],'Time between pulses (ms)', "Amplitude", "%s" %(self.calibfunction.cfn))
            self.parent.plotview_layout.addWidget(t_plotview)
            self.parent.plotview_layout.addWidget(f_plotview)
            
        elif self.calibfunction.cfn == 'Shimming':
            self.rxd, self.msgs=shimming(self.calibfunction, 'x')
            values_x = self.rxd
            self.rxd, self.msgs=shimming(self.calibfunction, 'y')
            values_y = self.rxd
            self.rxd, self.msgs=shimming(self.calibfunction, 'z')
            values_z = self.rxd
            
            samples = np.int32(len(values_x)/self.calibfunction.N)
            
            i=0
            s=0
            self.peakValsf_x =[]
            self.fwhmf_x = []
            self.peakValsf_y =[]
            self.fwhmf_y = []
            self.peakValsf_z =[]
            self.fwhmf_z = []
            while i < self.calibfunction.N:
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
            repetitions = np.int32(self.calibfunction.step/self.calibfunction.resolution)
            while i < repetitions:            
                self.peakVals, self.freqs=larmorFreq(self.calibfunction)
                t_plotview = SpectrumPlot(self.freqs, self.peakVals, [],[],'Larmor Frequency variation (MHz)', "Amplitude (mV)", "%s" %(self.calibfunction.cfn))
                self.parent.plotview_layout.addWidget(t_plotview)
                self.calibfunction.step = self.calibfunction.step/2
                f_signalValue: float = round(np.max(self.peakVals), 4)
                f_signalIdx: int = np.argmax(self.f_signalValue)  
                self.calibfunction.lo_freq=self.freqs[f_signalIdx]
                i = i+1
                            
        elif self.calibfunction.cfn == 'Amplitude':
            self.rxd, self.msgs, self.amps=flipAngle(self.calibfunction)
            values = self.rxd
            samples = np.int32(len(values)/self.calibfunction.N)
            i=0
            s=0
            peakValsf =[]
            while i < self.calibfunction.N:
                d_cropped = values[s:s+samples] 
                
                dataobject: DataManager = DataManager(d_cropped, self.calibfunction.lo_freq, len(d_cropped),  [], self.calibfunction.BW)
                f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency = dataobject.get_peakparameters()
                peakValsf.append(f_signalValue)

                s=s+samples
                i=i+1
            
#            t_plotview = SpectrumPlot(np.linspace(-self.calibfunction.N/2*self.calibfunction.step, self.calibfunction.N/2*self.calibfunction.step, self.calibfunction.N), peakValsf, [],[],'Flip Angle value', "Amplitude (mV)", "%s" %(self.calibfunction.cfn))
            t_plotview = SpectrumPlot(self.amps, peakValsf, [],[],'Flip Angle value', "Amplitude (mV)", "%s" %(self.calibfunction.cfn))
            self.parent.plotview_layout.addWidget(t_plotview)
           
    def plot_shim(self, axis):
        
        shim_values = np.linspace(self.calibfunction.shim_initial, self.calibfunction.shim_final, self.calibfunction.N)
        
        if axis == 'x':
            plotview1 = SpectrumPlot(shim_values, self.peakValsf_x, [],[],"Shim value x", "Peak value", "%s x Peak value" %(self.calibfunction.cfn))
            plotview2 = SpectrumPlot(shim_values, self.fwhmf_x, [],[],"Shim value x ", "FHWM", "%s x FHWM" %(self.calibfunction.cfn))

        elif axis == 'y':
            plotview1 = SpectrumPlot(shim_values,  self.peakValsf_y, [],[],"Shim value", "Peak value", "%s y Peak value" %(self.calibfunction.cfn))
            plotview2 = SpectrumPlot(shim_values, self.fwhmf_y, [],[],"Shim value", "FHWM", "%s y FWHM" %(self.calibfunction.cfn))

        elif axis == 'z':
            plotview1 = SpectrumPlot(shim_values, self.peakValsf_z, [],[],"Shim value", "Peak value", "%s z Peak value" %(self.calibfunction.cfn))
            plotview2 = SpectrumPlot(shim_values, self.fwhmf_z, [],[],"Shim value", "FHWM", "%s z FWHM" %(self.calibfunction.cfn))

        self.layout.setParent(None)
        self.parent.clearPlotviewLayout()  
        self.parent.plotview_layout.addLayout(self.layout)
        self.parent.plotview_layout.addWidget(plotview1)
        self.parent.plotview_layout.addWidget(plotview2)
            
        max_x=self.peakValsf_x.index(round(np.max(self.peakValsf_x), 4))
        max_y=self.peakValsf_y.index(round(np.max(self.peakValsf_y), 4))
        max_z=self.peakValsf_z.index(round(np.max(self.peakValsf_z), 4))
        
        shim_x=shim_values[max_x]
        shim_y=shim_values[max_y]
        shim_z=shim_values[max_z]
        
        self.textEdit = QTextEdit()
        self.textEdit.setPlainText('Shim_x=%0.5f,       Shim_y=%0.5f,       Shim_z=%0.5f'%(shim_x, shim_y, shim_z))
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

    


