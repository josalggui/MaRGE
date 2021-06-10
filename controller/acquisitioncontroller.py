"""
Acquisition Manager
@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   2.0 (Beta)
@change:    19/06/2020

@summary:   Class for controlling the acquisition

@status:    Under development

"""

from plotview.spectrumplot import SpectrumPlot
from sequencemodes import defaultsequences
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
        
        self.sequence.plot_rx = True
        self.sequence.init_gpa = True 
        
        plotSeq=0
        if self.sequence.seq == 'SE':
            self.rxd, self.flodict, self.msgs=spin_echo(self.sequence, plotSeq)
        if self.sequence.seq == 'SE1D':
            self.rxd, self.flodict, self.msgs=spin_echo1D(self.sequence, plotSeq)
        if self.sequence.seq == 'SE2D':
            self.rxd, self.msgs=spin_echo2D(self.sequence, plotSeq)
        if self.sequence.seq == 'SE3D':
            self.rxd, self.msgs=spin_echo3D(self.sequence, plotSeq)
        if self.sequence.seq == 'FID':
            self.rxd, self.msgs=fid(self.sequence, plotSeq)
        if self.sequence.seq == 'R':
            self.rxd, self.msgs = radial(self.sequence, plotSeq)
        elif self.sequence.seq == 'GE':
            self.rxd, self.msgs = grad_echo(self.sequence, plotSeq)
        elif self.sequence.seq == 'TSE':
            self.sequence.rf_pi_duration=None, # us, rf pi pulse length  - if None then automatically gets set to 2 * rf_pi2_duration
            self.rxd, self.msgs = turbo_spin_echo(self.sequence, plotSeq)

        dataobject: DataManager = DataManager(self.rxd, self.sequence.lo_freq, len(self.rxd))
        self.parent.f_plotview = SpectrumPlot(dataobject.f_axis, dataobject.f_fftMagnitude,[],[],"frequency", "signal intensity", "%s Spectrum" %(self.sequence.seq), 'Frequency')
        self.parent.t_plotview = SpectrumPlot(dataobject.t_axis, dataobject.t_magnitude, dataobject.t_real,dataobject.t_imag,"time", "signal intensity", "%s Raw data" %(self.sequence.seq), 'Time')
       # outputvalues = AcquisitionManager().getOutputParameterObject(dataobject, self.sequence.systemproperties)

        #self.outputsection.set_parameters(outputvalues)
        self.parent.plotview_layout.addWidget(self.parent.t_plotview)
        self.parent.plotview_layout.addWidget(self.parent.f_plotview)
        
        self.save_data()

        self.parent.rxd = self.rxd
        self.parent.lo_freq = self.sequence.lo_freq
        print(self.msgs)

#        self.parent.save_data(self)
        
    def save_data(self):
        
        dataobject: DataManager = DataManager(self.rxd, self.sequence.lo_freq, len(self.rxd))
        dict = vars(defaultsequences[self.sequencelist.getCurrentSequence()])
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H_%M")
        dt2 = date.today()
        dt2_string = dt2.strftime("%d-%m-%Y")
        dict["flodict"] = self.flodict
        dict["rawdata"] = self.rxd
        dict["fft"] = dataobject.f_fftData
        if not os.path.exists('/home/physiomri/share_vm/results_experiments/%s' % (dt2_string)):
            os.makedirs('/home/physiomri/share_vm/results_experiments/%s' % (dt2_string))
            
        if not os.path.exists('/home/physiomri/share_vm/results_experiments/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('/home/physiomri/share_vm/results_experiments/%s/%s' % (dt2_string, dt_string)) 
            
        savemat("/home/physiomri/share_vm/results_experiments/%s/%s/%s.mat" % (dt2_string, dt_string, dict["seq"]), dict) 



        
    


