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
from plotview.spectrumplot import Spectrum2DPlot
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
            self.rxd, self.msgs, self.data_avg=spin_echo(self.sequence, plotSeq)
            self.rxd=self.data_avg
#            dataobject: DataManager = DataManager(self.data_avg, self.sequence.lo_freq, len(self.data_avg))
#            if (self.n_ph ==0 & self.n_sl == 0):
#                self.parent.f_plotview = SpectrumPlot(dataobject.f_axis, dataobject.f_fftMagnitude,[],[],"frequency", "signal intensity", "%s Spectrum" %(self.sequence.seq), 'Frequency')
#                self.parent.t_plotview = SpectrumPlot(dataobject.t_axis, dataobject.t_magnitude, dataobject.t_real,dataobject.t_imag,"time", "signal intensity", "%s Raw data" %(self.sequence.seq), 'Time')
#                # outputvalues = AcquisitionManager().getOutputParameterObject(dataobject, self.sequence.systemproperties)
#                #self.outputsection.set_parameters(outputvalues)
#            elif (self.n_sl == 0 & self.n_ph != 0):
#                self.parent.f_plotview = Spectrum2DPlot(dataobject.f_fft2Magnitude,"%s Spectrum" %(self.sequence.seq))
#            else:
#                self.parent.f_plotview = Spectrum2DPlot(dataobject.f_fft2Magnitude,"%s Spectrum" %(self.sequence.seq))
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
            self.sequence.rf_pi_duration=None, # us, rf pi pulse length  - if None then automatically gets set to 2 * rf_pi2_duration
            self.rxd, self.msgs = turbo_spin_echo(self.sequence, plotSeq)
            
#        self.n_rd = self.sequence.n[0]
        dataobject: DataManager = DataManager(self.rxd, self.sequence.lo_freq, len(self.rxd),  self.sequence.n, 250000)
        if (self.sequence.n[1] ==1 & self.sequence.n[2] == 1):
            self.parent.f_plotview = SpectrumPlot(dataobject.f_axis, dataobject.f_fftMagnitude,[],[],"frequency", "signal intensity", "%s Spectrum" %(self.sequence.seq), 'Frequency')
            self.parent.t_plotview = SpectrumPlot(dataobject.t_axis, dataobject.t_magnitude, dataobject.t_real,dataobject.t_imag,"time", "signal intensity", "%s Raw data" %(self.sequence.seq), 'Time')
            self.parent.plotview_layout.addWidget(self.parent.t_plotview)
        elif(self.sequence.n[2] == 1 & self.sequence.n[1] != 1):
            self.parent.f_plotview = Spectrum2DPlot(dataobject.f_fft2Magnitude,"%s Spectrum" %(self.sequence.seq))
        else:
            self.parent.f_plotview = Spectrum2DPlot(dataobject.f_fft2Magnitude,"%s Spectrum" %(self.sequence.seq))
        
        self.parent.plotview_layout.addWidget(self.parent.f_plotview)
        self.save_data()

        self.parent.rxd = self.rxd
        self.parent.lo_freq = self.sequence.lo_freq
        print(self.msgs)


        
    def save_data(self):
        
        dataobject: DataManager = DataManager(self.rxd, self.sequence.lo_freq, len(self.rxd), self.sequence.n)
        dict = vars(defaultsequences[self.sequencelist.getCurrentSequence()])
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H_%M")
        dt2 = date.today()
        dt2_string = dt2.strftime("%d-%m-%Y")
#        dict["flodict"] = self.flodict
        dict["rawdata"] = self.rxd
        dict["fft2D"] = dataobject.f_fft2Data
        if not os.path.exists('/home/physiomri/share_vm/results_experiments/%s' % (dt2_string)):
            os.makedirs('/home/physiomri/share_vm/results_experiments/%s' % (dt2_string))
            
        if not os.path.exists('/home/physiomri/share_vm/results_experiments/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('/home/physiomri/share_vm/results_experiments/%s/%s' % (dt2_string, dt_string)) 
            
        savemat("/home/physiomri/share_vm/results_experiments/%s/%s/%s_%s.mat" % (dt2_string, dt_string, dict["seq"],dt_string),  dict) 



        
    


