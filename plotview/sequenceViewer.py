"""
SequenceViewer
@author:    Yolanda Vives
@contact:   yolanda.vives@phisiomri.es
@version:   1.0 (Beta)
@change:    21/04/2021

@summary:   Representation of the sequence

@status:    
@todo:      

"""

from PyQt5.uic import loadUiType
from sequencemodes import defaultsequences
from plotview.spectrumplot import SpectrumPlot
from seq.gradEcho2 import grad_echo 
from seq.radial2 import radial
from seq.turboSpinEcho2 import turbo_spin_echo
#from manager.datamanager import DataManager
import pdb
st = pdb.set_trace

SequenceViewer_Form, SequenceViewer_Base = loadUiType('ui/sequenceViewer.ui')

class SequenceViewer(SequenceViewer_Base, SequenceViewer_Form):

    def __init__(self, parent=None, sequencelist=None):
        super(SequenceViewer, self).__init__(parent)

        self.setupUi(self)
        self.parent = parent
        self.sequencelist = sequencelist
        
        self.action_close.triggered.connect(self.close) 
        
    def plotSequence(self):
#        self.clearPlotviewLayout()
        self.sequence = defaultsequences[self.sequencelist.getCurrentSequence()]
        self.sequence.plot_rx = True
        self.sequence.init_gpa = True
        # us, rf pi pulse length  - if None then automatically gets set to 2 * rf_pi2_duration
        self.sequence.rf_pi_duration=None
        
        if self.sequence.seq == 'R':
            self.tx0_t, self.tx0_y,  self.grad_y_t_float, self.grad_y_a_float, self.grad_z_t_float, self.grad_z_a_float = radial(self.sequence)     
        elif self.sequence.seq == 'GE':
            self.tx0_t, self.tx0_y,  self.grad_x_t_float, self.grad_x_a_float, self.grad_y_t_float, self.grad_y_a_float, self.grad_z_t_float, self.grad_z_a_float = grad_echo(self.sequence)   
        elif self.sequence.seq == 'TSE':
            self.tx0_t, self.tx0_y,  self.grad_x_t_float, self.grad_x_a_float, self.grad_y_t_float, self.grad_y_a_float, self.grad_z_t_float, self.grad_z_a_float = turbo_spin_echo(self.sequence)    

        self.tx = SpectrumPlot(self.tx0_t, self.tx0_y, "time", "RF", "RF")
        self.plotview_layout.addWidget(self.tx)  
        if self.sequence.seq != 'R':
            self.gradx = SpectrumPlot(self.grad_x_t_float, self.grad_x_a_float, "time", "X gradient", "Gradient X")
            self.plotview_layout.addWidget(self.gradx)
        self.grady = SpectrumPlot(self.grad_y_t_float, self.grad_y_a_float, "time", "Y gradient", "Gradient Y")
        self.plotview_layout.addWidget(self.grady)
        self.gradz = SpectrumPlot(self.grad_z_t_float, self.grad_z_a_float, "time", "Y gradient", "Gradient Z")
        self.plotview_layout.addWidget(self.gradz)
#        #outputvalues = AcquisitionManager().getOutputParameterObject(dataobject, self.sequence.systemproperties)
#
#        #self.outputsection.set_parameters(outputvalues)
#


        
        
        
#    def clearPlotviewLayout(self) -> None:
#        """
#        Clear the plot layout
#        @return:    None
#        """
#        for i in reversed(range(self.plotview_layout.count())):
#            self.plotview_layout.itemAt(i).widget().setParent(None)
            

