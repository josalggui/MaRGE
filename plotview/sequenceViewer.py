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
from seq.graph_seq.GradEcho_graph import grad_echo 
from manager.datamanager import DataManager

SequenceViewer_Form, SequenceViewer_Base = loadUiType('ui/sequenceViewer.ui')

class SequenceViewer(SequenceViewer_Base, SequenceViewer_Form):

    def __init__(self, parent=None, sequencelist=None):
        super(SequenceViewer, self).__init__(parent)

        self.setupUi(self)
        self.parent = parent
        self.sequencelist = sequencelist
        
        self.action_close.triggered.connect(self.close) 
        
    def representSequence(self):
        self.clearPlotviewLayout()
        self.sequence = defaultsequences[self.sequencelist.getCurrentSequence()]
        
        if self.sequence.seq == 'R':
            self.sequence.plot_rx = True
            self.sequence.init_gpa = True
#                self.gdict = radial(self.sequence)     
        elif self.sequence.seq == 'GE':
            self.gdict = grad_echo(self.sequence)   
        elif self.sequence.seq == 'TSE':
            self.sequence.plot_rx = True
            self.sequence.init_gpa = True
            self.sequence.rf_pi_duration=None, # us, rf pi pulse length  - if None then automatically gets set to 2 * rf_pi2_duration
#                self.rxd = turbo_spin_echo(self.sequence)    
        

        self.tx = SpectrumPlot(dataobject.t_axis, self.gdict['tx0'], "time", "RF")
        self.gradx = SpectrumPlot(dataobject.t_axis, self.gdict['grad_vx'], "time", "X gradient")
        self.grady = SpectrumPlot(dataobject.t_axis, self.gdict['grad_vy'], "time", "Y gradient")
        self.gradz = SpectrumPlot(dataobject.t_axis, self.gdict['grad_vz'], "time", "Z gradient")
        #outputvalues = AcquisitionManager().getOutputParameterObject(dataobject, self.sequence.systemproperties)

        #self.outputsection.set_parameters(outputvalues)
        self.plotview_layout.addWidget(self.tx)
        self.plotview_layout.addWidget(self.gradx)
        self.plotview_layout.addWidget(self.grady)
        self.plotview_layout.addWidget(self.gradz)
        
    def clearPlotviewLayout(self) -> None:
        """
        Clear the plot layout
        @return:    None
        """
        for i in reversed(range(self.plotview_layout.count())):
            self.plotview_layout.itemAt(i).widget().setParent(None)
            

