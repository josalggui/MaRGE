"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: All sequences on the GUI must be here
"""

from PyQt5.QtWidgets import QTextEdit, QCheckBox, QHBoxLayout
from plotview.spectrumplot import SpectrumPlot
from seq.sweepImage import defaultSweep     # Import general sweep
from PyQt5.QtCore import QObject,  pyqtSlot
from manager.datamanager import DataManager
import numpy as np

class AcqController(QObject):
    def __init__(self, parent=None, functionslist=None):
        super(AcqController, self).__init__(parent)

        self.parent = parent
        self.functionslist = functionslist
        self.acquisitionData = None
        self.layout = QHBoxLayout()

    @pyqtSlot(bool)
    def startAcquisition(self): # It runs when you press acquire buttom
    
        self.layout.setParent(None)
        self.parent.clearPlotviewLayout()

        self.funName = defaultSweep[self.functionslist.getCurrentFunction()].mapVals['seqName']

        # Execute selected sequence
        print('Start sequence')
        defaultSweep[self.funName].sequenceRun(0, self.parent.defaultsequences)  # Run sequence
        defaultSweep[self.funName].sequenceAnalysis()   # Plot results
        print('End sequence')
