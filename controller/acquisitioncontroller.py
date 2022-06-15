"""
Acquisition Manager
@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   2.0 (Beta)
@change:    19/06/2020

@summary:   Class for controlling the acquisition

@status:    Under development

"""

from PyQt5.QtWidgets import QLabel, QPushButton, QHBoxLayout
from plotview.spectrumplot import SpectrumPlot
from plotview.spectrumplot import Spectrum2DPlot
from plotview.spectrumplot import Spectrum3DPlot
from seq.sequences import defaultsequences
#from seq.utilities import change_axes
from PyQt5 import QtCore
from PyQt5.QtCore import QObject,  pyqtSlot,  pyqtSignal, QThread
from manager.datamanager import DataManager
from datetime import date,  datetime 
from scipy.io import savemat
import os
import pyqtgraph as pg
import numpy as np
import nibabel as nib
import pyqtgraph.exporters
from functools import partial
from sessionmodes import defaultsessions

class AcquisitionController(QObject):
    def __init__(self, parent=None, session=None, sequencelist=None):
        super(AcquisitionController, self).__init__(parent)

        self.parent = parent
        self.sequencelist = sequencelist
        self.acquisitionData = None
        self.session = session
        self.layout = QHBoxLayout()

        # Example of how to add a button
        # self.button = QPushButton("Change View")
        # self.button.setChecked(False)
        # self.button.clicked.connect(self.button_clicked)
        # self.parent.plotview_layout.addWidget(self.button)
    
    def startAcquisition(self):

        # Delete previous plots
        if hasattr(self.parent, 'clearPlotviewLayout'):
            self.parent.clearPlotviewLayout()

        # Load sequence name
        self.seqName = defaultsequences[self.sequencelist.getCurrentSequence()].mapVals['seqName']

        # Execute selected sequence
        print('Start sequence')
        defaultsequences[self.seqName].sequenceRun(0)
        defaultsequences[self.seqName].sequenceAnalysis(self)
        print('End sequence')

    def startSequencePlot(self):
        self.layout.setParent(None)
        self.parent.clearPlotviewLayout()

        self.seqName = defaultsequences[self.sequencelist.getCurrentSequence()].mapVals['seqName']

        # Execute selected sequence
        print('Plot sequence')
        defaultsequences[self.seqName].sequenceRun(1)  # Run sequence
        defaultsequences[self.seqName].sequencePlot(self)  # Plot results