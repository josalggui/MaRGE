"""
Acquisition Controller

@author:    Yolanda Vives
@author:    J.M. Algarín, josalggui@i3m.upv.es
@version:   2.0 (Beta)
"""

from PyQt5.QtWidgets import QLabel, QHBoxLayout
from plotview.spectrumplot import SpectrumPlotSeq
from plotview.spectrumplot import Spectrum3DPlot
from seq.sequences import defaultsequences
from PyQt5 import QtCore
from PyQt5.QtCore import QObject
from seq.localizer import Localizer
import imageio
import numpy as np


class AcquisitionController(QObject):
    def __init__(self, parent=None, session=None, sequencelist=None):
        super(AcquisitionController, self).__init__(parent)

        self.parent = parent
        self.sequencelist = sequencelist
        self.acquisitionData = None
        self.session = session
        # self.layout = QHBoxLayout()
        self.firstPlot()

        # Example of how to add a button
        # self.button = QPushButton("Change View")
        # self.button.setChecked(False)
        # self.button.clicked.connect(self.button_clicked)
        # self.parent.plotview_layout.addWidget(self.button)

    def startAcquisition(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: run selected sequence
        """
        print('Start sequence')

        # Delete previous plots
        # if hasattr(self.parent, 'clearPlotviewLayout'):
        self.parent.clearPlotviewLayout()

        # Load sequence name
        self.seqName = self.sequencelist.getCurrentSequence()

        # Save sequence list into the current sequence, just in case you need to do sweep
        defaultsequences[self.seqName].sequenceList = defaultsequences

        # Save input parameters
        defaultsequences[self.seqName].saveParams()

        # Create and execute selected sequence
        defaultsequences[self.seqName].sequenceRun(0)

        # Do sequence analysis and acquire de plots
        out = defaultsequences[self.seqName].sequenceAnalysis()

        # Create label with rawdata name
        fileName = defaultsequences[self.sequencelist.getCurrentSequence()].mapVals['fileName']
        self.label = QLabel(fileName)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;color: white")
        self.parent.plotview_layout.addWidget(self.label)
        # self.parent.localizer_layout.addWidget(self.label)

        # Add plots to the plotview_layout
        for item in out:
            self.parent.plotview_layout.addWidget(item)
            item.label = self.label

        # self.parent.onSequenceChanged.emit(self.parent.sequence)
        print('End sequence')

    def startSequencePlot(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: rare sequence class
        """
        # Delete previous plots
        # if hasattr(self.parent, 'clearPlotviewLayout'):
        #     self.parent.clearPlotviewLayout()
        # self.layout.setParent(None)
        self.parent.clearPlotviewLayout()

        self.seqName = defaultsequences[self.sequencelist.getCurrentSequence()].mapVals['seqName']

        # Execute selected sequence
        print('Plot sequence')
        defaultsequences[self.seqName].sequenceRun(1)  # Run sequence
        out = defaultsequences[self.seqName].sequencePlot()  # Plot results

        n = 0
        plot = []
        for item in out:
            plot.append(SpectrumPlotSeq(item[0], item[1], item[2], 'Time (ms)', 'Amplitude (a.u.)', item[3]))
            if n > 0: plot[n].plotitem.setXLink(plot[0].plotitem)
            n += 1
        for n in range(4):
            self.parent.plotview_layout.addWidget(plot[n])

    def firstPlot(self):
        logo = imageio.imread("resources/images/logo.png")
        logo = np.flipud(logo)
        self.parent.clearPlotviewLayout()
        welcome = Spectrum3DPlot(logo.transpose([1, 0, 2]),
                                 title='Institute for Instrimentation in Molecular Imaging (i3M)')
        welcome.hideAxis('bottom')
        welcome.hideAxis('left')
        welcome.showHistogram(False)
        welcomeWidget = welcome.getImageWidget()
        self.parent.plotview_layout.addWidget(welcomeWidget)

    def localizer(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: run localizer
        """

        print('Start localizer')

        # Delete previous localizer
        self.parent.clearLocalizerLayout()

        # Set localizer sequence to RARE
        localizer = Localizer()

        # Load default parameters
        localizer.loadParams()
        localizer.mapVals['seqName'] = 'Localizer'
        localizer.mapNmspc['seqName'] = 'LocalizerInfo'
        localizer.saveParams()

        # Add parent to localizer so it can update sequences parameters
        localizer.parent = self.parent

        # Save all sequences into the localizer to set the fov
        localizer.sequenceList = defaultsequences

        # Create and execute selected sequence
        localizer.sequenceRunProjections(0)

        # Do sequence analysis and acquire de plots
        out = localizer.sequenceAnalysis()

        # Create label with rawdata name
        fileName = localizer.mapVals['fileName']
        self.label = QLabel(fileName)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;color: white")
        self.parent.localizer_layout.addWidget(self.label)

        # Add plots to the localizer_layout
        self.parent.localizer_layout.addWidget(out[0])


