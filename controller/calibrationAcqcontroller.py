"""
Calibration Acquisition Controller

@author:    Yolanda Vives
@author:    J.M. Algarín, josalggui@i3m.upv.es
@version:   2.0 (Beta)
"""

from PyQt5.QtWidgets import QTextEdit, QCheckBox, QHBoxLayout, QHBoxLayout
from plotview.spectrumplot import SpectrumPlotSeq
from seq.sequencesCalibration import defaultCalibFunctions
from PyQt5.QtCore import QObject,  pyqtSlot

class CalibrationAcqController(QObject):
    def __init__(self, parent=None, calibfunctionslist=None):
        super(CalibrationAcqController, self).__init__(parent)

        self.parent = parent
        self.calibfunctionslist = calibfunctionslist
        self.acquisitionData = None
        self.layout = QHBoxLayout()

    @pyqtSlot(bool)
    def startAcquisition(self):
        print('Start sequence')

        self.parent.clearPlotviewLayout()

        self.calibfunction = defaultCalibFunctions[self.calibfunctionslist.getCurrentCalibfunction()]
        self.funName = self.calibfunction.mapVals['seqName']

        # Create and execute selected sequence
        defaultCalibFunctions[self.funName].sequenceRun(0)  # Run sequence

        # Do sequence analysis and get plots
        out = defaultCalibFunctions[self.funName].sequenceAnalysis()   # Plot results

        # Add plots to layout
        for item in out:
            self.parent.plotview_layout.addWidget(item)

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

        self.calibfunction = defaultCalibFunctions[self.calibfunctionslist.getCurrentCalibfunction()]
        self.funName = self.calibfunction.mapVals['seqName']

        # Create selected sequence
        print('Plot sequence')
        defaultCalibFunctions[self.funName].sequenceRun(1)  # Run sequence

        # Get sequence instructions plot
        out = defaultCalibFunctions[self.funName].sequencePlot()  # Plot results

        # Add plots to layout
        n = 0
        plot = []
        for item in out:
            plot.append(SpectrumPlotSeq(item[0], item[1], item[2], 'Time (ms)', 'Amplitude (a.u.)', item[3]))
            if n > 0: plot[n].plotitem.setXLink(plot[0].plotitem)
            n += 1
        for n in range(4):
            self.parent.plotview_layout.addWidget(plot[n])