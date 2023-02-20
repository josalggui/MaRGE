"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import imageio.v2 as imageio

from controller.controller_plot3d import Plot3DController as Spectrum3DPlot
from widgets.widget_figures import FiguresLayoutWidget


class FiguresLayoutController(FiguresLayoutWidget):
    def __init__(self, *args, **kwargs):
        super(FiguresLayoutController, self).__init__(*args, **kwargs)

        # Show the initial plot
        self.firstPlot()

        # Show the wellcome message
        self.wellcomeMessage()

    @staticmethod
    def wellcomeMessage():
        print("Graphical User Interface for MaRCoS")
        print("J.M. Algarín, PhD")
        print("josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Valencia, Spain")
        print("https://github.com/yvives/PhysioMRI_GUI\n")

    def firstPlot(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: show the initial figure
        """
        logo = imageio.imread("resources/images/logo.png")
        self.clearFiguresLayout()
        welcome = Spectrum3DPlot(main=self, data=logo.transpose([1, 0, 2]),
                                 title='Institute for Instrumentation in Molecular Imaging (i3M)')
        welcome.hideAxis('bottom')
        welcome.hideAxis('left')
        welcome.showHistogram(False)
        welcome.ui.menuBtn.hide()
        welcome.ui.roiBtn.hide()
        self.addWidget(welcome)

    def clearFiguresLayout(self) -> None:
        for ii in range(self.layout.count()):
            item = self.layout.takeAt(0)
            item.widget().deleteLater()
