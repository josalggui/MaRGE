"""
:author:    José Miguel Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import imageio.v2 as imageio
from importlib import resources
from marge.controller.controller_plot3d import Plot3DController as Spectrum3DPlot
from marge.widgets.widget_figures import FiguresLayoutWidget


class FiguresLayoutController(FiguresLayoutWidget):
    """
    Controller for figures layout
    """
    def __init__(self, main):
        super().__init__()
        self.main = main

        # Show the initial plot
        self.firstPlot()

        # Show the wellcome message
        self.wellcomeMessage()

    @staticmethod
    def wellcomeMessage():
        """
        Display the welcome message.

        This static method displays the welcome message for the Graphical User Interface for MaRCoS. It prints information about the developer, contact details, and the GitHub repository URL.

        Note:
            The method does not return any value.
        """
        print("Graphical User Interface for MaRCoS")
        print("J.M. Algarín, PhD")
        print("josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Valencia, Spain")
        print("https://github.com/mriLab-i3M/MaRGE\n")

    def firstPlot(self):
        """
        Display the first plot.

        This method displays the first plot by loading an image, creating a Spectrum3DPlot object, and customizing its appearance. The plot is added to the figures layout.

        Note:
            The method does not return any value.
        """
        try:
            with resources.path("marge.resources.images", "logo.png") as logo_path:
                logo = imageio.imread(str(logo_path))
        except FileNotFoundError:
            logo = None

        self.clearFiguresLayout()

        if logo is not None:
            welcome = Spectrum3DPlot(
                main=self,
                data=logo.transpose([1, 0, 2]),
                title='Institute for Instrumentation in Molecular Imaging (i3M)'
            )

        welcome.hideAxis('bottom')
        welcome.hideAxis('left')
        welcome.showHistogram(False)
        welcome.ui.menuBtn.hide()
        welcome.ui.roiBtn.hide()
        self.addWidget(welcome)

    def clearFiguresLayout(self) -> None:
        """
        Clear the figures layout.

        This method removes all widgets from the figures layout.

        Returns:
            None
        """
        for ii in range(self.layout.count()):
            item = self.layout.takeAt(0)
            item.widget().deleteLater()
