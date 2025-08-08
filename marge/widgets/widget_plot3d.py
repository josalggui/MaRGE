"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import numpy as np
import pyqtgraph as pg


class Plot3DWidget(pg.ImageView):
    def __init__(self, main):

        # Save inputs into the self
        self.main = main

        # Define the PlotItem to display the image
        self.plot_item = pg.PlotItem()

        # Create text item
        self.text_item = pg.TextItem()
        self.text_item.setText('', color='red')

        # Execute the parent init
        super(Plot3DWidget, self).__init__(view=self.plot_item)

        # Create ROI to get FOV
        self.roiFOV = pg.ROI([0, 0], [1, 1])
        self.view.addItem(self.roiFOV)

        # Insert text_item into the ImageViewer widget
        self.vbox = self.getView()
        self.vbox.addItem(self.text_item)
