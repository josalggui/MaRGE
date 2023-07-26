"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import pyqtgraph as pg


class Plot1DWidget(pg.GraphicsLayoutWidget):
    def __init__(self):
        super(Plot1DWidget, self).__init__()

        # Add label to show data from the cross hair
        self.label2 = pg.LabelItem(justify='left')
        self.addItem(self.label2)

        # Create set of colors
        self.pen = [[255, 0, 0],
                    [0, 255, 0],
                    [255, 255, 0],
                    [0, 0, 255],
                    [255, 0, 255],
                    [0, 255, 255],
                    [255, 255, 255],
                    [128, 128, 128]]

        # Create plot_item to show the figure
        self.plot_item = self.addPlot(row=1, col=0)
        self.plot_item.addLegend()

        # Add lines to show cursor
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        self.plot_item.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_item.addItem(self.crosshair_h, ignoreBounds=True)
        self.proxy = pg.SignalProxy(self.plot_item.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
