"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import pyqtgraph as pg


class Plot1DWidget(pg.GraphicsLayoutWidget):
    def __init__(self,
                 xData,  # numpy array
                 yData,  # list of numpy array
                 legend,  # list of strings
                 xLabel,  # string
                 yLabel,  # string
                 title,  # string
                 ):
            super(Plot1DWidget, self).__init__()
            self.yData = yData
            self.xData = xData
            self.xLabel = xLabel
            self.yLabel = yLabel
    
            # Check data consistency
            if len(xData) != len(yData[0]):
                print("Length of x and y data does not match.")
                return
    
            # Add label to show data from the cross hair
            self.label2 = pg.LabelItem(justify='left')
            self.addItem(self.label2)
            self.label2.setText("<span style='font-size: 8pt'>%s=%0.2f, %s=%0.2f</span>" % (xLabel, 0, yLabel, 0))
    
            # Create set of colors
            nLines = len(yData)
            pen = [[255, 0, 0],
                   [0, 255, 0],
                   [0, 0, 255],
                   [255, 255, 0],
                   [255, 0, 255],
                   [0, 255, 255],
                   [255, 255, 255],
                   [128, 128, 128]]
    
            self.plot_item = self.addPlot(row=1, col=0)
            self.plot_item.addLegend()
            self.lines = []
            for line in range(nLines):
                self.lines.append(self.plot_item.plot(xData, yData[line], pen=pen[line], name=legend[line]))
                self.plot_item.setXRange(xData[0], xData[len(xData) - 1], padding=0)
    
            self.plot_item.setTitle("%s" % title)
            self.plot_item.setLabel('bottom', xLabel)
            self.plot_item.setLabel('left', yLabel)
            self.crosshair_v = pg.InfiniteLine(angle=90, movable=False)
            self.crosshair_h = pg.InfiniteLine(angle=0, movable=False)
            self.plot_item.addItem(self.crosshair_v, ignoreBounds=True)
            self.plot_item.addItem(self.crosshair_h, ignoreBounds=True)
            self.proxy = pg.SignalProxy(self.plot_item.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)