"""
Plotview Spectrum (1D Plot)

@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   2.0 (Beta)
@change:    13/07/2020

@summary:   Class for plotting a spectrum

@status:    Plots x and y data
@todo:      Implement more feature from pyqtgraph

"""

from pyqtgraph import GraphicsLayoutWidget, PlotItem
from warnings import warn
from datetime import datetime 


class SpectrumPlot (GraphicsLayoutWidget):
    def __init__(self,
                 xData: list,
                 yData: list,
                 xLabel: str,
                 yLabel: str
                 ):
        super(SpectrumPlot, self).__init__()

        if len(xData) != len(yData):
            warn("Length of x and y data does not match.")
            return
            
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M")

        plotitem = self.addPlot(row=0, col=0)
        plotitem.plot(xData, yData)
        plotitem.setTitle("%s" % dt_string)
#        vb = plotitem.getViewBox()
#        vb.setBackgroundColor('w')
        
        print("x: {}, y: {}".format(xLabel, yLabel))



