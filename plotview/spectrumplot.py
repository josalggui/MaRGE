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

from pyqtgraph import GraphicsLayoutWidget
from warnings import warn
from datetime import datetime 

class SpectrumPlot (GraphicsLayoutWidget):
    def __init__(self,
                 xData: list,
                 yData: list,
                 yData2:list, 
                 yData3:list, 
                 xLabel: str,
                 yLabel: str, 
                 title:str, 
                 xlabel:str
                 ):
        super(SpectrumPlot, self).__init__()

        if len(xData) != len(yData):
            warn("Length of x and y data does not match.")
            return
            
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M")

        plotitem = self.addPlot(row=0, col=0)
        plotitem.addLegend()
        plotitem.plot(xData, yData, pen=[255, 0, 0], name="magnitude")
        if yData2 !=[]:
            plotitem.plot(xData, yData2, pen=[0, 255, 0], name="real part")
            plotitem.plot(xData, yData3, pen=[0, 0, 255], name="imaginary part")
        plotitem.setTitle("%s %s" % (title, dt_string))
        plotitem.setLabel('left', 'Amplitude')
        plotitem.setLabel('bottom', xlabel)
 
#        vb = plotitem.getViewBox()
#        vb.setBackgroundColor('w')
        
#        print("x: {}, y: {}".format(xLabel, yLabel))

class Spectrum2DPlot(GraphicsLayoutWidget):
    def __init__(self,
                 Data: list,
                 title:str
                 ):
        super(Spectrum2DPlot, self).__init__()

        plotitem = self.addPlot(row=0, col=0)
        plotitem.addLegend()
        plotitem.plot(Data, pen=[255, 0, 0], name="magnitude")
        
