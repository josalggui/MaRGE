"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import numpy as np

from widgets.widget_plot1d import Plot1DWidget


class Plot1DController(Plot1DWidget):

    def mouseMoved(self, evt):
        pos = evt[0]
        if self.plotitem.sceneBoundingRect().contains(pos):
            curves = self.plotitem.listDataItems()
            x, y = curves[0].getData()
            mousePoint = self.plotitem.vb.mapSceneToView(pos)
            index = np.argmin(np.abs(self.xData - mousePoint.x()))
            self.label2.setText("<span style='font-size: 8pt'>%s=%0.2f, %s=%0.2f</span>" % (self.xLabel,
                                                                                            x[index],
                                                                                            self.yLabel,
                                                                                            y[index]))
            self.crosshair_v.setPos(x[index])
            self.crosshair_h.setPos(y[index])
