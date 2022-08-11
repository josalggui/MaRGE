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
from PyQt5.QtWidgets import QLabel
from PyQt5 import QtCore
from pyqtgraph import GraphicsLayoutWidget
from warnings import warn
import pyqtgraph as pg
import numpy as np

class SpectrumPlot(GraphicsLayoutWidget):
    def __init__(self,
                 xData, # numpy array
                 yData, # list of numpy array
                 legend, # list of strings
                 xLabel, # string
                 yLabel, # string
                 title, # string
                 ):
        super(SpectrumPlot, self).__init__()
        self.yData = yData
        self.xData = xData
        self.xLabel = xLabel
        self.yLabel = yLabel

        # Check data consistency
        if len(xData) != len(yData[0]):
            warn("Length of x and y data does not match.")
            return

        # Add label to show data from the cross hair
        # self.label = pg.LabelItem(justify='right')
        # self.addItem(self.label)
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

        self.plotitem = self.addPlot(row=1, col=0)
        self.plotitem.addLegend()
        for line in range(nLines):
            self.plotitem.plot(xData, yData[line], pen=pen[line], name=legend[line])
            self.plotitem.setXRange(xData[0], xData[len(xData)-1], padding=0)

        self.plotitem.setTitle("%s" % title)
        self.plotitem.setLabel('bottom', xLabel)
        self.plotitem.setLabel('left', yLabel)
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        self.plotitem.addItem(self.crosshair_v, ignoreBounds=True)
        self.plotitem.addItem(self.crosshair_h, ignoreBounds=True)
        self.proxy = pg.SignalProxy(self.plotitem.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

    def mouseMoved(self, evt):
        pos = evt[0]
        if self.plotitem.sceneBoundingRect().contains(pos):
            mousePoint = self.plotitem.vb.mapSceneToView(pos)
            index = np.argmin(np.abs(self.xData-mousePoint.x()))
            self.label2.setText("<span style='font-size: 8pt'>%s=%0.2f, %s=%0.2f</span>" % (self.xLabel,
                                                                                           self.xData[index],
                                                                                           self.yLabel,
                                                                                           self.yData[0][index]))
            self.crosshair_v.setPos(self.xData[index])
            self.crosshair_h.setPos(self.yData[0][index])

class SpectrumPlotSeq(GraphicsLayoutWidget):
    def __init__(self,
                 xData,
                 yData,
                 legend,
                 xLabel,
                 yLabel,
                 title,
                 ):
        super(SpectrumPlotSeq, self).__init__()
        self.yData = yData

        nLines = len(yData)
        pen = [[255, 0, 0],
               [0, 255, 0],
               [0, 0, 255],
               [255, 255, 0],
               [255, 0, 255],
               [0, 255, 255],
               [255, 255, 255],
               [128, 128, 128]]

        self.plotitem = self.addPlot(row=0, col=0)
        self.plotitem.addLegend()
        for line in range(nLines):
            y = yData[line]
            self.plotitem.plot(xData[line], yData[line], pen=pen[line], name=legend[line])
            self.plotitem.setXRange(xData[line][0], xData[line][-1], padding=0)
            if np.min(y) == np.max(y):
                self.plotitem.setYRange(-1, 1, padding=0)

        self.label = pg.TextItem(color=(200, 200, 200), anchor=(0, 0))
        self.plotitem.addItem(self.label)
        self.plotitem.setTitle("%s" % title)
        self.plotitem.setLabel('bottom', xLabel)
        self.plotitem.setLabel('left', yLabel)
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        self.plotitem.addItem(self.crosshair_v, ignoreBounds=True)
        self.plotitem.addItem(self.crosshair_h, ignoreBounds=True)
        self.proxy = pg.SignalProxy(self.plotitem.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

    def mouseMoved(self, e):
        pos = e[0]  ## using signal proxy turns original arguments into a tuple
        if self.plotitem.sceneBoundingRect().contains(pos):
            mousePoint = self.plotitem.vb.mapSceneToView(pos)
            self.label.setText("x = %0.4f, y = %0.4f" % (mousePoint.x(), mousePoint.y()))
            self.crosshair_v.setPos(mousePoint.x())
            self.crosshair_h.setPos(mousePoint.y())

class Spectrum3DPlot():
    def __init__(self,
                 data = np.random.randn(10, 50, 50),
                 xLabel = '',
                 yLabel = '',
                 title = '',
                 ):

        """
        @author: J.M. Algarín, february 03th 2022
        MRILAB @ I3M
        """

        # Save data into the self
        self.data = data

        # Create plot area
        self.plotitem = pg.PlotItem()
        self.plotitem.setLabel(axis='left', text=yLabel)
        self.plotitem.setLabel(axis='bottom', text=xLabel)
        self.plotitem.setTitle(title=title)

        # Create text item
        self.textitem = pg.TextItem()
        self.textitem.setText('', color='red')

        # Create imageView and fit into the plotArea
        self.imv = ImageViewer(view=self.plotitem, textitem=self.textitem)
        self.imv.setImage(data)

        # Insert textitem into the ImageViewer widget
        self.vbox = self.imv.getView()
        self.vbox.addItem(self.textitem)

    def getImageWidget(self):
        return(self.imv)

    def hideAxis(self, axis='bottom'):
        self.plotitem.hideAxis(axis)

    def updateText(self, info):
        self.vbox.removeItem()
        self.textitem.setText(info)
        self.vbox.addItem(self.textitem)

    def setLabel(self, axis, text):
        self.plotitem.setLabel(axis=axis, text=text)

    def setTitle(self, title):
        self.plotitem.setTitle(title=title)

    def showHistogram(self, show=True):
        hist = self.imv.getHistogramWidget()
        hist.setVisible(show)


class ImageViewer(pg.ImageView):
    """
    @author: J.M. Algarín, february 03th 2022
    MRILAB @ I3M
    ImageViewer inherits from ImageView, but when cliking the ROI buttom it shows the mean, std and snr of the given roi
    instead of showing the mean plot
    """

    def __init__(self, parent=None, name="ImageView", view=None, imageItem=None,
                 levelMode='mono', textitem=None, *args):
        # pg.ImageView.__init__(self, parent=None, name="ImageView", view=None, imageItem=None,
        #          levelMode='mono', *args)

        super(ImageViewer, self).__init__(parent=parent, name=name, view=view, imageItem=imageItem,
                 levelMode=levelMode, *args)
        self.textitem = textitem

    def roiChanged(self):
        # Extract image data from ROI
        if self.image is None:
            return

        image = self.getProcessedImage()

        # getArrayRegion axes should be (x, y) of data array for col-major,
        # (y, x) for row-major
        # can't just transpose input because ROI is axisOrder aware
        colmaj = self.imageItem.axisOrder == 'col-major'
        if colmaj:
            axes = (self.axes['x'], self.axes['y'])
        else:
            axes = (self.axes['y'], self.axes['x'])

        data, coords = self.roi.getArrayRegion(
            image.view(np.ndarray), img=self.imageItem, axes=axes,
            returnMappedCoords=True)

        if data is None:
            return

        # Convert extracted data into 1D plot data
        if self.axes['t'] is None:
            # Average across y-axis of ROI
            data = data.mean(axis=self.axes['y'])

            # Get average and std of current slice
            dataSlice = np.reshape(data, -1)  # Here I get the roi of current index
            self.dataAvg = dataSlice.mean()
            self.dataStd = dataSlice.std()
            self.dataSnr = self.dataAvg / self.dataStd

            # get coordinates along x axis of ROI mapped to range (0, roiwidth)
            if colmaj:
                coords = coords[:, :, 0] - coords[:, 0:1, 0]
            else:
                coords = coords[:, 0, :] - coords[:, 0, 0:1]
            xvals = (coords ** 2).sum(axis=0) ** 0.5
        else:
            # Get average and std of current slice
            (ind, time) = self.timeIndex(self.timeLine)
            dataSlice = np.reshape(data[ind, :, :], -1)  # Here I get the roi of current index
            self.dataAvg = dataSlice.mean()
            self.dataStd = dataSlice.std()
            self.dataSnr = self.dataAvg / self.dataStd

            # Average data within entire ROI for each frame
            mean = data.mean(axis=axes)
            std = data.std(axis=axes)
            data = mean/std
            xvals = self.tVals

        # Handle multi-channel data
        if data.ndim == 1:
            plots = [(xvals, data, 'w')]
        if data.ndim == 2:
            if data.shape[1] == 1:
                colors = 'w'
            else:
                colors = 'rgbw'
            plots = []
            for i in range(data.shape[1]):
                d = data[:, i]
                plots.append((xvals, d, colors[i]))

        # Update plot line(s)
        while len(plots) < len(self.roiCurves):
            c = self.roiCurves.pop()
            c.scene().removeItem(c)
        while len(plots) > len(self.roiCurves):
            self.roiCurves.append(self.ui.roiPlot.plot())
        for i in range(len(plots)):
            x, y, p = plots[i]
            self.roiCurves[i].setData(x, y, pen=p)

        # Update textitem
        self.textitem.setText("Mean = %0.1f \nstd = %0.1f \nsnr = %0.1f"%(self.dataAvg, self.dataStd, self.dataSnr))
        self.textitem.show()

    def roiClicked(self):

        showRoiPlot = False
        if self.ui.roiBtn.isChecked():
            showRoiPlot = True
            self.roi.show()
            self.ui.roiPlot.setMouseEnabled(True, True)
            # self.ui.splitter.setSizes([int(self.height() * 0.6), int(self.height() * 0.4)])
            self.ui.splitter.handle(1).setEnabled(True)
            self.roiChanged()
            for c in self.roiCurves:
                c.show()
            self.ui.roiPlot.showAxis('left')
        else:
            self.roi.hide()
            self.ui.roiPlot.setMouseEnabled(False, False)
            for c in self.roiCurves:
                c.hide()
            self.ui.roiPlot.hideAxis('left')
            if hasattr(self, 'textitem'):
                self.textitem.hide()

        if self.hasTimeAxis():
            showRoiPlot = True
            mn = self.tVals.min()
            mx = self.tVals.max()
            self.ui.roiPlot.setXRange(mn, mx, padding=0.01)
            self.timeLine.show()
            self.timeLine.setBounds([mn, mx])
            if not self.ui.roiBtn.isChecked():
                # self.ui.splitter.setSizes([self.height() - 35, 35])
                self.ui.splitter.handle(1).setEnabled(False)
        else:
            self.timeLine.hide()

        self.ui.roiPlot.setVisible(showRoiPlot)

    def updateImage(self, autoHistogramRange=True):
        ## Redraw image on screen
        if self.image is None:
            return

        image = self.getProcessedImage()

        if autoHistogramRange:
            self.ui.histogram.setHistogramRange(self.levelMin, self.levelMax)

        # Transpose image into order expected by ImageItem
        if self.imageItem.axisOrder == 'col-major':
            axorder = ['t', 'x', 'y', 'c']
        else:
            axorder = ['t', 'y', 'x', 'c']
        axorder = [self.axes[ax] for ax in axorder if self.axes[ax] is not None]
        image = image.transpose(axorder)

        # Select time index
        if self.axes['t'] is not None:
            self.ui.roiPlot.show()
            image = image[self.currentIndex]
            if self.ui.roiBtn.isChecked():
                self.roiChanged()

        self.imageItem.updateImage(image)