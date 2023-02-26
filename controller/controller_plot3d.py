"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import numpy as np

from seq.sequences import defaultsequences
from widgets.widget_plot3d import Plot3DWidget
import configs.hw_config as hw


class Plot3DController(Plot3DWidget):
    def __init__(self, data=np.random.randn(10, 50, 50), x_label='', y_label='', title='', *args, **kwargs):
        super(Plot3DController, self).__init__(*args, **kwargs)
        self.data = data
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.img_resolution = None

        # Set plot_item properties
        self.plot_item.setLabel(axis='left', text=y_label)
        self.plot_item.setLabel(axis='bottom', text=x_label)
        self.plot_item.setTitle(title=title)

        # hide FOV button if not image
        self.ui.menuBtn.hide()
        if title == "Sagittal" or title == "Coronal" or title == "Transversal":
            self.ui.menuBtn.show()

        # Change the Norm button to FOV button
        self.ui.menuBtn.setText("FOV")
        self.ui.menuBtn.setCheckable(True)

        # Create ROI to get FOV
        self.roiFOV.setSize(np.size(data, 1), np.size(data, 2), update=False)
        self.roiFOV.addScaleHandle([1, 1], [0.5, 0.5])
        self.roiFOV.addRotateHandle([0, 0], [0.5, 0.5])
        self.roiFOV.setZValue(20)
        self.roiFOV.setPen('y')
        self.roiFOV.hide()
        self.roiFOV.sigRegionChangeFinished.connect(self.roiFOVChanged)

        # Modify ROI to get SNR
        self.roi.setPen('g')

        # Add image
        self.setImage(data)

    def menuClicked(self):
        # Now the menu button is the FOV button
        if self.ui.menuBtn.isChecked():
            # Get the corresponding axes from the image
            x_axis = 0
            y_axis = 1
            z_axis = 2
            if self.title == "Sagittal":
                x_axis = 1
                y_axis = 0
                z_axis = 2
            elif self.title == "Coronal":
                x_axis = 2
                y_axis = 0
                z_axis = 1
            elif self.title == "Transversal":
                x_axis = 2
                y_axis = 1
                z_axis = 0

            # Get image fov and resolution
            current_output = self.main.history_list.current_output
            n_points = np.array(np.shape(self.getProcessedImage()))[1::]
            fov = self.main.history_list.fovs[current_output][-1]
            img_fov_ru = [fov[x_axis], fov[y_axis]]
            self.img_resolution = np.array(img_fov_ru) / n_points

            # Set roi size and angle
            self.roiFOV.setPos([0.0, 0.0], update=False)
            self.roiFOV.setSize(n_points, update=False)
            self.roiFOV.setAngle(0.0, update=False)
            self.roiFOV.stateChanged()

            # Show the roi
            self.roiFOV.show()
        else:
            self.roiFOV.hide()

    def roiFOVChanged(self):
        # Get the corresponding axes from the image
        x_axis = 0
        y_axis = 1
        z_axis = 2
        d = [1, 1]
        if self.title == "Sagittal":
            d = [-1, -1]
            x_axis = 1
            y_axis = 0
            z_axis = 2
        elif self.title == "Coronal":
            d = [1, -1]
            x_axis = 2
            y_axis = 0
            z_axis = 1
        elif self.title == "Transversal":
            d = [1, -1]
            x_axis = 2
            y_axis = 1
            z_axis = 0

        # Get roi properties in pixel units
        ima_fov_px = np.array(np.shape(self.getProcessedImage()))[1::]
        roi_fov_px = self.roiFOV.size()
        roi_pos_px = self.roiFOV.pos()
        roi_angle = self.roiFOV.angle()

        # ROI center in pixel units
        x0_px = (+ (ima_fov_px[0] / 2 - roi_pos_px[0]) * np.cos(roi_angle * np.pi / 180)
                 + (ima_fov_px[1] / 2 - roi_pos_px[1]) * np.sin(roi_angle * np.pi / 180))
        y0_px = (- (ima_fov_px[0] / 2 - roi_pos_px[0]) * np.sin(roi_angle * np.pi / 180)
                 + (ima_fov_px[1] / 2 - roi_pos_px[1]) * np.cos(roi_angle * np.pi / 180))

        # ROI center in real units
        x0_ru = (x0_px - roi_fov_px[0]/2) * self.img_resolution[0]
        y0_ru = (y0_px - roi_fov_px[1]/2) * self.img_resolution[1]

        # Set fov properties in true units
        fov_roi = [0, 0, 0]
        fov_roi[x_axis] = np.round(roi_fov_px[0]*self.img_resolution[0] * 1e2, decimals=1) # cm
        fov_roi[y_axis] = np.round(roi_fov_px[1]*self.img_resolution[1] * 1e2, decimals=1) # cm
        dfov_roi = [0, 0, 0]
        dfov_roi[x_axis] = d[0]*np.round(x0_ru * 1e3, decimals=1) # mm
        dfov_roi[y_axis] = d[1]*np.round(y0_ru * 1e3, decimals=1) # mm
        hw.fov = fov_roi.copy()
        hw.dfov = dfov_roi.copy()

        # Define rotation
        rotation = [0, 0, 0, 0]
        rotation[z_axis] = 1
        rotation[3] = roi_angle

        # Update sequence parameters
        for sequence in defaultsequences.values():
            if 'fov' in sequence.mapKeys:
                sequence.mapVals['fov'][x_axis] = np.round(fov_roi[x_axis])       # cm
                sequence.mapVals['fov'][y_axis] = np.round(fov_roi[y_axis])       # cm
            if 'dfov' in sequence.mapKeys:
                sequence.mapVals['dfov'][x_axis] = np.round(dfov_roi[x_axis])     # mm
                sequence.mapVals['dfov'][y_axis] = np.round(dfov_roi[y_axis])     # mm
            if 'angle' in sequence.mapKeys:
                sequence.mapVals['angle'] = np.round(rotation[3], decimals=2)                        # degrees
            if 'rotationAxis' in sequence.mapKeys:
                sequence.mapVals['rotationAxis'] = rotation[0:3]

        self.main.sequence_list.updateSequence()

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

        # Update text_item
        self.text_item.setText("Mean = %0.1f \nstd = %0.1f \nsnr = %0.1f"%(self.dataAvg, self.dataStd, self.dataSnr))
        self.text_item.show()

    def roiClicked(self):

        show_roi_plot = False
        if self.ui.roiBtn.isChecked():
            show_roi_plot = True
            self.roi.show()
            self.ui.roiPlot.setMouseEnabled(True, True)
            # self.ui.splitter.setSizes([int(self.height() * 0.6), int(self.height() * 0.4)])
            self.ui.splitter.handle(1).setEnabled(True) # Allow to change the window size
            self.roiChanged()
            for c in self.roiCurves:
                c.hide()
            # self.ui.roiPlot.showAxis('left')
        else:
            self.roi.hide()
            self.ui.roiPlot.setMouseEnabled(False, False)
            for c in self.roiCurves:
                c.hide()
            self.ui.roiPlot.hideAxis('left')
            if hasattr(self, 'text_item'):
                self.text_item.hide()

        if self.hasTimeAxis():
            show_roi_plot = True
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

        self.ui.roiPlot.setVisible(show_roi_plot)
    
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

    def hideAxis(self, axis):
        self.plot_item.hideAxis(axis)

    def updateText(self, info):
        self.vbox.removeItem()
        self.text_item.setText(info)
        self.vbox.addItem(self.textitem)

    def setLabel(self, axis, text):
        self.plot_item.setLabel(axis=axis, text=text)

    def setTitle(self, title):
        self.plot_item.setTitle(title=title)

    def showHistogram(self, show=True):
        hist = self.getHistogramWidget()
        hist.setVisible(show)
