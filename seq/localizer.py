"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: All sequences on the GUI must be here
"""

import os
import sys
# *****************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char == '\\' or char == '/') and path[ii + 1:ii + 14] == 'PhysioMRI_GUI':
        sys.path.append(path[0:ii + 1] + 'PhysioMRI_GUI')
        sys.path.append(path[0:ii + 1] + 'marcos_client')
    ii += 1
# ******************************************************************************
import seq.rare as rare
import numpy as np
import pyqtgraph as pg
import copy


class Localizer(rare.RARE):
    def __init__(self):
        super(Localizer, self).__init__()
        self.mapVals['seqName'] = 'Localizer'
        self.mapNmspc['seqName'] = 'LocalizerInfo'
        self.pos = [0, 0, 0]                    # Global position of each axis
        self.fov = self.mapVals['nPoints']      # Global fov (in units of matrix size)

    def sequenceRunProjections(self, plotSeq=0, demo=False):
        # Set projection image
        self.mapVals['axesEnable'] = [1, 1, 0]
        nSL = self.mapVals['nPoints'][2]
        self.mapVals['nPoints'][2] = 1
        self.mapVals['freqCal'] = 0

        # Run first projection localizer
        self.mapVals['axes'] = [0, 1, 2]
        self.sequenceRun()
        self.proj1 = np.squeeze(np.abs(self.mapVals['image3D']))

        # Run second projection localizer
        self.mapVals['axes'] = [0, 2, 1]
        self.sequenceRun()
        self.proj2 = np.squeeze(np.abs(self.mapVals['image3D']))

        # Run third projection localizer
        self.mapVals['axes'] = [1, 2, 0]
        self.sequenceRun()
        self.proj3 = np.squeeze(np.abs(self.mapVals['image3D']))

        # Set the nPoints and axes to the original values:
        self.mapVals['nPoints'][2] = nSL
        self.mapVals['axes'] = [0, 1, 2]
        self.mapVals['axesEnable'] = [1, 1, 1]

    def sequenceAnalysis(self, obj=''):
        nPoints = np.array(self.mapVals['nPoints'])
        realFov = np.array(self.mapVals['fov'])
        resolution = realFov / nPoints
        self.fov = self.mapVals['nPoints']

        # Get axes in strings
        axes = self.mapVals['axes']
        axesDict = {'x': 0, 'y': 1, 'z': 2}
        axesKeys = list(axesDict.keys())
        axesVals = list(axesDict.values())
        axesStr = ['', '', '']
        n = 0
        for val in axes:
            index = axesVals.index(val)
            axesStr[n] = axesKeys[index]
            n += 1

        # Create graphics layout
        win = pg.GraphicsLayoutWidget(show=(obj == 'Standalone'))
        win.resize(300, 1000)

        # Images
        plot1 = win.addPlot(row=0, col=0)  # Plot axes
        img1 = pg.ImageItem()  # Image item to show image
        plot1.addItem(img1)  # Fix image item into plot axes
        img1.setImage(self.proj1)  # Set the image to show into image item
        plot1.setLabel("bottom", 'y')
        plot1.setLabel("left", 'x')
        plot1.getViewBox().invertY(True)

        plot2 = win.addPlot(row=0, col=1)
        img2 = pg.ImageItem()
        plot2.addItem(img2)
        img2.setImage(self.proj2)
        plot2.setLabel("bottom", 'z')
        plot2.setLabel("left", 'x')
        plot2.getViewBox().invertY(True)

        plot3 = win.addPlot(row=0, col=2)
        img3 = pg.ImageItem()
        plot3.addItem(img3)
        img3.setImage(self.proj3)
        plot3.setLabel("bottom", 'z')
        plot3.setLabel("left", 'y')
        plot3.getViewBox().invertY(True)

        # Custom ROI for selecting an image region
        roi1 = pg.ROI([0, 0], [nPoints[0], nPoints[1]])
        roi1.addScaleHandle([0.5, 1], [0.5, 0.5])
        roi1.addScaleHandle([0, 0.5], [0.5, 0.5])
        plot1.addItem(roi1)

        roi2 = pg.ROI([0, 0], [nPoints[1], nPoints[2]])
        roi2.addScaleHandle([0.5, 1], [0.5, 0.5])
        roi2.addScaleHandle([0, 0.5], [0.5, 0.5])
        plot2.addItem(roi2)

        roi3 = pg.ROI([0, 0], [nPoints[2], nPoints[0]])
        roi3.addScaleHandle([0.5, 1], [0.5, 0.5])
        roi3.addScaleHandle([0, 0.5], [0.5, 0.5])
        plot3.addItem(roi3)

        def update_plot():
            # Get positions and FOVs
            pos1 = roi1.pos()
            fov1 = roi1.size()
            pos2 = roi2.pos()
            fov2 = roi2.size()
            pos3 = roi3.pos()
            fov3 = roi3.size()

            # Update self.pos and self.fov according to new values
            if pos1[1] != self.pos[0] or pos1[0] != self.pos[1]:  # If we change roi 1, link roi 2 and 3
                self.pos[0] = pos1[1]
                self.fov[0] = fov1[1]
                self.pos[1] = pos1[0]
                self.fov[1] = fov1[0]

                roi2.setSize(size=(self.fov[2], self.fov[0]), update=False)
                roi2.setPos(pos=(self.pos[2], self.pos[0]), update=False)
                roi3.setSize(size=(self.fov[2], self.fov[1]), update=False)
                roi3.setPos(pos=(self.pos[2], self.pos[1]), update=False)
                roi2.stateChanged(finish=False)
                roi3.stateChanged(finish=False)

            elif pos2[1] != self.pos[0] or pos2[0] != self.pos[2]:  # If we change roi 2, link roi 1 and 3
                self.pos[0] = pos2[1]
                self.fov[0] = fov2[1]
                self.pos[2] = pos2[0]
                self.fov[2] = fov2[0]

                roi1.setPos(pos=(self.pos[1], self.pos[0]), update=False)
                roi1.setSize(size=(self.fov[1], self.fov[0]), update=False)
                roi3.setPos(pos=(self.pos[2], self.pos[1]), update=False)
                roi3.setSize(size=(self.fov[2], self.fov[1]), update=False)
                roi1.stateChanged(finish=False)
                roi3.stateChanged(finish=False)

            elif pos3[1] != self.pos[1] or pos3[0] != self.pos[2]:  # If we change roi 3, link roi 2 and 1
                self.pos[1] = pos3[1]
                self.fov[1] = fov3[1]
                self.pos[2] = pos3[0]
                self.fov[2] = fov3[0]

                roi1.setPos(pos=(self.pos[1], self.pos[0]), update=False)
                roi1.setSize(size=(self.fov[1], self.fov[0]), update=False)
                roi2.setPos(pos=(self.pos[2], self.pos[0]), update=False)
                roi2.setSize(size=(self.fov[2], self.fov[0]), update=False)
                roi1.stateChanged(finish=False)
                roi2.stateChanged(finish=False)

            # Update fov of dafaultsequences
            realFov = self.fov*resolution
            realPos = self.pos*resolution
            dFov = realPos + realFov/2 - np.array(self.mapVals['fov'])/2
            realFov = np.round(realFov, decimals=1)
            dFov = np.round(dFov, decimals=1)
            localFov = copy.copy(realFov)
            localdFov = -copy.copy(dFov)
            for key in self.sequencelist:
                if ('fov' and 'dfov') in self.sequencelist[key].mapKeys:
                    self.sequencelist[key].mapVals['fov'] = copy.copy(localFov)
                    self.sequencelist[key].mapVals['dfov'] = copy.copy(localdFov)*10
            self.parent.onSequenceChanged.emit(self.parent.sequence)

        roi1.sigRegionChangeFinished.connect(update_plot)
        roi2.sigRegionChangeFinished.connect(update_plot)
        roi3.sigRegionChangeFinished.connect(update_plot)
        update_plot()

        self.saveRawData()


        if obj == 'Standalone':
            pg.exec()
        else:
            return ([win])

if __name__ == '__main__':
    seq = Localizer()
    seq.sequenceRun()
    seq.sequenceAnalysis(obj='Standalone')
