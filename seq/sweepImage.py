"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: mri blank sequence with common methods that will be inherited by any sequence
"""

import numpy as np
import seq.mriBlankSeq as blankSeq
from plotview.spectrumplot import SpectrumPlot # To plot nice 1d images
from PyQt5.QtWidgets import QLabel  # To set the figure title
from PyQt5 import QtCore            # To set the figure title
import pyqtgraph as pg              # To plot nice 3d images

class SweepImage(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(SweepImage, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='SWEEPinfo', val='SWEEP')
        self.addParameter(key='seqNameSweep', string='Sequence', val='RARE', field='OTH')
        self.addParameter(key='parameter0', string='Parameter 0 X-axis', val='rfExTime', field='OTH')
        self.addParameter(key='start0', string='Start point 0', val=0.01, field='OTH')
        self.addParameter(key='end0', string='End point 0', val=50.0, field='OTH')
        self.addParameter(key='nSteps0', string='Number of steps 0', val=2, field='OTH')
        self.addParameter(key='parameter1', string='Parameter 1 Y-axis', val='rfExAmp', field='OTH')
        self.addParameter(key='start1', string='Start point 1', val=0.0, field='OTH')
        self.addParameter(key='end1', string='End point 1', val=0.3, field='OTH')
        self.addParameter(key='nSteps1', string='Number of steps 1', val=2, field='OTH')

    def sequenceInfo(self):
        print(" ")
        print("Genera sweep sequence")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")


    def sequenceTime(self):
        return(0)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, defaultsequences=''):
        # Inputs
        seqName = self.mapVals['seqNameSweep']
        parameters = [self.mapVals['parameter0'], self.mapVals['parameter1']]
        start = [self.mapVals['start0'], self.mapVals['start1']]
        end = [self.mapVals['end0'], self.mapVals['end1']]
        nSteps = [self.mapVals['nSteps0'], self.mapVals['nSteps1']]

        # Sweep
        sampled = []
        parVector0 = np.linspace(start[0], end[0], nSteps[0]) # Create vector with parameters to sweep
        parVector1 = np.linspace(start[1], end[1], nSteps[1])
        seq = defaultsequences[seqName] # Select the sequence that we want to sweep
        parMatrix = np.zeros((nSteps[0]*nSteps[1], 2))
        n = 0
        for step0 in range(nSteps[0]):
            for step1 in range(nSteps[1]):
                parMatrix[n, 0] = parVector0[step0]
                parMatrix[n, 1] = parVector1[step1]
                seq.mapVals[parameters[0]] = parVector0[step0]
                seq.mapVals[parameters[1]] = parVector1[step1]
                seq.sequenceRun(0)
                seq.sequenceAnalysis()
                if 'sampledCartesian' in seq.mapVals:
                    sampled.append(seq.mapVals['sampledCartesian'])
                    self.kind = 'Image'
                elif 'sampledSignal' in seq.mapVals:
                    sampled.append(seq.mapVals['sampledSignal'])
                    self.kind = 'Point'
                else:
                    print('No signal to plot')
                    return 0
                n += 1

        self.seq = seq
        self.sampled = sampled
        return 0

    def sequenceAnalysis(self, obj):
        nPoints = np.array(self.seq.mapVals['nPoints'])
        nSteps = [self.mapVals['nSteps0'], self.mapVals['nSteps1']]

        self.saveRawData()

        if self.kind == 'Image':    # In case of images
            # Initialize data and image variables as zeros
            dataSteps = np.zeros((nSteps[0] * nSteps[1], nPoints[1], nPoints[0]), dtype=complex)
            imageSteps = dataSteps

            # Generate k-space maps and images
            for step in range(nSteps[0]*nSteps[1]):
                data = self.sampled[step][:, 3]
                data = np.reshape(data, (nPoints[2], nPoints[1], nPoints[0]))
                image = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
                dataSteps[step, :, :] = data[int(nPoints[2]/2), :, :]
                imageSteps[step, :, :] = image[int(nPoints[2]/2), :, :]

            # Create label with rawdata name
            obj.label = QLabel('Sweep ' + self.mapVals['fileName'])
            obj.label.setAlignment(QtCore.Qt.AlignCenter)
            obj.label.setStyleSheet("background-color: black;color: white")
            obj.parent.plotview_layout.addWidget(obj.label)

            # Plot image
            obj.parent.plotview_layout.addWidget(pg.image(np.abs(imageSteps)))

            # Plot k-space
            obj.parent.plotview_layout.addWidget(pg.image(np.log10(np.abs(dataSteps))))

        elif self.kind == 'Point':  # In case of points (calibration sequences)
            image = np.zeros((nSteps[0], nSteps[1]), dtype=complex)
            n = 0
            for step0 in range(nSteps[0]):
                for step1 in range(nSteps[1]):
                    image[step0, step1] = self.sampled[n][0]
                    n += 1

            # Create label with rawdata name
            obj.label = QLabel('Sweep ' + self.mapVals['fileName'])
            obj.label.setAlignment(QtCore.Qt.AlignCenter)
            obj.label.setStyleSheet("background-color: black;color: white")
            obj.parent.plotview_layout.addWidget(obj.label)

            # Plot image
            obj.parent.plotview_layout.addWidget(pg.image(np.abs(image)))


defaultSweep = {
    'SWEEP': SweepImage(),
}


