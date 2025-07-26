"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: This class is able to sweep any parameter from any sequence
"""

import os
import sys
#*****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory = os.path.dirname(parent_directory)

# Define the subdirectories you want to add to sys.path
subdirs = ['MaRGE', 'marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
#******************************************************************************
import numpy as np
import marge.seq.mriBlankSeq as blankSeq

class SweepImage(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(SweepImage, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='SWEEPinfo', val='SWEEP')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='seqNameSweep', string='Sequence', val='Noise', field='OTH')
        self.addParameter(key='parameter0', string='Parameter 0 X-axis', val='bw', field='OTH')
        self.addParameter(key='start0', string='Start point 0', val=30.0, field='OTH')
        self.addParameter(key='end0', string='End point 0', val=50.0, field='OTH')
        self.addParameter(key='logScale', string='Log scale', val=0, field='OTH')
        self.addParameter(key='nSteps0', string='Number of steps 0', val=5, field='OTH')
        self.addParameter(key='parameter1', string='Parameter 1 Y-axis', val='larmorFreq', field='OTH')
        self.addParameter(key='start1', string='Start point 1', val=3.0, field='OTH')
        self.addParameter(key='end1', string='End point 1', val=4.0, field='OTH')
        self.addParameter(key='nSteps1', string='Number of steps 1', val=5, field='OTH')

    def sequenceInfo(self):
        
        print("Genera sweep sequence")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain\n")

    def sequenceTime(self):
        return(0)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=True):
        # Inputs
        seqName = self.mapVals['seqNameSweep']
        parameters = [self.mapVals['parameter0'], self.mapVals['parameter1']]
        start = [self.mapVals['start0'], self.mapVals['start1']]
        end = [self.mapVals['end0'], self.mapVals['end1']]
        nSteps = [self.mapVals['nSteps0'], self.mapVals['nSteps1']]

        # Sweep
        sampled = []
        parVector0 = np.linspace(start[0], end[0], nSteps[0]) # Create vector with parameters to sweep
        if self.mapVals['logScale'] == 1:
            parVector0 = np.geomspace(start[0], end[0], nSteps[0])
        parVector1 = np.linspace(start[1], end[1], nSteps[1])
        seq = self.sequence_list[seqName] # Select the sequence that we want to sweep with modified parameters
        parMatrix = np.zeros((nSteps[0]*nSteps[1], 2))
        n = 0
        for step0 in range(nSteps[0]):
            for step1 in range(nSteps[1]):
                parMatrix[n, 0] = parVector0[step0]
                parMatrix[n, 1] = parVector1[step1]
                seq.mapVals[parameters[0]] = parVector0[step0]
                seq.mapVals[parameters[1]] = parVector1[step1]
                seq.sequenceAtributes()
                seq.sequenceRun(plotSeq=0, demo=demo)
                seq.sequenceAnalysis()
                if 'sampledCartesian' in seq.mapVals:
                    sampled.append(seq.mapVals['sampledCartesian']) # sampledCartesian is four column kx, ky, kz and S(kx, ky, kz)
                elif 'sampledPoint' in seq.mapVals:
                    sampled.append(seq.mapVals['sampledPoint'])
                else:
                    print('No signal to plot')
                    return 0
                n += 1

        self.seq = seq
        self.sampled = sampled

        return True

    def sequenceAnalysis(self, obj=''):
        nSteps = [self.mapVals['nSteps0'], self.mapVals['nSteps1']]
        start = [self.mapVals['start0'], self.mapVals['start1']]
        end = [self.mapVals['end0'], self.mapVals['end1']]
        parVector0 = np.linspace(start[0], end[0], nSteps[0])  # Create vector with parameters to sweep


        if 'sampledCartesian' in self.seq.mapVals:    # In case of images
            # Initialize data and image variables as zeros
            nPoints = np.array(self.seq.mapVals['nPoints'])
            dataSteps = np.zeros((nSteps[0] * nSteps[1], nPoints[1], nPoints[0]), dtype=complex)
            imageSteps = dataSteps.copy()

            # Get axes in strings
            axes = self.seq.mapVals['axesOrientation']
            axesDict = {'x': 0, 'y': 1, 'z': 2}
            axesKeys = list(axesDict.keys())
            axesVals = list(axesDict.values())
            axesStr = ['', '', '']
            n = 0
            for val in axes:
                index = axesVals.index(val)
                axesStr[n] = axesKeys[index]
                n += 1

            # Generate k-space maps and images
            for step in range(nSteps[0]*nSteps[1]):
                data = self.sampled[step][:, 3]
                data = np.reshape(data, (nPoints[2], nPoints[1], nPoints[0]))
                image = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
                dataSteps[step, :, :] = data[int(nPoints[2]/2), :, :]
                imageSteps[step, :, :] = image[int(nPoints[2]/2), :, :]

            # Plot image
            image = np.abs(imageSteps)
            image = image / np.max(np.reshape(image, -1)) * 100
            result1 = {'widget': 'image',
                       'data': image,
                       'xLabel': axesStr[0],
                       'yLabel': axesStr[1],
                       'title': "%s sweep images" % self.mapVals['seqNameSweep'],
                       'row': 0,
                       'col': 0}

            # Plot k-space
            kSpace = np.log10(np.abs(dataSteps))
            kSpace = kSpace / np.max(np.reshape(kSpace, -1)) * 100
            result2 = {'widget': 'image',
                       'data': kSpace,
                       'xLabel': axesStr[0],
                       'yLabel': axesStr[1],
                       'title': "%s sweep k-spaces" % self.mapVals['seqNameSweep'],
                       'row': 0,
                       'col': 1}

            self.output = [result1, result2]

            self.saveRawData()

        elif 'sampledPoint' in self.seq.mapVals:  # In case of points (calibration sequences)
            image = np.zeros((1, nSteps[0], nSteps[1]), dtype=complex)
            n = 0
            for step0 in range(nSteps[0]):
                for step1 in range(nSteps[1]):
                    image[0, step0, step1] = self.sampled[n]
                    n += 1

            # Plot image
            if nSteps[1]>1: # If we sweep two parameters, show a map
                image = np.abs(image)
                map = image = image / np.max(np.reshape(image, -1)) * 100
                result1 = {'widget': 'image',
                           'data': np.abs(image),
                           'xLabel': self.seq.mapNmspc[self.mapVals['parameter0']],
                           'yLabel': self.seq.mapNmspc[self.mapVals['parameter1']],
                           'title': '%s sweep' % self.mapVals['seqNameSweep'],
                           'row': 0,
                           'col': 0}
            else: # If we sweep only one parameter, show a line plot
                image = np.reshape(image, -1)
                result1 = {'widget': 'curve',
                           'xData': parVector0,
                           'yData': [np.abs(image)],
                           'xLabel': self.seq.mapNmspc[self.mapVals['parameter0']],
                           'yLabel': 'Output amplitude',
                           'title': '%s sweep' % self.mapVals['seqNameSweep'],
                           'legend': [''],
                           'row': 0,
                           'col': 0}
                self.mapVals['sweepResult'] = [parVector0, np.abs(image)]

            self.output = [result1]

            self.saveRawData()

        return self.output

if __name__=='__main__':
    seq = SweepImage()
    seq.sequenceRun()
    seq.sequenceAnalysis()

