"""
Created on Tu August 8 2023
@author: T. Guallart Naval, MRILab, i3M, CSIC, Valencia
@email: tguanav@i3m.upv.es
@Summary: Petra sequence class
"""

import os
import sys
import controller.experiment_gui as ex
import configs.hw_config as hw
import configs.units as units
import seq.mriBlankSeq as blankSeq
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class PETRAphysio(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(PETRAphysio, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='PETRAphysioInfo', val='PETRAphysio')

        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.053, units=units.MHz, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=35.0, units=units.us, field='RF')
        self.addParameter(key='rfPhase', string='RF phase (º)', val=0.0, field='RF')
        self.addParameter(key='deadTime', string='Dead Time (us)', val=500., units=units.us, field='RF')

        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='fov', string='FOV[x,y,z] (cm)', val=[15.0, 15.0, 15.0], units=units.cm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM',
                          tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[10,10,10], field='IM')
        self.addParameter(key='nRdManual', string='Readout number', val=40, field='IM',
                          tip = "Use 0 for automatic allocation according to nPoints")
        self.addParameter(key='axesEnable', string='Axes enable', val=[1, 1, 1], field='IM',
                          tip="Use 0 for directions with matrix size 1, use 1 otherwise.")
        self.addParameter(key='nRadius', string='Number of radius', val=10, field='IM')
        self.addParameter(key='nSPManual', string='Number of Single Points', val=[0,0,0], field='IM',
                          tip = 'Use [0,0,0] for automatic allocation according to nPoints')

        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=50., units=units.ms, field='SEQ',
                          tip="0 to ommit this pulse")
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=2.0, units=units.ms, field='SEQ')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=5, field='SEQ',
                          tip="Use last dummy pulse to calibrate k = 0")

        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='gRiseTime', string='Gradient Rise Time (us)', val=500, units=units.us, field='OTH')
        self.addParameter(key='gradSteps', string='Gradient steps', val=16, field='OTH')
        self.addParameter(key='enableRadial', string='enableRadial', val=1, field='OTH')
        self.addParameter(key='enableSinglePoints', string='enableSinglePoints', val=1, field='OTH')
    def sequenceInfo(self):
            print(" ")
            print("PETRA sequence")
            print("Author: Teresa Guallart-Naval")
            print("Contact: tguanav@i3m.upv.es")
            print("mriLab @ i3M, CSIC, Spain \n")

    def sequenceTime(self):
        # Necessary initialisations
        fov =  np.array(self.mapVals['fov']) * 1e-2
        nPoints = self.mapVals['nPoints']
        deadTime = self.mapVals['deadTime'] * 1e-6
        acqTime = self.mapVals['acqTime'] * 1e-3
        nScans = self.mapVals['nScans']
        nRadius = self.mapVals['nRadius']
        enableRadial = self.mapVals['enableRadial']
        enableSinglePoints = self.mapVals['enableSinglePoints']
        dummyPulses = self.mapVals['dummyPulses']
        repetitionTime = self.mapVals['repetitionTime'] * 1e-3
        nSPManual = np.array(self.mapVals['nSPManual'])

        # Number of Single points calculus
        resolution = fov / nPoints
        kMaxCart = 1 / (2 * resolution)
        kMaxEll = kMaxCart * np.sqrt(2)  # minimum ellipsoid enclosing the kMaxCart prism
        kMinEll = kMaxEll * deadTime / (acqTime + deadTime)
        if nSPManual[0] == 0:
            nSP = nPoints
        else:
            nSP = nSPManual
        kSinglePoints = self.createSinglePoints(kMaxCart, nSP, kMinEll)
        nSinglePoints = len(kSinglePoints)

        print('Number of Radial Lines: ', nRadius * enableRadial)
        print('Number of Single Points: ', nSinglePoints * enableSinglePoints)
        #Time Seq calculus
        seqTime = (nScans * (nRadius * enableRadial + nSinglePoints * enableSinglePoints
                    + dummyPulses * (enableRadial + enableSinglePoints)) * repetitionTime) / 60
        return(seqTime)

    def sequenceAtributes(self):
        super().sequenceAtributes()
        self.sequenceParameters()

    def sequenceParameters(self):
        # Image characteristics
        resolution =  np.array(self.fov) / self.nPoints
        kMaxCart = 1 / (2*resolution)
        kMaxEll = kMaxCart * np.sqrt(2) # minimum ellipsoid enclosing the kMaxCart prism
        kMinEll = kMaxEll * self.deadTime / (self.acqTime+self.deadTime)
        self.mapVals['resolution'] = resolution
        self.mapTips['resolution'] = 'new'
        self.mapVals['kMaxCart'] = kMaxCart
        self.mapVals['kMaxEll'] = kMaxEll
        self.mapVals['kMinEll'] = kMinEll

        # RADIAL k-Space and corresponding gradients
        kRadialMax = self.golden3D(self.nRadius//2, kMaxEll)
        radialGradients = kRadialMax/(hw.gammaB*(self.acqTime+self.deadTime)) #T/m
        nRadialLines = len(kRadialMax)
        self.mapVals['kRadialMax'] = kRadialMax
        self.mapVals['radialGradients'] = radialGradients
        self.mapVals['nRadialLines'] = nRadialLines
        print('Number of Radial Lines: ', nRadialLines*self.enableRadial)

        # SINGLE-POINT k-Space and corresponding gradients
        if self.nSPManual[0] == 0:
            nSP = self.nPoints
        else:
            nSP = self.nSPManual
        kSinglePoints = self.createSinglePoints(kMaxCart, nSP, kMinEll)
        singlepGradients = kSinglePoints/(hw.gammaB*self.deadTime)  #T/m
        nSinglePoints = len(kSinglePoints)
        self.mapVals['kSinglePoints'] = kSinglePoints  # Need for minor time correction
        self.mapVals['singlepGradients'] = singlepGradients
        self.mapVals['nSinglePoints'] = nSinglePoints
        print('Number of Single Points: ', nSinglePoints*self.enableSinglePoints)

        # CARTESIAN k-Space - space in which regridding takes place
        kCartesianFull = self.createKCartesian(kMaxCart, self.nPoints)
        self.mapVals['kCartesianFull'] = kCartesianFull

        # Create the new inputs automatically as a property of the class
        for key in self.mapVals:
            if key not in self.mapKeys:
                setattr(self, key, self.mapVals[key])

    def sequenceTimeParameters(self):
        """"
        Time parameters are defined.
        Time inputs in us
        """
        # Parameters used to call the experiment
        BWCart = (np.array(self.nPoints) // 2) / self.acqTime
        if self.nRdManual == 0:
            BWmax = np.max(BWCart)
            nRd = self.nPoints[np.argmax(BWCart)]//2
        else:
            nRd = self.nRdManual
            BWmax  = nRd/self.acqTime
        samplingPeriod = 1 / BWmax
        self.mapVals['BWCart'] = BWCart
        self.mapVals['BWmax'] = BWmax
        self.mapVals['nRd'] = nRd
        self.mapVals['samplingPeriod'] = samplingPeriod

        # Initialisation of the experiment to correct the time for digitisation.
        # (The experiment is automatically closed).
        self.expt = ex.Experiment(lo_freq=self.larmorFreq, rx_t=samplingPeriod, init_gpa=False)
        samplingPeriod_real = self.expt.getSamplingRate()
        self.expt.__del__()
        BW_real = 1 / samplingPeriod_real
        acqTime_real = nRd / BW_real
        self.mapVals['BW_real'] = BW_real
        self.mapVals['acqTime_real'] = acqTime_real
        self.mapVals['samplingPeriod_real'] = samplingPeriod_real

        # Time vector for Radial acquisition
        timeVectorRadial = (self.deadTime*1e-6 + np.array(np.linspace(0.5, nRd+0.5, int(nRd), endpoint=False))[:, np.newaxis]/(BWmax*1e6))
        self.mapVals['timeVectorRadial'] = timeVectorRadial
        kRadialFull = []
        for ii in range(len(self.radialGradients)):
            kRadialFull_aux = []
            gf = self.radialGradients[ii, :]
            kRadialFull_aux = np.concatenate(
                (gf[0] * timeVectorRadial, gf[1] * timeVectorRadial, gf[2] * timeVectorRadial), axis=1) * hw.gammaB
            if ii == 0:
                kRadialFull = kRadialFull_aux
            else:
                kRadialFull = np.concatenate((kRadialFull, kRadialFull_aux), axis=0)
        self.mapVals['kRadialFull'] = kRadialFull
        self.mapVals['nRadialPoints'] = len(kRadialFull)

        # Time correction for Single Point acquisition
        timeSP = self.deadTime * 1e-6 + 0.5 / (BW_real * 1e6)
        self.mapVals['timeSP'] = timeSP
        kSinglePFull = []
        for ii in range(len(self.singlepGradients)):
            kRadialFull_aux = []
            gf = self.singlepGradients[ii, :][np.newaxis,:]
            kSPFull_aux = gf * timeSP * hw.gammaB
            if ii == 0:
                kSinglePFull = kSPFull_aux
            else:
                kSinglePFull = np.concatenate((kSinglePFull, kSPFull_aux), axis=0)
        self.mapVals['kSinglePFull'] = kSinglePFull

        # Create the new inputs automatically as a property of the class
        for key in self.mapVals:
            if key not in self.mapKeys:
                setattr(self, key, self.mapVals[key])

    def sequenceRun(self, plotSeq=0, demo=False):
        # Changing time parameters to us
        self.rfExTime = self.rfExTime*1e6
        self.repetitionTime = self.repetitionTime*1e6
        self.gRiseTime = self.gRiseTime * 1e6
        self.deadTime = self.deadTime * 1e6
        self.acqTime = self.acqTime * 1e6

        # Define time parameters
        self.sequenceTimeParameters()

        # Plot kSpace
        self.plot_kSpace(plot = False)
        def createSequence(gradVector, acqMode):
            # Set shimming
            self.iniSequence(20, self.shimming)

            # Initialisations
            tIni = 20e3
            tGrad = self.repetitionTime-(self.deadTime+self.acqTime)
            g0 = np.array([0,0,0])
            repeIndex = 0
            nRepetitions = (len(gradVector)+self.dummyPulses)

            ## Create sequence
            while repeIndex < nRepetitions:
                # Initialize time
                t0 = tIni+self.repetitionTime*repeIndex

                # Gradients
                if repeIndex >= self.dummyPulses:
                    gf = gradVector[repeIndex-self.dummyPulses,:]
                    self.setGradientRamp(t0, self.gRiseTime, self.gradSteps, g0[0], gf[0], 0, self.shimming)
                    self.setGradientRamp(t0, self.gRiseTime, self.gradSteps, g0[1], gf[1], 1, self.shimming)
                    self.setGradientRamp(t0, self.gRiseTime, self.gradSteps, g0[2], gf[2], 2, self.shimming)
                    g0 =gf

                # Excitation pulse
                t0 += tGrad
                tEx = t0 -hw.blkTime-self.rfExTime/2
                self.rfRecPulse(tEx, self.rfExTime, self.rfExAmp, self.rfPhase)

                # Adquisition window
                t0 += self.deadTime
                if repeIndex >= self.dummyPulses:
                    if acqMode == 'Radial':
                        self.rxGateSync(t0, self.acqTime_real)
                    if acqMode == 'SinglePoint':
                        self.rxGateSync(t0, self.samplingPeriod_real)

                repeIndex += 1

            # End sequence
            self.endSequence(t0 + self.repetitionTime)
            # Return the output variables
            return()

        # RADIAL ACQUISITION
        print('\nRunning...')
        if self.enableRadial == 1:
            self.expt = ex.Experiment(lo_freq=self.larmorFreq, rx_t=self.samplingPeriod, init_gpa=False)
            createSequence(self.radialGradients, 'Radial')
            # Checks for errors in the sequence
            if self.floDict2Exp():
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("\nERROR: sequence waveforms out of hardware bounds")
                return False
            if plotSeq == 0:
                for i in range(self.nScans):
                    rxd, msgs = self.expt.run()
                    print(i + 1, "/", self.nScans, "Radial acquisition")
                    # Get data
                    if i == 0:
                        dataOverRadial = rxd['rx0']
                    else:
                        dataOverRadial = np.concatenate((dataOverRadial, rxd['rx0']), axis=0)
                dataOverRadial = np.reshape(dataOverRadial, (self.nScans, -1))
                dataRadial = self.decimate(dataOverRadial, self.nScans * self.nRadialLines)
                dataAvgRadial = np.average(np.reshape(dataRadial, (self.nScans, -1)),axis=0)[:,np.newaxis]
                kRadial4D = np.concatenate((self.kRadialFull, dataAvgRadial), axis=1)
                self.mapVals['dataRadial'] = dataRadial
                self.mapVals['dataAvgRadial'] = dataAvgRadial
                self.mapVals['kRadial4D'] = kRadial4D
                kAcquired4D = kRadial4D
            elif plotSeq == 1:
                self.expt.plot_sequence()
                plt.show()
            self.expt.__del__()


        # SINGLE POINT ACQUISITION
        if self.enableSinglePoints == 1:
            self.expt = ex.Experiment(lo_freq=self.larmorFreq, rx_t=self.samplingPeriod, init_gpa=False)
            createSequence(self.singlepGradients, 'SinglePoint')
            # Checks for errors in the sequence
            if self.floDict2Exp():
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("\nERROR: sequence waveforms out of hardware bounds")
                return False
            if plotSeq == 0:
                for i in range(self.nScans):
                    rxd, msgs = self.expt.run()
                    print(i + 1, "/", self.nScans, "Single acquisition")
                    # Get data
                    if i == 0:
                        dataOverSingleP = rxd['rx0']
                    else:
                        dataOverSingleP = np.concatenate((dataOverSingleP, rxd['rx0']), axis=0)
                dataOverSingleP = np.reshape(dataOverSingleP, (self.nScans, -1))
                dataSingleP = self.decimate(dataOverSingleP, self.nScans * self.nSinglePoints)
                dataAvgSingleP = np.average(np.reshape(dataSingleP, (self.nScans, -1)), axis=0)[:, np.newaxis]
                kSingleP4D = np.concatenate((self.kSinglePFull, dataAvgSingleP), axis=1)
                self.mapVals['dataRadial'] = dataSingleP
                self.mapVals['dataAvgRadial'] = dataAvgSingleP
                self.mapVals['kRadial4D'] = kSingleP4D
                kAcquired4D = kSingleP4D
            elif plotSeq == 1:
                self.expt.plot_sequence()
                plt.show()
            self.expt.__del__()

        if self.enableRadial == 1 and self.enableSinglePoints == 1:
            kAcquired4D = np.concatenate((kRadial4D, kSingleP4D), axis=0)
        self.mapVals['kAcquired4D'] =  kAcquired4D
        print('Acquisition completed.')


        return True

    def sequenceAnalysis(self, mode=None):
        self.mode = mode

        # REGRIDDING
        print('\nRegridding...')
        self.kAcquired4D = self.mapVals['kAcquired4D']
        kSpaceInterReal = griddata(self.kAcquired4D[:, 0:3], np.real(self.kAcquired4D[:, 3]), self.kCartesianFull,
                                   method='linear', fill_value=0, rescale=False)
        kSpaceInterImag = griddata(self.kAcquired4D[:, 0:3], np.imag(self.kAcquired4D[:, 3]), self.kCartesianFull,
                                   method='linear', fill_value=0, rescale=False)
        kSpace3D = kSpaceInterReal + 1j * kSpaceInterImag
        kSpace3D = np.reshape(kSpace3D, (self.nPoints[0], self.nPoints[1], self.nPoints[2]))
        ImageFFT = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace3D)))
        self.mapVals['kSpace3D'] = kSpace3D
        self.mapVals['ImageFFT'] = ImageFFT
        print('Regridding completed.')

        # # REGRIDDING - Test -
        # # Probar CloughTocher2DInterpolator en vez de LinearNDInterpolator que me da problema con los complejos

        # if self.enableRadial == 1 and self.enableSinglePoints == 1:
        #     kAcquired4D = np.concatenate((kRadial4D, kSingleP4D), axis=0)
        # interpolator = LinearNDInterpolator(kAcquired4D[:,0:2], kAcquired4D[:,3])
        # test1 = kAcquired4D[:,0:3]
        # test2 = kAcquired4D[:,3]
        # kSpaceInter = interpolator(self.kCartesianFull)
        # kSpace3D = np.reshape(kSpaceInter, (self.nPoints[0], self.nPoints[1], self.nPoints[2]))
        # ImageFFT = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace3D)))

        # Create representations
        result1 = {}

        result1['widget'] = 'image'
        result1['data'] = np.abs(self.mapVals['ImageFFT'])
        result1['xLabel'] = " "
        result1['yLabel'] = " "
        result1['title'] = "Image"
        result1['row'] = 0
        result1['col'] = 0

        result2 = {}
        kSpace = np.abs(self.mapVals['kSpace3D'])
        result2['widget'] = 'image'
        result2['data'] = kSpace
        result2['xLabel'] = " "
        result2['yLabel'] = " "
        result2['title'] = "k-Space"
        result2['row'] = 0
        result2['col'] = 1

        self.output = [result1, result2]

        # Save results
        self.saveRawData()

        if mode == 'Standalone':
            self.plotResults()

        return self.output

    def golden3D(self,nRadius,radiusFactor):
        """"
        Calculate the distributed radii according to golden means 3d - see paper
        Returns the maximum points of each radius to be mapped.
        """
        n_points = np.linspace(1,nRadius,nRadius, endpoint=True)
        phi1 = 0.4656
        phi2 = 0.6823
        alpha = 2*np.pi*((n_points * phi2) % 1)
        beta = np.arccos(((n_points * phi1) % 1))

        kx = radiusFactor[0] * np.sin(beta) * np.cos(alpha)
        ky = radiusFactor[1] * np.sin(beta) * np.sin(alpha)
        kz = radiusFactor[2] * np.cos(beta)

        kRadius = np.array([kx, ky, kz])
        kRadius = np.concatenate((kRadius.T,-kRadius.T), axis=0)

        return kRadius

    def createSinglePoints(self, kMax, nPoints, kMinEll):
        x = np.linspace(-kMax[0], kMax[0], nPoints[0])
        y = np.linspace(-kMax[1], kMax[1], nPoints[1])
        z = np.linspace(-kMax[2], kMax[2], nPoints[2])
        xv, yv, zv = np.meshgrid(x, y, z)
        mask = (xv ** 2 / (kMinEll[0]) ** 2) + (yv ** 2 / (kMinEll[1]) ** 2) + (zv ** 2 / (kMinEll[2]) ** 2) <= 1
        singlePoints = np.column_stack((xv[mask], yv[mask], zv[mask]))
        return singlePoints

    def createKCartesian(self, kMax, nPoints):
        x = np.linspace(-kMax[0], kMax[0], nPoints[0])
        y = np.linspace(-kMax[1], kMax[1], nPoints[1])
        z = np.linspace(-kMax[2], kMax[2], nPoints[2])
        xv, yv, zv = np.meshgrid(x, y, z)
        kCartesian = np.concatenate((xv.flatten()[:, np.newaxis],
                                     yv.flatten()[:, np.newaxis],
                                     zv.flatten()[:, np.newaxis]),
                                     axis=1)
        return kCartesian

    def plot_kSpace(self, plot = False):
            def plot_ellipsoid(radii, col = 'green'):
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 50)
                x = radii[0] * np.outer(np.cos(u), np.sin(v))
                y = radii[1] * np.outer(np.sin(u), np.sin(v))
                z = radii[2] * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color=col, alpha=0.2)
            def plot_prism(width, height, depth, col = 'green'):
                half_width = width / 2
                half_height = height / 2
                half_depth = depth / 2
                vertices = [
                    [-half_width, -half_height, -half_depth],
                    [half_width, -half_height, -half_depth],
                    [half_width, half_height, -half_depth],
                    [-half_width, half_height, -half_depth],
                    [-half_width, -half_height, half_depth],
                    [half_width, -half_height, half_depth],
                    [half_width, half_height, half_depth],
                    [-half_width, half_height, half_depth]
                ]
                caras = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],
                    [vertices[4], vertices[5], vertices[6], vertices[7]],
                    [vertices[0], vertices[1], vertices[5], vertices[4]],
                    [vertices[2], vertices[3], vertices[7], vertices[6]],
                    [vertices[0], vertices[3], vertices[7], vertices[4]],
                    [vertices[1], vertices[2], vertices[6], vertices[5]]
                ]
                cara = Poly3DCollection(caras, alpha=0.1, facecolors=col)
                ax.add_collection3d(cara)

            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # Radial
                ax.scatter(self.kRadialMax[:,0], self.kRadialMax[:,1], self.kRadialMax[:,2], c='blue', s=5)
                ax.scatter(self.kRadialFull[:, 0], self.kRadialFull[:, 1], self.kRadialFull[:, 2], c='blue', s=5)
                for i in range(len(self.kRadialMax[:,0])):
                    ax.plot([0, self.kRadialMax[i, 0]], [0, self.kRadialMax[i, 1]], [0, self.kRadialMax[i, 2]], c='red')
                plot_ellipsoid(self.kMaxEll, col = 'green')
                # Single Point
                ax.scatter(self.kSinglePFull[:, 0], self.kSinglePFull[:, 1], self.kSinglePFull[:, 2], c='blue', s=6)
                plot_ellipsoid(self.kMinEll, col = 'green')
                # Cartesian
                ax.scatter(self.kCartesianFull[:, 0], self.kCartesianFull[:, 1], self.kCartesianFull[:, 2], c='blue', s=0.05,alpha=0.1)
                plot_prism(2*self.kMaxCart[0], 2*self.kMaxCart[1], 2*self.kMaxCart[2], col = 'blue')

                ax.set_xlabel('kx')
                ax.set_ylabel('ky')
                ax.set_zlabel('kz')
                plt.show()

if __name__=="__main__":
    seq = PETRAphysio()
    seqTime = seq.sequenceTime()
    print('Sequence time (min):', seqTime)
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq = 0)
    seq.plot_kSpace(False)
    seq.sequenceAnalysis(mode='Standalone')
