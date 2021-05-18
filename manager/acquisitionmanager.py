"""
Acquisition Manager

@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   1.0
@change:    02/05/2020

@summary:   Manages acquisitions, purpose of this class is to perform the right processing for any operation

@status:    Most of the methods in here are obsolete, divide work between acquisition controller and manager
@todo:      Cleanup

"""

from PyQt5.QtCore import pyqtSignal
#from server.communicationmanager import CommunicationManager as Com
#from manager.sequencemanager import SqncMngr
from manager.datamanager import DataManager as Data
from globalvars import grads, sqncs
from sequencesnamespace import Namespace as nmspc
import numpy as np
import time
import csv
from plotview.exampleplot import ExamplePlot
from plotview.spectrumplot import SpectrumPlot


class AcquisitionManager:
    """
    Acquisition manager class
    """
    readoutFinished = pyqtSignal()

    def __init__(self, p_samples: int = 50000):
        """
        Initialization of AcquisitionManager class
        @type p_samples: Amount of samples
        """
        super(AcquisitionManager, self).__init__()
        self._samples: int = p_samples

        # TODO: Set sequence here (once!)

#    def get_exampleFidData(self, properties) -> [dict, ExamplePlot]:
#        """
#        Get prototype data set (FID spectrum) with f = 20.0971, at = 10, ts = 7.5
#        @param properties:  Operation properties object
#        @return:            Dict with output parameters, plot object
#        """
#        with open('exampledata.csv', 'r') as _csvfile:
#            _csvread = csv.reader(_csvfile, delimiter='\n')
#            _csvdata = list(_csvread)
#
#        cpxData = [complex(_row[0]) for _row in _csvdata]
#
#        dataobject = Data(cpxData,
#                          properties[nmspc.frequency][0],
#                          properties[nmspc.sampletime][0])
#        outputvalues = self.getOutputParameterObject(dataobject, properties)
#
#        # plot = ExamplePlot(dataobject.f_axis, dataobject.f_fftMagnitude, "frequency", "signal intensity")
#        plot = SpectrumPlot(dataobject.f_axis, dataobject.f_fftMagnitude, "frequency", "signal intensity")
#
#        return [outputvalues, plot, dataobject]
#
#    def get_spectrum(self, properties, shim) -> [dict, ExamplePlot]:
#        """
#        Get the spectrum data of the sample volume
#        @param properties:  Operation properties object
#        @param shim:        Operation shim object
#        @return:            Dict with output parameters, plot object
#        """
#        t0: float = time.time()
#        self.set_systemproperties(properties[nmspc.lo_freq],
#                                  properties[nmspc.rf_amp],
#                                  shim)
##        SqncMngr.packSequence(sqncs.FID)
##        Com.acquireSpectrum()
##        Com.waitForTransmission()
##        tmp_data: np.complex64 = Com.readAcquisitionData(self._samples)
#
#        t1: float = time.time()
#        acquisitiontime: float = (t1-t0)/60
#
#        freq_range = 50000
#        dataobject: Data = Data(tmp_data,
#                                properties[nmspc.lo_freq],
#                                properties[nmspc.sampletime],
#                                freq_range)
#
#        plot = ExamplePlot(dataobject.f_axis, dataobject.f_fftMagnitude, "frequency", "signal intensity")
#        outputvalues = self.getOutputParameterObject(dataobject, properties, acquisitiontime)
#
#        return [outputvalues, plot]
#
#    def get_kspace(self, p_frequency: float, p_npe: int = 16, p_tr: int = 4000) -> [np.complex, float]:
#        """
#        Get 2D k-space of sample volume (no slice selection)
#        @param p_frequency  Acquisition frequency (parameter)
#        @param p_npe:       Number of phase encoding steps (parameter)
#        @param p_tr:        Repetition time in ms (parameter)
#        @return:            Raw data in 2D array, acquisition time
#        TODO:   Rework function
#        """
#        tmp_data: np.ndarray = np.array(np.zeros(p_npe, self._samples), ndmin=2, dtype=np.complex64)
#
#        t0: float = time.time()
#        Com.acquireImage(p_npe, p_tr)
#        Com.waitForTransmission()
#
#        for n in range(p_npe):
#            tmp_data[n, :] = Com.readAcquisitionData(self._samples)
#            self.readoutFinished.emit()
#
#        t1: float = time.time()
#        acquisitiontime: float = (t1 - t0) / 60
#
#        print('Finished image manager in {:.4f} min'.format((t1 - t0) / 60))
#
#        return [tmp_data, acquisitiontime]
#
#    # Function to acquire 1D projection
#    def get_projection(self, p_axis: int, p_frequency: float) -> [np.complex, float, int]:
#        """
#        Get 1D projection along a dedicated axis
#        @param p_axis:      Axis (parameter)
#        @param p_frequency: Acquisition frequency (parameter)
#        @return:            1D raw data, acquisition time
#        TODO:   Rework function
#        """
#        t0: float = time.time()
#        Com.acquireProjection(p_axis)
#        Com.waitForTransmission()
#
#        tmp_data: np.complex64 = Com.readAcquisitionData(self._samples)
#
#        t1: float = time.time()
#        acquisitiontime: float = (t1 - t0) / 60
#        self.readoutFinished.emit()
#
#        # TODO: Return datahandler object for projection
#
#        return [tmp_data, acquisitiontime, p_axis]
#
#    def reaquireFrequency(self, frequency):
#        """
#        Reaquire with a different frequency (e.g. focus frequency)
#        @param frequency:   Frequency to be set
#        @return:            Output values, plot
#        TODO:   Rework function
#        """
#        freq_range = 50000
#        sampletime = 10
#        """
#        Com.setFrequency(frequency)
#        Com.waitForTransmission()
#        Com.acquireSpectrum()
#        Com.waitForTransmission()
#        tmp_data: np.complex64 = Com.readAcquisitionData(self._samples)
#        dataobject: Data = Data(tmp_data, frequency, sampletime, freq_range)
#        plot = ExamplePlot(dataobject.f_axis, dataobject.f_fftMagnitude, "frequency", "signal intensity")
#        outputvalues = self.getOutputParameterObject(dataobject)
#
#        return [outputvalues, plot, dataobject]
#        """
#        print("Reaquire spectrum at {} MHz".format(frequency))
#
#    @staticmethod
#    def set_systemproperties(p_frequency: float, p_attenuation: float, p_gradients: list) -> None:
#        """
#        Setup system parameters
#        @param p_frequency:     TX frequency
#        @param p_attenuation:   TX Attenuation
#        @param p_gradients:     Gradient offset values
#        @return:                None
#        """
#        Com.setFrequency(p_frequency)
#        Com.waitForTransmission()
#        Com.setAttenuation(p_attenuation)
#        Com.waitForTransmission()
#        Com.setGradients(p_gradients[nmspc.x_grad],
#                         p_gradients[nmspc.y_grad],
#                         p_gradients[nmspc.z_grad])
#        Com.waitForTransmission()

    @staticmethod
    def getOutputParameterObject(dataobject=None, properties=None, acquisitiontime=None) -> dict:
        """
        Function to create a dictionary of output parameters
        @param dataobject:          Dataobject from DataManager()
        @param properties:          Acquisition properties from operation
        @param acquisitiontime:     Measured acquisition time
        @return:                    Dict with output parameters
        """
        outputvalues: dict = {}
        if dataobject is not None:
            outputvalues["SNR"] = round(dataobject.get_snr(), 4)
            outputvalues["FWHM [Hz]"] = round(dataobject.get_fwhm()[1], 4)
            outputvalues["FWHM [ppm]"] = round(dataobject.get_fwhm()[2], 4)
            outputvalues["Center Frequency [MHz]"] = round(dataobject.get_peakparameters()[1], 4)
            outputvalues["Signal Maximum [V]"] = round(dataobject.get_peakparameters()[3], 4)
        # if properties is not None:
            # outputvalues["Sample Time [ms]"] = round(properties[nmspc.sampletime][0], 4)
            # outputvalues["Attenuation"] = round(properties[nmspc.attenuation][0], 4)
        if acquisitiontime is not None:
            outputvalues["Acquisition Time [s]"] = round(acquisitiontime, 4)

        return outputvalues
