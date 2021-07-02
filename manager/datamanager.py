"""
Data Manager

@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   1.0
@change:    02/05/2020

@summary:   Class for managing the data procession of acquired data.
            Processes data in time (t_) and frequency (f_) domain.

@status:    Under testing
@todo:      Return snr in dB too

"""

from PyQt5.QtCore import QObject, pyqtSignal
from dataclasses import dataclass
import numpy as np


# just for debugging calculations:
# import matplotlib.pyplot as plt
timePerSample = 4e-3


@dataclass(repr=False, eq=False)
class DataManager(QObject):
    """
    Data manager class
    """

    # Init signal that's emitted when readout is processed
    t1_finished = pyqtSignal()
    t2_finished = pyqtSignal()
    uploaded = pyqtSignal(bool)

    __slots__ = ['_t_magnitude',
                 '_t_real',
                 '_t_imag',
                 '_t_axis',
                 '_k_space'
                 '_t_magnitudeCon',
                 '_t_realCon',
                 '_f_axis',
                 '_f_fftData',
                 '_f_fftMagnitude', 
                 '_f_fft2Data', 
                 '_f_fft2DMagnitude']

    def __init__(self, data: np.complex, p_frequency: float, samples: int, n:list, bandWidth: float, ):
        """
        Initialisation of data manager class
        @param data:        Raw data
        @param p_ts:        Sample time (property)
        @param bandWidth:     Spectral frequency / acquisition bandwidth
        """
        super(DataManager, self).__init__()
        self.data = data
        self.bandWidth = bandWidth
        self.samples = samples
        self.p_ts = self.samples/self.bandWidth
 
        d_cropped = self.data[0:self.samples]  # * 2000.0
        self._t_axis = np.linspace(-self.p_ts/2, self.p_ts/2, self.samples)
        self._t_magnitude = np.abs(d_cropped)
        self._t_magnitudeCon = np.convolve(self.t_magnitude, np.ones((50,)) / 50, mode='same')
        self._t_real = np.real(d_cropped)
        self._t_realCon = np.convolve(self.t_real, np.ones((50,)) / 50, mode='same')
        self._t_imag = np.imag(d_cropped)

        self._frequency = p_frequency
        self._f_axis = np.linspace(-self.bandWidth / 2, self.bandWidth / 2, self.samples)
        self._f_fftData = np.fft.fftshift(np.fft.fft(np.fft.fftshift(d_cropped), n=self.samples))
        self._f_fftMagnitude = abs(self.f_fftData)
        
        if(n != [] and (n[1]>1 or n[2]>1)):
            self._data_kS = np.reshape(self.data, (n[2], n[1], n[0]))
            self._f_fft2Data = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(self._data_kS)))
            self._f_fft2Magnitude = abs(self._f_fft2Data)
        
        # self._dataTimestamp = datetime.now().strftime('%m/%d/%Y, %H:%M:%S')

    @property
    def t_axis(self):
        return self._t_axis

    @property
    def t_magnitude(self):
        return self._t_magnitude
        
    @property
    def k_space(self):
        return self._data_kS

    @property
    def t_magnitudeCon(self):
        return self._t_magnitudeCon

    @property
    def t_real(self):
        return self._t_real

    @property
    def t_realCon(self):
        return self._t_realCon

    @property
    def t_imag(self):
        return self._t_imag

    @property
    def f_axis(self):
        return self._f_axis

    @property
    def f_fftData(self):
        return self._f_fftData

    @property
    def f_fftMagnitude(self):
        return self._f_fftMagnitude
        
    @property
    def f_fft2Data(self):
        return self._f_fft2Data
    
    @property
    def f_fft2Magnitude(self):
        return self._f_fft2Magnitude

    # TODO: Implementation of params-setter (?)
    def get_fwhm(self, f_fwhmWindow: int = 1000) -> [int, float, float]:
        """
        Get full width at half maximum
        @param f_fwhmWindow:    Frequency window
        @return:                FWHM in datapoint indices, hertz and ppm
        """
#        if not self.is_evaluateable():
#            return [0, float("nan"), float("nan")]

        [_peakValue, _, _peakIdx, _peakFreq] = self.get_peakparameters()
        fft = self.f_fftMagnitude[int(_peakIdx - f_fwhmWindow / 2):int(_peakIdx + f_fwhmWindow / 2)]
        candidates: np.ndarray = np.abs([x - _peakValue / 2 for x in fft])
        # Calculate index difference by find indices of minima, calculate fwhm in Hz thereafter
        _winC = int(f_fwhmWindow / 2)
        _fwhm: int = np.argmin(candidates[_winC:-1]) + _winC - np.argmin(candidates[0:_winC])
        _fwhm_hz: float = _fwhm * (abs(np.min(self._f_axis)) + abs(np.max(self._f_axis))) / self.samples
        _fwhm_ppm: float = _fwhm_hz / _peakFreq

        return [_fwhm, _fwhm_hz, _fwhm_ppm]

    def get_snr(self, f_windowfactor: float = 10) -> float:
        """
        Get signal to noise ratio
        @param f_windowfactor:  Factor for fwhm to define peak window
        @param n:               N datapoints for moving average
        @return:                SNR
        """
#        if not self.is_evaluateable():
#            return float("nan")

        [_fwhm, _, _] = self.get_fwhm()
        [_signalValue, _, _signalIdx, _] = self.get_peakparameters()
        _peakWin = int(_fwhm * f_windowfactor)
        _winC = int(len(self._f_fftData) / 2)
        _noiseBorder = int(len(self._f_fftData) * 0.05)

        _noiseFloor = np.concatenate((self._f_fftData[_noiseBorder:int(_winC - _peakWin / 2)],
                                      self._f_fftData[int(_winC + _peakWin / 2):-1 - _noiseBorder]))

        _noise = np.std(_noiseFloor / _signalValue)
        _snr = round(1 / _noise)

        return _snr  # TODO: Add return in dB

    def get_peakparameters(self) -> [float, float, int, float]:
        """
        Get peak parameters
        @return:            Frequency peak, time domain peak, index of frequency peak and frequency of peak
        """
#        if not self.is_evaluateable():
#            return [float("nan"), float("nan"), 0, float("nan")]

        t_signalValue: float = round(np.max(self._t_magnitudeCon), 4)
        f_signalValue: float = round(np.max(self._f_fftMagnitude), 4)
        f_signalIdx: int = np.argmax(self._f_fftMagnitude)  # [0]
        f_signalFrequency: float = round(self._frequency + ((f_signalIdx - self.samples / 2)
                                                            * self.bandWidth / self.samples) / 1.0e6, 6)

        return [f_signalValue, t_signalValue, f_signalIdx, f_signalFrequency]

    @property
    def get_sign(self) -> int:
        """
        Get sign of real part signal in time domain
        @return:    Sign
        """
        index: np.ndarray = np.argmin(self._t_realCon[0:50])
        return np.sign(self._t_realCon[index])

    def is_evaluateable(self) -> bool:
        """
        Check if acquired data is evaluateable
        @return:    Evaluateable (true/false)
        """
        minValue = min(self._f_fftMagnitude)
        maxValue = max(self._f_fftMagnitude)
        difference = maxValue - minValue
        if difference > 1:
            return True
        else:
            return False
