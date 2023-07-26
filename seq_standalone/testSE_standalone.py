"""
@author: T. Guallart Naval
MRILAB @ I3M
"""
import os
import sys
#*****************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt

def testSE_standalone(
        larmorFreq = 3.0495,
        rfExAmp = 0.3,
        rfExTime = 40,
        rfReAmp = 0.6,
        rfReTime = 40.0,
        phaseRe = 0.0,

        echoSpacing = 20,
        nPoints = 200,
        acqTime = 4.0,
        repetitionTime = 50,
        nRepetitions  = 10,
        nScans = 1,
        acqCenter = 0.0,
        plotSeq = 0,
        init_gpa = False,):

    def createSequence():
        # Initialize time
        tEx = 20e3
        blkTime = 15

        for nRep in range(nRepetitions):
            # Excitation pulse
            t0 = tEx - blkTime - rfExTime / 2
            txTime = np.array([t0 + blkTime, t0 + blkTime + rfExTime])
            txAmp = np.array([rfExAmp * np.exp(1j * 0), 0.])
            txGateTime = np.array([t0, t0 + blkTime + rfExTime])
            txGateAmp = np.array([1, 0])
            expt.add_flodict({
                'tx0': (txTime, txAmp),
                'tx_gate': (txGateTime, txGateAmp)
            })
            # ttl channel 0:
            ttlGateTime = np.array([t0, t0 + rfExTime + blkTime])
            ttlAmp = np.array([1, 0])
            expt.add_flodict({
                'tx_gate': (ttlGateTime, ttlAmp),
            })

            # Refocusing pulse
            t0 = tEx + echoSpacing / 2 - blkTime - rfReTime / 2
            txTime = np.array([t0 + blkTime, t0 + blkTime + rfReTime])
            txAmp = np.array([rfReAmp * np.exp(1j * phaseRe * np.pi / 180), 0.])
            txGateTime = np.array([t0, t0 + blkTime + rfExTime])
            txGateAmp = np.array([1, 0])
            expt.add_flodict({
                'tx0': (txTime, txAmp),
                'tx_gate': (txGateTime, txGateAmp)
            })
            # ttl channel 1:
            ttlGateTime = np.array([t0, t0 + rfReTime + blkTime])
            ttlAmp = np.array([1, 0])
            expt.add_flodict({
                'rx_gate': (ttlGateTime, ttlAmp),
            })

            # Rx gate
            tEcho = tEx + echoSpacing - acqCenter
            t0 = tEcho - acqTime / 2
            rxGateTime = np.array([t0, t0 + acqTime])
            rxGateAmp = np.array([1, 0])
            expt.add_flodict({
                'rx0_en': (rxGateTime, rxGateAmp),
            })

            # Update time for next repetition
            tEx = tEx + repetitionTime

        tEnd = 20e3 + repetitionTime * nRepetitions
        txTime = np.array([0, tEnd-25])
        txAmp = np.array([rfExAmp * np.exp(1j * 0), 0.])
        expt.add_flodict({
            'tx1': (txTime, txAmp),
        })
        expt.add_flodict({
            'grad_vx': (np.array([tEnd]), np.array([0])),
            'grad_vy': (np.array([tEnd]), np.array([0])),
            'grad_vz': (np.array([tEnd]), np.array([0])),
            'rx0_en': (np.array([tEnd]), np.array([0])),
            'rx1_en': (np.array([tEnd]), np.array([0])),
            'rx_gate': (np.array([tEnd]), np.array([0])),
            'tx0': (np.array([tEnd]), np.array([0 * np.exp(0)])),
            'tx1': (np.array([tEnd]), np.array([0 * np.exp(0)])),
            'tx_gate': (np.array([tEnd]), np.array([0]))
        })
    # Time variables in us
    echoSpacing = echoSpacing * 1e3
    repetitionTime = repetitionTime * 1e3
    acqTime = acqTime * 1e3
    acqCenter = acqCenter * 1e3

    # Initialize the experiment
    bw = nPoints / acqTime  # * hw.oversamplingFactor  # MHz
    samplingPeriod = 1 / bw  # us
    # gpa_fhdo_offset_time= (1 / 0.2 / 3.1)
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=0)
    samplingPeriod = expt.get_rx_ts()[0]
    bw = 1 / samplingPeriod  # / hw.oversamplingFactor  # MHz
    acqTime = nPoints / bw  # us

    createSequence()

    if plotSeq == 0:
        # Run the experiment and get data
        print('Running...')
        dataFull = []
        spectrumFull = []
        for nScan in range(nScans):
            rxd, msgs = expt.run()
            print(msgs)
            data = rxd['rx0']  # * 13.788
            # data = sig.decimate(data, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            dataFull = np.concatenate((dataFull, data), axis=0)
            data = np.reshape(data, (nRepetitions, nPoints))
            for nRep in range(nRepetitions):
                spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data[nRep]))))
                spectrumFull = np.concatenate((spectrumFull, spectrum), axis=0)

        data = np.reshape(dataFull, (nRepetitions * nScans, -1))
        data = np.reshape(data, -1)
        spectrum = np.reshape(spectrumFull, (nRepetitions * nScans, -1))

        # Plotting echoes
        timeVector = np.linspace(0, acqTime * nRepetitions * nScans, num=nPoints * nRepetitions * nScans)
        timeVector = np.transpose(timeVector)*1e-3
        plt.figure(1)
        plt.plot(timeVector, np.abs(data))
        plt.plot(timeVector, np.real(data))
        plt.plot(timeVector, np.imag(data))
        plt.xlabel('t(ms)')
        plt.ylabel('A(mV)')

        # Plotting Phase
        repetitions = np.linspace(1, nRepetitions*nScans, nRepetitions*nScans)
        data = np.reshape(data, (nRepetitions*nScans, -1))
        phase = np.angle(data[:, int(nPoints/2)])
        plt.figure(2)
        plt.plot(repetitions, phase)
        plt.xlabel('Repetition')
        plt.ylabel('Phase (rad)')
        plt.show()
    elif plotSeq == 1:
        expt.plot_sequence()
        plt.show()
    expt.__del__()

    # def sequenceAnalysis(self, obj=''):
    #     self.saveRawData()
    #     data = self.mapVals['data']
    #     spectrum = self.mapVals['spectrum']
    #     bw = self.mapVals['bw']
    #
    #     # magnitude = Spectrum3DPlot(np.abs(data), title="Magnitude")
    #     # magnitudeWidget = magnitude.getImageWidget()
    #     #
    #     # phase = Spectrum3DPlot(np.angle(data), title="Phase")
    #     # phaseWidget = phase.getImageWidget()
    #     #
    #     # win = pg.LayoutWidget()
    #     # win.resize(300, 1000)
    #     # win.addWidget(magnitudeWidget, row=0, col=0)
    #     # win.addWidget(phaseWidget, row=0, col=1)
    #     # return([win])
    #
    #     # data = np.reshape(data, -1)
    #     acqTime = self.mapVals['acqTime']
    #     nRepetitions = self.mapVals['nRepetitions']
    #     nScans = self.mapVals['nScans']
    #     nPoints = self.mapVals['nPoints']
    #     timeVector = np.linspace(0, acqTime*nRepetitions*nScans, num=nPoints*nRepetitions*nScans)
    #     timeVector = np.transpose(timeVector)
    #
    #
    #     # fVector = np.linspace(0, bw*nRepetitions*nScans, nPoints*nRepetitions*nScans)
    #     # fVector = np.transpose(fVector)
    #     fVectorFull = []
    #     fVector = np.linspace(-bw/2, bw/2, nPoints)
    #     for nIndex in range(nRepetitions*nScans):
    #         fVectorFull = np.concatenate((fVectorFull, fVector), axis=0)
    #     fVector = np.transpose(fVectorFull)
    #
    #     data = np.reshape(data, -1)
    #     spectrum = np.reshape(spectrum, -1)
    #
    #     # Plot signal versus time
    #     magPlotWidget = SpectrumPlot(xData=timeVector,
    #                             yData=[np.abs(data), np.real(data), np.imag(data)],
    #                             legend=['abs', 'real', 'imag'],
    #                             xLabel='Time (ms)',
    #                             yLabel='Signal amplitude (mV)',
    #                             title='Magnitude')
    #
    #     specPlotWidget = SpectrumPlot(xData=fVector,
    #                                  yData=[spectrum],
    #                                  legend=['abs'],
    #                                  xLabel='f (kHz)',
    #                                  yLabel='spectrum amplitude (a. u)',
    #                                  title='FFT')
    #     anglePlotWidget = SpectrumPlot(xData=timeVector,
    #                                   yData=[np.angle(data)],
    #                                   legend=['abs', 'real', 'imag'],
    #                                   xLabel='Time (ms)',
    #                                   yLabel='Phase (rad)',
    #                                   title='Phase')
    #
    #     repetitions = np.linspace(1, nRepetitions*nScans, nRepetitions*nScans)
    #     data = np.reshape(data, (nRepetitions*nScans, -1))
    #     phase = np.angle(data[:, int(nPoints/2)])
    #     phasePlotWidget = SpectrumPlot(xData=repetitions,
    #                                    yData=[np.unwrap(phase)],
    #                                    legend=[''],
    #                                    xLabel='Repetition',
    #                                    yLabel='Phase (rad)',
    #                                    title='Phase')
    #
    #     self.out = [magPlotWidget, phasePlotWidget]
    #     return(self.out)

#  MAIN  ######################################################################################################
if __name__ == "__main__":
    testSE_standalone()