"""
@author: T. Guallart Naval
MRILAB @ I3M
"""

import experiment as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import configs.hw_config as hw



class testSE(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(testSE, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='testSE', val='testSE')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.075, field='RF')
        self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=36.0, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=72.0, field='RF')
        self.addParameter(key='phaseRe', string='RF refocusing phase', val=np.pi/2, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=10.0, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=50., field='SEQ')
        self.addParameter(key='nRepetitions', string='Number of repetitions ', val=60, field='SEQ')
        self.addParameter(key='nScans', string='Number of scans ', val=60, field='SEQ')
        self.addParameter(key='acqCenter', string='Acq center (ms)', val=0.0, field='SEQ')
        self.addParameter(key='nPoints', string='nPoints', val=90, field='IM')
        self.addParameter(key='acqTime', string='Acquisition time (ms)', val=4.0, field='SEQ')
        self.addParameter(key='ttlExtra', string='TTL (1-pi/2 pulse; 2-pi pulse) (ms)', val=2, field='SEQ')

    def sequenceInfo(self):
        print(" ")
        print("Testing SE")
        print("Author: T. Guallart Naval")
        print("mriLab @ i3M, CSIC, Spain")
        print("This sequence runs a spin echo without gradients")

    def sequenceTime(self):
        repetitionTime = self.mapVals['repetitionTime']*1e-3
        nRepetitions  =  self.mapVals['nRepetitions']
        nScans = self.mapVals['nScans']
        return(repetitionTime*nRepetitions*nScans/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False  # Starts the gpa

        seqName = self.mapVals['seqName']
        larmorFreq = self.mapVals['larmorFreq']
        rfExAmp = self.mapVals['rfExAmp']
        rfExTime = self.mapVals['rfExTime']
        rfReAmp = self.mapVals['rfReAmp']
        rfReTime = self.mapVals['rfReTime']
        phaseRe = self.mapVals['phaseRe']
        echoSpacing = self.mapVals['echoSpacing']
        repetitionTime = self.mapVals['repetitionTime']
        nRepetitions  =  self.mapVals['nRepetitions']
        nScans = self.mapVals['nScans']
        nPoints = self.mapVals['nPoints']
        acqTime = self.mapVals['acqTime']
        acqCenter = self.mapVals['acqCenter']
        ttlExtra = self.mapVals['ttlExtra']

        def createSequence():
            # Initialize time
            t0 = 25
            tEx = 20e3

            for nRep in range(nRepetitions):
                # Excitation pulse

                t0 = tEx - hw.blkTime - rfExTime / 2
                t0Ex = t0
                # self.rfRecPulse(t0, rfExTime, rfExAmp, 0)
                # self.ttl(t0, rfExTime+hw.blkTime, channel=0)
                # if ttlExtra == 1:
                #     self.ttl(t0, rfExTime, channel=1)

                # Refocusing pulse
                t0 = tEx + echoSpacing/2 - hw.blkTime - rfReTime / 2
                # self.rfRecPulse(t0, rfReTime, rfReAmp, phaseRe*np.pi/180)
                # self.ttl(t0, rfReTime+hw.blkTime, channel=0)
                # self.ttl(t0, rfReTime+hw.blkTime, channel=1)
                # if ttlExtra == 2:
                #     self.ttl(t0, rfReTime, channel=1)

                # Rx gate
                tEcho = tEx + echoSpacing - acqCenter
                t0 = tEcho - acqTime / 2
                self.rxGate(t0, acqTime)
                self.ttl(t0, acqTime, channel=0)
                # Update time for next repetition
                tEx = tEx + repetitionTime
            self.endSequence(20e3 + repetitionTime*nRepetitions)

        # Time variables in us
        echoSpacing = echoSpacing * 1e3
        repetitionTime = repetitionTime * 1e3
        acqTime = acqTime * 1e3
        acqCenter = acqCenter * 1e3

        # Initialize the experiment
        # bw = 50
        bw = nPoints / acqTime #* hw.oversamplingFactor  # MHz
        samplingPeriod = 1 / bw  # us
        # gpa_fhdo_offset_time= (1 / 0.2 / 3.1)
        self.expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time= 0)
        samplingPeriod = self.expt.get_rx_ts()[0]
        bw = 1 / samplingPeriod #/ hw.oversamplingFactor  # MHz
        acqTime = nPoints / bw  # us
        self.mapVals['bw'] = bw
        createSequence()
        if self.floDict2Exp():
            print("\nSequence waveforms loaded successfully")
            pass
        else:
            print("\nERROR: sequence waveforms out of hardware bounds")
            return False

        if plotSeq == 0:
            # Run the experiment and get data
            print('Running...')
            dataFull = []
            spectrumFull = []
            for nScan in range(nScans):
                rxd, msgs = self.expt.run()
                print(msgs)
                self.mapVals['dataFull'] = rxd['rx0'] #* 13.788
                data = rxd['rx0'] #* 13.788
                # data = sig.decimate(data, hw.oversamplingFactor, ftype='fir', zero_phase=True)
                dataFull = np.concatenate((dataFull, data), axis=0)
                data = np.reshape(data, (nRepetitions, nPoints))
                for nRep in range(nRepetitions):
                    spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data[nRep]))))
                    spectrumFull = np.concatenate((spectrumFull, spectrum), axis=0)

            data = np.reshape(dataFull, (nRepetitions*nScans, -1))
            self.mapVals['data'] = data
            spectrum = np.reshape(spectrumFull, (nRepetitions * nScans, -1))
            self.mapVals['spectrum'] = spectrum

        self.expt.__del__()

        return True

    def sequenceAnalysis(self, obj=''):
        data = self.mapVals['data']
        spectrum = self.mapVals['spectrum']
        bw = self.mapVals['bw']

        # magnitude = Spectrum3DPlot(np.abs(data), title="Magnitude")
        # magnitudeWidget = magnitude.getImageWidget()
        #
        # phase = Spectrum3DPlot(np.angle(data), title="Phase")
        # phaseWidget = phase.getImageWidget()
        #
        # win = pg.LayoutWidget()
        # win.resize(300, 1000)
        # win.addWidget(magnitudeWidget, row=0, col=0)
        # win.addWidget(phaseWidget, row=0, col=1)
        # return([win])

        # data = np.reshape(data, -1)
        acqTime = self.mapVals['acqTime']
        nRepetitions = self.mapVals['nRepetitions']
        nScans = self.mapVals['nScans']
        nPoints = self.mapVals['nPoints']
        timeVector = np.linspace(0, acqTime*nRepetitions*nScans, num=nPoints*nRepetitions*nScans)
        timeVector = np.transpose(timeVector)


        # fVector = np.linspace(0, bw*nRepetitions*nScans, nPoints*nRepetitions*nScans)
        # fVector = np.transpose(fVector)
        fVectorFull = []
        fVector = np.linspace(-bw/2, bw/2, nPoints)
        for nIndex in range(nRepetitions*nScans):
            fVectorFull = np.concatenate((fVectorFull, fVector), axis=0)
        fVector = np.transpose(fVectorFull)

        data = np.reshape(data, -1)
        spectrum = np.reshape(spectrum, -1)

        # Plot signal versus time
        result1 = {'widget': 'curve',
                   'xData': timeVector,
                   'yData': [np.abs(data), np.real(data), np.imag(data)],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Magnitude',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 0,
                   'col': 0}

        result2 = {'widget': 'curve',
                   'xData': fVector,
                   'yData': [spectrum],
                   'xLabel': 'Frequency (kHz)',
                   'yLabel': 'Spectrum amplitude (a.u.)',
                   'title': 'Spectrum magnitude',
                   'legend': ['abs'],
                   'row': 1,
                   'col': 0}

        result3 = {'widget': 'curve',
                   'xData': timeVector,
                   'yData': [np.angle(data)],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Phase (rad)',
                   'title': 'Signal phase',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 0,
                   'col': 1}

        repetitions = np.linspace(1, nRepetitions*nScans, nRepetitions*nScans)
        data = np.reshape(data, (nRepetitions*nScans, -1))
        phase = np.angle(data[:, int(nPoints/2)])
        result4 = {'widget': 'curve',
                   'xData': repetitions,
                   'yData': [np.unwrap(phase)],
                   'xLabel': 'Repetition',
                   'yLabel': 'Phase (rad)',
                   'title': 'Signal phase',
                   'legend': [''],
                   'row': 1,
                   'col': 1}

        self.output = [result1, result2, result3, result4]

        self.saveRawData()

        return self.output
