"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: rare sequence class
"""

import numpy as np
import controller.experiment_gui as ex
import torch
import configs.hw_config as hw  # Import the scanner hardware config
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.


# *********************************************************************************
# *********************************************************************************
# *********************************************************************************

class MRID(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(MRID, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='MRIDInfo', val='MRID')
        self.addParameter(key='n_diameter', string='Number of diameters', val=4, field='IM')
        self.addParameter(key='larmor_freq', string='Larmor frequency (MHz)', val=11.13, field='RF')
        self.addParameter(key='rf_time', string='RF excitation time (us)', val=22.0, field='RF')
        #self.addParameter(key='rf_ex_amp', string='Amplitude of excitation (a.u.)', val=0.4, field='RF')
        self.addParameter(key='rf_calib_amp', string='Amplitude of 90º excitation (a.u.)', val=0.4, field='RF')
        self.addParameter(key='rf_re_amp', string='Amplitude of refocusing excitation (a.u.)', val=0.8, field='RF')
        #self.addParameter(key='echo_time', string='Echo Time (ms)', val=10.0, field='SEQ')
        self.addParameter(key='dead_time', string='TxRx dead time (us)', val=100.0, field='RF')
        self.addParameter(key='gap_g_to_rf', string='Gap G to RF (us)', val=100.0, field='RF')
        self.addParameter(key='fov', string='FOV (cm)', val=[1.0, 1.0, 1.0], field='IM')
        self.addParameter(key='dfov', string='dFOV (mm)', val=[0.0, 0.0, 0.0], field='IM')
        self.addParameter(key='n_points', string='n_points (rd, ph, sl)', val=[20, 20, 1], field='IM')
        self.addParameter(key='acq_time', string='Acquisition time (ms)', val=0.5, field='SEQ', tip="For radius")
        self.addParameter(key='axes', string='Axes', val=[0, 1, 2], field='IM')
        self.addParameter(key='axes_enable', string='Axes enable', val=[1, 1, 0], field='IM', tip="Always [1, 1, 0]")
        self.addParameter(key='axesOn', string='Axes ON', val=[1, 1, 1], field='IM')
        self.addParameter(key='drfPhase', string='Phase of excitation pulse (º)', val=0.0, field='RF')
        self.addParameter(key='shimming', string='Shimming (*1e4)', val=[-70, -90, 10], field='OTH')
        self.addParameter(key='grad_rise_time', string='Grad Rise Time (us)', val=8000, field='OTH')
        self.addParameter(key='nStepsGradRise', string='Grad steps', val=5, field='OTH')
        self.addParameter(key='tx_channel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rx_channel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='nyquist_os', string='Radial oversampling', val=1, field='SEQ')
        # self.addParameter(key='reco', string='ART->0,  FFT->1', val=1, field='IM')
        self.addParameter(key='repetition_time', string='Repetition time (ms)', val=100, field='SEQ')
        self.addParameter(key='gap', string='Gap', val=2.0, field='IM')
        self.addParameter(key='nScans', string='Number of Scans', val=1, field='SEQ')
        self.addParameter(key='T1', string='T1 sample (ms)', val=1, field='SEQ')
        self.addParameter(key='T2', string='T2 sample (ms)', val=1, field='SEQ')

    def sequenceInfo(self):
        
        print("3D MRID sequence")
        print("Author: Jose Borreguero")
        print("Contact: pepe.morata@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")

    def sequenceTime(self):
        return self.mapVals['repetition_time']*self.mapVals['n_diameter']*self.mapVals['nScans']*1e-3/60 #minutes

    def sequenceAtributes(self):
        super().sequenceAtributes()

        # Conversion of variables to non-multiplied units
        self.larmor_freq = self.larmor_freq * 1e6
        self.rf_time = self.rf_time * 1e-6
        self.gap_g_to_rf = self.gap_g_to_rf * 1e-6
        self.dead_time = self.dead_time * 1e-6
        self.grad_rise_time = self.grad_rise_time * 1e-6  # s
        self.fov = np.array(self.fov) * 1e-2
        self.dfov = np.array(self.dfov) * 1e-3
        self.acq_time = self.acq_time * 1e-3  # s
        self.shimming = np.array(self.shimming) * 1e-3 * 1e-4
        self.repetition_time = self.repetition_time * 1e-3  # s
        self.echo_time = np.loadtxt(r'C:\Users\User\MRID_parameters\TE(70-150).txt',delimiter=',') * 1e-3  # s
        self.mapVals['echo_time']=self.echo_time

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False  # Starts the gpa
        freqCal = True  # Swich off only if you want and you are on debug mode
        self.demo = demo

        seqName = self.mapVals['seqName']
        self.flip_angle =  np.loadtxt(r'C:\Users\User\MRID_parameters\FA100(90).txt',delimiter=',')
        self.rf_ex_amp= self.flip_angle* self.mapVals['rf_calib_amp']/ (90)  # a.u. FA in degrees! (verificar formula)
        rf_re_amp = self.mapVals['rf_re_amp']
        # repetition_time = np.loadtxt('TR.txt', delimeter=',')  # ms
        n_points = np.array(self.mapVals['n_points'])
        axes = self.mapVals['axes']
        axes_enable = self.mapVals['axes_enable']
        drfPhase = self.mapVals['drfPhase']  # degrees
        tx_channel = self.mapVals['tx_channel']
        rx_channel = self.mapVals['rx_channel']
        nyquist_os = self.mapVals['nyquist_os']
        nStepsGradRise= self.mapVals['nStepsGradRise']
        self.n_repetition=len(self.flip_angle)
        self.mapVals['n_repetition']=self.n_repetition
        self.mapVals['flip_angle']=self.flip_angle
        self.mapVals['rf_ex_amp']=self.rf_ex_amp


        # Miscellaneous
        self.larmor_freq = self.larmor_freq * 1e-6  # MHz
        resolution = self.fov / n_points
        self.mapVals['resolution'] = resolution

        # Get cartesian parameters
        dK = 1 / self.fov
        kMax = n_points / (2 * self.fov)  # m-1

        # SetSamplingParameters
        BW = (np.max(n_points)) * nyquist_os / (2 * self.acq_time)   # Hz
        samplingPeriod = 1 / BW
        self.mapVals['BW'] = BW
        self.mapVals['kMax'] = kMax
        self.mapVals['dK'] = dK

        nPPL_rad = int((self.echo_time[0]/2-2*self.dead_time-self.rf_time) * BW )-hw.addRdPoints*hw.oversamplingFactor #(addPoints so the acquisition windows do not overlap)
        nPPL_dia = int((self.repetition_time-self.echo_time[0]/2-self.rf_time-2*self.dead_time) * BW )-hw.addRdPoints*hw.oversamplingFactor

        if nPPL_dia<=0 or nPPL_rad<=0:
            print('Too few points. Please select a greater amount of points')

            return False

        acq_time_rad = (nPPL_rad) / (BW*1e-6)  # us
        acq_time_dia = (nPPL_dia) / (BW*1e-6) # us
        self.mapVals['acq_timeReal'] = self.acq_time*1e3   # ms
        self.mapVals['nPPL_rad'] = nPPL_rad
        self.mapVals['nPPL_dia'] = nPPL_dia


        gradientAmplitudes = kMax / (hw.gammaB * self.acq_time)

        # Calculate radial gradients
        normalizedGradientsRadial = np.zeros((self.n_diameter, 3))
        n = -1

        # Calculate the normalized gradients:
        for jj in range(self.n_diameter):
            theta = 111.25 * (np.pi /180)* (jj+1)

            n += 1
            normalizedGradientsRadial[n, 0] = np.sin(theta)
            normalizedGradientsRadial[n, 1] = np.cos(theta)
            normalizedGradientsRadial[n, 2] = 0

        # Set gradients to T/m
        gradientVectors1 = np.matmul(normalizedGradientsRadial, np.diag(gradientAmplitudes))


        # Calculate radial k-points at t = 0.5*rf_time+td
        kRadial = []
        kDia = []
        normalizedKRadial = np.zeros((self.n_diameter, 3, nPPL_rad))
        normalizedKDiam = np.zeros((self.n_diameter, 3, nPPL_dia))
        normalizedKRadial[:, :, 0] = (0.5 * self.rf_time + self.dead_time + (0.5 / (BW ))) * normalizedGradientsRadial


        # Calculate all k-points
        for jj in range(1, nPPL_rad):
            normalizedKRadial[:, :, jj] = normalizedKRadial[:, :, 0] + jj * normalizedGradientsRadial / (BW)

        normalizedKDiam[:, :, 0] = -normalizedKRadial[:,:,-1]
        for jj in range(1, nPPL_dia):
            normalizedKDiam[:, :, jj] = normalizedKDiam[:, :, 0] + jj * normalizedGradientsRadial / (BW)

        a = np.zeros(shape=(normalizedKRadial.shape[2], normalizedKRadial.shape[0], normalizedKRadial.shape[1]))
        a[:, :, 0] = np.transpose(np.transpose(np.transpose(normalizedKRadial[:, 0, :])))
        a[:, :, 1] = np.transpose(np.transpose(np.transpose(normalizedKRadial[:, 1, :])))
        a[:, :, 2] = np.transpose(np.transpose(np.transpose(normalizedKRadial[:, 2, :])))

        aux0reshape = np.reshape(np.transpose(a[:, :, 0]), [self.n_diameter * nPPL_rad, 1])
        aux1reshape = np.reshape(np.transpose(a[:, :, 1]), [self.n_diameter * nPPL_rad, 1])
        aux2reshape = np.reshape(np.transpose(a[:, :, 2]), [self.n_diameter * nPPL_rad, 1])

        normalizedKRadial = np.concatenate((aux0reshape, aux1reshape, aux2reshape), axis=1)
        kRadial = (np.matmul(normalizedKRadial, np.diag((hw.gammaB * gradientAmplitudes))))

        b = np.zeros(shape=(normalizedKDiam.shape[2], normalizedKDiam.shape[0], normalizedKDiam.shape[1]))
        b[:, :, 0] = np.transpose(np.transpose(np.transpose(normalizedKDiam[:, 0, :])))
        b[:, :, 1] = np.transpose(np.transpose(np.transpose(normalizedKDiam[:, 1, :])))
        b[:, :, 2] = np.transpose(np.transpose(np.transpose(normalizedKDiam[:, 2, :])))

        aux0reshape1 = np.reshape(np.transpose(b[:, :, 0]), [ self.n_diameter * nPPL_dia, 1])
        aux1reshape2 = np.reshape(np.transpose(b[:, :, 1]), [ self.n_diameter * nPPL_dia, 1])
        aux2reshape3 = np.reshape(np.transpose(b[:, :, 2]), [ self.n_diameter * nPPL_dia, 1])

        normalizedKDiam = np.concatenate((aux0reshape1, aux1reshape2, aux2reshape3), axis=1)
        kDia = (np.matmul(normalizedKDiam, np.diag((hw.gammaB * gradientAmplitudes))))

        # Get cartesian kPoints
        # Get minimun time

        # Get the full cartesian points
        kx = np.linspace(-kMax[0] * (n_points[0] != 1), kMax[0] * (n_points[0] != 1), n_points[0])
        ky = np.linspace(-kMax[1] * (n_points[1] != 1), kMax[1] * (n_points[1] != 1), n_points[1])
        kz = np.linspace(-kMax[2] * (n_points[2] != 1), kMax[2] * (n_points[2] != 1), n_points[2])

        kx, ky, kz = np.meshgrid(kx, ky, kz)
        kx = torch.from_numpy(kx)
        kx = kx.permute(2, 0, 1)
        ky = torch.from_numpy(ky)
        ky = ky.permute(2, 0, 1)
        kz = torch.from_numpy(kz)
        kz = kz.permute(2, 0, 1)

        kCartesian = np.zeros(shape=(kx.shape[0] * kx.shape[1] * kx.shape[2], 3))
        kCartesian[:, 0] = np.reshape(kx, [kx.shape[0] * kx.shape[1] * kx.shape[2]])
        kCartesian[:, 1] = np.reshape(ky, [ky.shape[0] * ky.shape[1] * ky.shape[2]])
        kCartesian[:, 2] = np.reshape(kz, [kz.shape[0] * kz.shape[1] * kz.shape[2]])
        self.mapVals['kCartesian'] = kCartesian

        kSpaceValues = np.concatenate((kRadial,kDia))
        self.mapVals['kSpaceValues'] = kSpaceValues

        # Set gradients for cartesian sampling

        gSeq = - gradientVectors1
        if self.mapVals['n_diameter']==1:
            gSeqDif = np.diff(gSeq, n=0, axis=0)
        else:
            gSeqDif = np.diff(gSeq, n=1, axis=0)
        MaxGradTransitions = kMax / (hw.gammaB * self.acq_time)
        MaxGradTransitions[0] = max(gSeqDif[:, 0])
        MaxGradTransitions[1] = max(gSeqDif[:, 1])
        MaxGradTransitions[2] = max(gSeqDif[:, 2])

        print(gradientVectors1.shape[0], " radial lines")
        print("Radial max gradient strengths are  ", gradientAmplitudes * 1e3, " mT/m")

        print("Max grad transitions are  ", MaxGradTransitions * 1e3, " mT/m")

        # Gradientes a 0 para sin imagen
        gSeq=gSeq*0

        self.mapVals['SequenceGradients'] = gSeq




        def createSequence():
            g_rise_time = self.grad_rise_time * 1e6
            tr = self.repetition_time * 1e6
            delayGtoRF = self.gap_g_to_rf * 1e6
            RFpulsetime = self.rf_time * 1e6
            axesOn=self.mapVals['axesOn']
            TxRxtime = self.dead_time * 1e6
            echo_time = self.mapVals['echo_time']*1e6
            tInit = 20
            acq_points = 0

            # Set shimming
            self.iniSequence(tInit, self.shimming)

            # while diaIndex < self.mapVals['n_diameter']:
            repeIndex=0
            #     # Set gradients
            #     if diaIndex == 0:
            #         ginit = np.array([0, 0, 0])
            #         self.setGradientRamp(t0, g_rise_time, nStepsGradRise, ginit[0], gSeq[0, 0] * axesOn[0], axes[0],
            #                              self.shimming)
            #         self.setGradientRamp(t0, g_rise_time, nStepsGradRise, ginit[1], gSeq[0, 1] * axesOn[1], axes[1],
            #                              self.shimming)
            #     elif diaIndex > 0:
            #         self.setGradientRamp(t0, g_rise_time, nStepsGradRise, gSeq[diaIndex - 1, 0] * axesOn[0],gSeq[diaIndex, 0] * axesOn[0], axes[0], self.shimming)
            #         self.setGradientRamp(t0, g_rise_time, nStepsGradRise, gSeq[diaIndex - 1, 1] * axesOn[1], gSeq[diaIndex, 1] * axesOn[1], axes[1], self.shimming)

            while repeIndex < self.mapVals['n_repetition'] :
                # Initialize time
                t0 = tInit + tr * (repeIndex + 1)

                # Excitation pulse
                trf0 = t0 + g_rise_time + delayGtoRF
                rx_amp=self.mapVals['rf_ex_amp'][repeIndex]
                self.rfRecPulse(trf0, RFpulsetime, rx_amp, drfPhase * np.pi / 180,  channel=tx_channel)

                # Rx gate
                t0rx = trf0 + hw.blkTime + RFpulsetime + TxRxtime
                self.rxGateSync(t0rx, acq_time_rad, channel=rx_channel)
                acq_points += nPPL_rad

                # Refocusing pulse
                trf0 = trf0 + echo_time[repeIndex]/2
                self.rfRecPulse(trf0, RFpulsetime, self.rf_re_amp, drfPhase * np.pi / 180+np.pi/2, channel=tx_channel)

                # Rx gate Refocusing
                t0rx = trf0 + hw.blkTime + RFpulsetime + TxRxtime
                self.rxGateSync(t0rx, acq_time_dia, channel=rx_channel)
                acq_points += nPPL_dia


                if repeIndex == self.n_repetition - 1:
                    self.endSequence(t0 + tr)

                repeIndex = repeIndex + 1
                # diaIndex= diaIndex + 1
            return acq_points

        # Time parameters to microseconds
        samplingPeriod = samplingPeriod * 1e6  # us
        self.mapVals['samplingPeriod']=samplingPeriod # us
        over_data_rad = []
        over_data_dia = []
        over_data = []
        data_full = []
        if not self.demo:
            # Create experiment and get true acquisition time
            self.expt = ex.Experiment(lo_freq=self.larmor_freq, rx_t=samplingPeriod, init_gpa=init_gpa,
                                      gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
            samplingPeriod = self.expt.getSamplingRate()
            BW = 1 / samplingPeriod  # MHz
            acq_time_rad = (nPPL_rad ) / BW  # us
            acq_time_dia = (nPPL_dia ) / BW  # us
            self.mapVals['samplingPeriod'] = samplingPeriod  # us

            # Create the sequence
            acq_points = createSequence()

            # Instructions to the red pitaya
            if self.floDict2Exp():
                print("Sequence waveforms loaded successfully")
                pass
            else:
                print("ERROR: sequence waveforms out of hardware bounds")
                return False

            # Execute the experiment
            if plotSeq==0:
                n_diameter=self.mapVals['n_diameter']
                nScans=self.mapVals['nScans']
                # Run the experiment and get data as many times as required by user
                for scan in range(nScans):
                    rxd, msgs = self.expt.run()
                    data=rxd['rx0']

                    # Split data to "after excitation" and "after refocusing"
                    for repetition in range(self.mapVals['n_repetition']):
                        # Concatenate points after excitation
                        n_ini = ((nPPL_rad + nPPL_dia + 4 * hw.addRdPoints) * hw.oversamplingFactor) * repetition
                        n_fin = n_ini + (nPPL_rad + 2 * hw.addRdPoints) * hw.oversamplingFactor
                        over_data_rad = np.concatenate((over_data_rad, data[n_ini:n_fin]), axis=0)

                        # Concatenate points after refocusing
                        n_ini = n_fin * 1
                        n_fin = n_ini + (nPPL_dia + 2 * hw.addRdPoints) * hw.oversamplingFactor
                        over_data_dia = np.concatenate((over_data_dia, data[n_ini:n_fin]), axis=0)

                # Decimate the results
                data_full_rad = self.decimate(over_data_rad, self.n_repetition * self.nScans)
                data_full_dia = self.decimate(over_data_dia, self.n_repetition * self.nScans)

                # Concatenate the results
                for repetition in range(self.n_repetition * self.nScans):
                    # Add "after excitation" data
                    n_ini = nPPL_rad * repetition
                    n_fin = nPPL_rad * (repetition + 1)
                    data_full = np.concatenate((data_full, data_full_rad[n_ini:n_fin]), axis=0)

                    # Add "after refocusing data
                    n_ini = nPPL_dia * repetition
                    n_fin = nPPL_dia * (repetition + 1)
                    data_full = np.concatenate((data_full, data_full_dia[n_ini:n_fin]), axis=0)

                # Save results into mapVals
                self.mapVals['dataFullrad'] = data_full_rad
                self.mapVals['dataFulldia'] = data_full_dia
                self.mapVals['overData_rad'] = over_data_rad
                self.mapVals['overData_dia'] = over_data_dia
                self.mapVals['dataFull'] = data_full

            # Delete the experiment
            self.expt.__del__()

        else:
            # Create the sequence and get the number of expected acquired points
            acq_points = createSequence()

            # Generate random points
            for scan in range(self.nScans):
                data = np.random.randn((acq_points+self.n_diameter*hw.addRdPoints*4)*hw.oversamplingFactor)

                # Split data to "after excitation" and "after refocusing"
                for repetition in range(self.n_repetition):
                    # Concatenate points after excitation
                    n_ini = ((nPPL_rad + nPPL_dia + 4 * hw.addRdPoints) * hw.oversamplingFactor) * repetition
                    n_fin = n_ini + (nPPL_rad + 2 * hw.addRdPoints) * hw.oversamplingFactor
                    over_data_rad = np.concatenate((over_data_rad, data[n_ini:n_fin]), axis=0)

                    # Concatenate points after refocusing
                    n_ini = n_fin * 1
                    n_fin = n_ini + (nPPL_dia + 2 * hw.addRdPoints) * hw.oversamplingFactor
                    over_data_dia = np.concatenate((over_data_dia, data[n_ini:n_fin]), axis=0)

            # Decimate the results
            data_full_rad = self.decimate(over_data_rad, self.n_repetition * self.nScans)
            data_full_dia = self.decimate(over_data_dia, self.n_repetition * self.nScans)

            # Concatenate the results
            for repetition in range(self.n_repetition * self.nScans):
                # Add "after excitation" data
                n_ini = nPPL_rad * repetition
                n_fin = nPPL_rad * (repetition + 1)
                data_full = np.concatenate((data_full, data_full_rad[n_ini:n_fin]), axis=0)

                # Add "after refocusing data
                n_ini = nPPL_dia * repetition
                n_fin = nPPL_dia * (repetition + 1)
                data_full = np.concatenate((data_full, data_full_dia[n_ini:n_fin]), axis=0)

            # Save results into mapVals
            self.mapVals['dataFullrad'] = data_full_rad
            self.mapVals['dataFulldia'] = data_full_dia
            self.mapVals['overData_rad'] = over_data_rad
            self.mapVals['overData_dia'] = over_data_dia
            self.mapVals['dataFull'] = data_full

        return True

    def sequenceAnalysis(self, obj=''):


        # Load data
        data_full = np.abs(self.mapVals['dataFull'])
        data_rad = np.abs(self.mapVals['dataFullrad'])
        data_dia = np.abs(self.mapVals['dataFulldia'])
        n_echos=np.linspace(1,self.mapVals['n_repetition'],self.mapVals['n_repetition'])
        nPPL_rad = self.mapVals['nPPL_rad']
        nPPL_dia = self.mapVals['nPPL_dia']
        sampling_period = self.mapVals['samplingPeriod'] # us
        echo_time = self.mapVals['echo_time']*1e3 # us
        repetition_time = self.mapVals['repetition_time']*1e3 # us

        # Create time vectors for first repetition
        t_ini = self.rf_time/2 + self.dead_time + sampling_period/2
        t_fin = t_ini + (nPPL_rad - 1) * sampling_period
        time_vector_rad_0 = np.linspace(t_ini, t_fin, nPPL_rad)
        t_ini = echo_time[0] / 2 + self.rf_time / 2 + self.dead_time + sampling_period / 2
        t_fin = t_ini + (nPPL_dia - 1) * sampling_period
        time_vector_dia_0 = np.linspace(t_ini, t_fin, nPPL_dia)

        time_vector = []
        max_echo = []

        echo_point=int(np.ceil(echo_time/(2*sampling_period)))
        for kk in range(self.mapVals['n_repetition']):
            time_vector_rad = time_vector_rad_0 + repetition_time * kk
            time_vector_dia = time_vector_dia_0 + repetition_time * kk
            time_vector = np.concatenate((time_vector, time_vector_rad), axis=0)
            time_vector = np.concatenate((time_vector, time_vector_dia), axis=0)
            echo_max=data_dia[int(kk*(self.mapVals['nPPL_dia'])+echo_point[kk])]
            max_echo=np.concatenate((max_echo,[echo_max]),axis=0)

        self.mapVals['data_time'] =[time_vector, data_full]


        echo_val=np.zeros(self.mapVals['n_repetition'])
        echo1=np.sin(self.flip_angle[0])*np.exp(-(echo_time[0]*1e-3)/self.mapVals['T2'])
        echo_val[0]=echo1
        for kk in range(self.mapVals['n_repetition']-1):
            echo_val[kk+1] = np.sin(self.flip_angle[kk+1])*np.exp(-echo_time[kk]*1e-3/self.mapVals['T2'])\
                            *(-echo_val[kk]*np.exp(-repetition_time*1e-3/self.mapVals['T1'])* np.cos(self.flip_angle[kk+1])
                             +1-2*np.exp(-1e-3*(repetition_time-(echo_time[kk]/2))/self.mapVals['T1'])+np.exp(-1e-3*repetition_time/self.mapVals['T1']))


        norm_echo_max = abs(max_echo) / (max_echo.max())
        norm_echo_val= abs(echo_val) / echo_val.max()

        self.mapVals['norm_echos_max'] = [n_echos, norm_echo_max]

        print("Stationnary experimental amplitude:", norm_echo_max[-1] )
        print("Stationnary synthetic value:", norm_echo_val[-1])

        result1 = {'widget': 'curve',
                   'xData': [time_vector*1e-3],
                   'yData': [data_full],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Signal vs time',
                   'legend': ['abs data'],
                   'row': 0,
                   'col': 0}

        result2 = {'widget': 'curve',
                   'xData': n_echos,
                   'yData': [norm_echo_max,norm_echo_val],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (a.u.)',
                   'title': 'Echo max vs number of repetitions',
                   'legend': ['normalized experimental max', 'normalized synthetic max'],
                   'row': 1,
                   'col': 0}

        # Add results into the output attribute (result must be the image to save in dicom)
        self.output = [result1,result2]

        # Save results
        self.saveRawData()

        return self.output



if __name__=="__main__":
    seq = MRID()
    seq.sequenceRun(plotSeq=1)