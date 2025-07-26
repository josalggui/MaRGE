# -*- coding: utf-8 -*-
"""
Created on Sat Nov  13 13:45:05 2021

@author: José Miguel Algarín Guisado
MRILAB @ I3M
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
import marge.marcos.marcos_client.experiment
import scipy.signal as sig
import marge.configs.hw_config as hw # Import the scanner hardware config
import marge.seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import pyqtgraph as pg              
import marge.configs.units as units

from datetime import date
from datetime import datetime
import ismrmrd
import ismrmrd.xsd
import datetime
import ctypes
import matplotlib
import xml.etree.ElementTree as ET
from scipy.io import loadmat

class GRE3D(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(GRE3D, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='GRE3DInfo', val='GRE3D')
        self.addParameter(key='toMaRGE', val=True)
        self.addParameter(key='nScans', string='Number of scans', val=2, field='IM')
        self.addParameter(key='freqOffset', string='Larmor frequency offset (kHz)', val=0.0, units=units.kHz,
                          field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (º)', val=90, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30.0, units=units.us, field='RF')
        self.addParameter(key='echo_time', string='Echo time (ms)', val=4.0, units=units.ms, field='SEQ')
        self.addParameter(key='repetition_time', string='Repetition time (ms)', val=500., units=units.ms, field='SEQ')
        self.addParameter(key='fov', string='FOV[x,y,z] (cm)', val=[15.0, 15.0, 15.0], units=units.cm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM',
                          tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints (rd, ph, sl)', val=[60, 60, 2], field='IM')
        self.addParameter(key='angle', string='Angle (º)', val=0.0, field='IM')
        self.addParameter(key='rotationAxis', string='Rotation axis', val=[0, 0, 1], field='IM')
        self.addParameter(key='acq_time', string='Acquisition time (ms)', val=1.0, units=units.ms, field='SEQ')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[0, 1, 2], field='IM',
                          tip="0=x, 1=y, 2=z")
        self.addParameter(key='axesEnable', string='Axes enable', val=[1, 1, 0],
                          tip="Use 0 for directions with matrix size 1, use 1 otherwise.")
        self.addParameter(key='rdGradTime', string='Rd gradient time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='dephGradTime', string='Rd dephasing time (ms)', val=1.0, units = units.ms, field='OTH')
        self.addParameter(key='dummy_pulses', string='Dummy pulses', val=1, field='SEQ')
        self.addParameter(key='shimming', string='Shimming (a.u.)', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='sl_fraction', string='Partial fourier fraction', val=1.0, field='OTH',
                          tip="Fraction of k planes aquired in slice direction")
        self.addParameter(key='mode', string='Sequence mode', val=0, field='SEQ',
                          tip='0: normal, 1: rf spoiler, 2: gradient spoiler, 3: rf and gradient spoilre, 4: balanced.')
        self.addParameter(key='spoiler_order', string='Gradient spoiler order', val=3, field='SEQ',
                          tip='Higher orders will require longer repetition times')
        # self.addParameter(key='freqCal', string='Calibrate frequency', val=1, field='OTH',
        #                   tip="0 to not calibrate, 1 to calibrate")
        self.addParameter(key='unlock_orientation', string='Unlock image orientation', val=0, field='OTH',
                          tip='0: Images oriented according to standard. 1: Image raw orientation')
        self.acq = ismrmrd.Acquisition()
        self.img = ismrmrd.Image()
        self.header= ismrmrd.xsd.ismrmrdHeader() 
        
    # ******************************************************************************************************************
    # ******************************************************************************************************************
    # ******************************************************************************************************************


    def sequenceInfo(self):
        print("3D GRE sequence")
        print("Author: Dr. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain \n")

    def sequenceTime(self):
        nScans = self.mapVals['nScans']
        nPoints = np.array(self.mapVals['nPoints'])
        repetition_time = self.mapVals['repetition_time']
        return(nPoints[1]*nPoints[2]*repetition_time*1e-3*nScans/60)  # minutes, scanTime

    def sequenceRun(self, plotSeq=0, demo=False):
        init_gpa = False  # Starts the gpa
        self.demo = demo

        # Set the fov
        self.dfov = self.getFovDisplacement()
        self.dfov = self.dfov[self.axesOrientation]
        self.fov = self.fov[self.axesOrientation]

        # Check for used axes
        axesEnable = []
        for ii in range(3):
            if self.nPoints[ii] == 1:
                axesEnable.append(0)
            else:
                axesEnable.append(1)
        self.mapVals['axesEnable'] = axesEnable
        
        # Miscellaneous
        resolution = self.fov/self.nPoints
        rfExAmp = self.rfExFA / (self.rfExTime * 1e6 * hw.b1Efficiency) * np.pi / 180
        for ii in range(3):
            if self.nPoints[ii]==1: axesEnable[ii] = 0
        self.mapVals['resolution'] = resolution
        self.mapVals['grad_rise_time'] = hw.grad_rise_time
        self.mapVals['addRdPoints'] = hw.addRdPoints

        # Matrix size
        n_rd = self.nPoints[0]+2*hw.addRdPoints
        n_ph = self.nPoints[1]
        n_sl = self.nPoints[2]

        # par_acq_lines in case par_acq_lines = 0
        par_acq_lines = int(int(self.nPoints[2] * self.sl_fraction) - self.nPoints[2] / 2)
        self.mapVals['partial_acquisition'] = par_acq_lines

        # bw
        bw = self.nPoints[0] / self.acq_time * 1e-6  # MHz
        bw_ov = bw * hw.oversamplingFactor  # MHz
        sampling_period = 1 / bw_ov  # us

        # Check if dephasing grad time is ok
        max_deph_grad_time = self.echo_time-0.5*(self.rfExTime+self.rdGradTime)-3*hw.grad_rise_time
        if self.dephGradTime==0 or self.dephGradTime>max_deph_grad_time:
            self.dephGradTime = max_deph_grad_time*1

        # Max gradient amplitude
        rd_grad_amplitude = self.nPoints[0]/(hw.gammaB*self.fov[0]*self.acq_time)
        rd_deph_amplitude = -rd_grad_amplitude*(self.rdGradTime+hw.grad_rise_time)/(2*(self.dephGradTime+hw.grad_rise_time))
        ph_grad_amplitude = n_ph/(2*hw.gammaB*self.fov[1]*(self.dephGradTime+hw.grad_rise_time))*axesEnable[1]
        sl_grad_amplitude = n_sl/(2*hw.gammaB*self.fov[2]*(self.dephGradTime+hw.grad_rise_time))*axesEnable[2]
        self.mapVals['rd_grad_amplitude'] = rd_grad_amplitude
        self.mapVals['rd_deph_amplitude'] = rd_deph_amplitude
        self.mapVals['ph_grad_amplitude'] = ph_grad_amplitude
        self.mapVals['sl_grad_amplitude'] = sl_grad_amplitude

        # Phase and slice gradient vector
        ph_gradients = np.linspace(-ph_grad_amplitude,ph_grad_amplitude,num=n_ph,endpoint=False)
        sl_gradients = np.linspace(-sl_grad_amplitude,sl_grad_amplitude,num=n_sl,endpoint=False)
        self.mapVals['ph_gradients'] = ph_gradients
        self.mapVals['sl_gradients'] = sl_gradients

        # Now fix the number of slices to partailly acquired k-space
        if self.nPoints[2]==1:
            n_sl = 1
        else:
            n_sl = int(self.nPoints[2]/2)+par_acq_lines
        n_repetitions = n_ph*n_sl

        # Get the rotation matrix
        rot = self.getRotationMatrix()
        gradAmp = np.array([0.0, 0.0, 0.0])
        gradAmp[self.axesOrientation[0]] = 1
        gradAmp = np.reshape(gradAmp, (3, 1))
        result = np.dot(rot, gradAmp)

        print("Readout direction:")
        print(np.reshape(result, (1, 3)))

        # Initialize k-vectors
        k_ph_sl_xyz = np.ones((3, self.nPoints[0] * self.nPoints[1] * n_sl)) * hw.gammaB * (
                    self.dephGradTime + hw.grad_rise_time)
        k_rd_xyz = np.ones((3, self.nPoints[0] * self.nPoints[1] * n_sl)) * hw.gammaB
        
        # Changing time parameters to us
        self.rfExTime *= 1e6
        self.echo_time *= 1e6
        self.repetition_time *= 1e6
        grad_rise_time = hw.grad_rise_time*1e6
        self.dephGradTime *= 1e6
        self.rdGradTime *= 1e6
        self.acq_time *= 1e6
        self.mapVals['scanTime'] = n_repetitions*self.repetition_time*1e-6

        # Create sequence instructions
        def createSequence(ph_index=0, sl_index=0, ln_index=0, repe_index_global=0):
            repe_index = 0
            acq_points = 0
            orders = 0

            # check in case of dummy pulse filling the cache
            if(self.dummy_pulses>0 and n_rd*2>hw.maxRdPoints) or (self.dummy_pulses==0 and n_rd>hw.maxRdPoints):
                print('ERROR: Too many acquired points.')
                return 0

            # Set shimming
            self.iniSequence(20, self.shimming)

            # Run sequence batch
            while acq_points+n_rd<=hw.maxRdPoints and orders<=hw.maxOrders and repe_index_global<n_repetitions:
                # Initialize time
                t_ex = 20e3+self.repetition_time*repe_index

                # First I do a noise measurement
                if repe_index==0:
                    t0 = t_ex-4*self.acq_time
                    self.rxGate(t0, self.acq_time+2*hw.addRdPoints/bw)
                    acq_points += n_rd

                # Excitation pulse
                t0 = t_ex-hw.blkTime-self.rfExTime/2
                if self.mode==1 or self.mode==3: # rf spoiling
                    self.rfRecPulse(t0, self.rfExTime, rfExAmp * np.exp(1j * 117 * np.pi / 180 * repe_index))
                elif self.mode==4: # balanced
                    self.rfRecPulse(t0, self.rfExTime, rfExAmp * np.exp(1j * np.pi/2 * (1 + (-1) ** repe_index)))
                elif self.mode==0 or self.mode==2:
                    self.rfRecPulse(t0, self.rfExTime, rfExAmp)

                # Dephasing gradient
                if repe_index>=self.dummy_pulses:
                    grad_amp_deph = np.array([0.0, 0.0, 0.0])
                    grad_amp_deph[self.axesOrientation[0]] = rd_deph_amplitude
                    grad_amp_deph[self.axesOrientation[1]] = ph_gradients[ph_index]
                    grad_amp_deph[self.axesOrientation[2]] = sl_gradients[sl_index]
                    grad_amp_deph = np.dot(rot, np.reshape(grad_amp_deph, (3, 1)))
                    t0 = t_ex+self.rfExTime/2-hw.gradDelay
                    self.gradTrap(t0, grad_rise_time, self.dephGradTime, grad_amp_deph[0], hw.grad_steps, 0, self.shimming)
                    self.gradTrap(t0, grad_rise_time, self.dephGradTime, grad_amp_deph[1], hw.grad_steps, 1, self.shimming)
                    self.gradTrap(t0, grad_rise_time, self.dephGradTime, grad_amp_deph[2], hw.grad_steps, 2, self.shimming)
                    orders = orders+hw.grad_steps*6
                    # get k-point
                    k_ph_sl_xyz[:, self.nPoints[0] * ln_index:self.nPoints[0] * (ln_index + 1)] = \
                        np.diag(np.reshape(grad_amp_deph, -1)) @ \
                        k_ph_sl_xyz[:, self.nPoints[0] * ln_index:self.nPoints[0] * (ln_index + 1)]

                # Rephasing readout gradient
                if repe_index>=self.dummy_pulses:
                    grad_amp_reph = np.array([0.0, 0.0, 0.0])
                    grad_amp_reph[self.axesOrientation[0]] = rd_grad_amplitude
                    grad_amp_reph = np.dot(rot, np.reshape(grad_amp_reph, (3, 1)))
                    t0 = t_ex + self.echo_time - self.rdGradTime / 2 - grad_rise_time - hw.gradDelay
                    orders = orders + hw.grad_steps * 6
                    if self.mode==0 or self.mode==1 or self.mode==4: # normal, only rf spoiler, or balanced
                        rd_grad_time = self.rdGradTime
                    elif self.mode==2 or self.mode==3: # gradient spoiler
                        rd_grad_time = 0.5*(self.rdGradTime-grad_rise_time)+self.acq_time*self.spoiler_order
                    self.gradTrap(t0, grad_rise_time, rd_grad_time, grad_amp_reph[0], hw.grad_steps, 0,
                                  self.shimming)
                    self.gradTrap(t0, grad_rise_time, rd_grad_time, grad_amp_reph[1], hw.grad_steps, 1,
                                  self.shimming)
                    self.gradTrap(t0, grad_rise_time, rd_grad_time, grad_amp_reph[2], hw.grad_steps, 2,
                                  self.shimming)

                # Rx gate
                if repe_index>=self.dummy_pulses:
                    t0 = t_ex+self.echo_time-self.acq_time/2-hw.addRdPoints/bw
                    self.rxGate(t0, self.acq_time+2*hw.addRdPoints/bw)
                    acq_points += n_rd
                    # get k-point
                    k_rd_xyz[:, self.nPoints[0] * ln_index:self.nPoints[0] * (ln_index + 1)] = \
                        np.diag(np.reshape(gradAmp, -1)) @ \
                        k_rd_xyz[:, self.nPoints[0] * ln_index:self.nPoints[0] * (ln_index + 1)] @ \
                        np.diag(self.time_vector)
                
                # gradient balance
                if repe_index>=self.dummy_pulses and (self.mode==2 or self.mode==3 or self.mode==4):
                    grad_amp_bala = np.array([0.0, 0.0, 0.0])
                    if self.mode==4: # bSSFP
                        grad_amp_bala[self.axesOrientation[0]] = rd_deph_amplitude
                    grad_amp_bala[self.axesOrientation[1]] = -ph_gradients[ph_index]
                    grad_amp_bala[self.axesOrientation[2]] = -sl_gradients[sl_index]
                    grad_amp_bala = np.dot(rot, np.reshape(grad_amp_bala, (3, 1)))
                    t0 = t_ex + self.echo_time - self.rdGradTime / 2 + rd_grad_time + grad_rise_time - hw.gradDelay
                    self.gradTrap(t0, grad_rise_time, self.dephGradTime, grad_amp_bala[0], hw.grad_steps, 0,
                                  self.shimming)
                    self.gradTrap(t0, grad_rise_time, self.dephGradTime, grad_amp_bala[1], hw.grad_steps, 1,
                                  self.shimming)
                    self.gradTrap(t0, grad_rise_time, self.dephGradTime, grad_amp_bala[2], hw.grad_steps, 2,
                                  self.shimming)

                # Update the phase and slice gradient
                if repe_index>=self.dummy_pulses:
                    ln_index += 1
                    if ph_index == n_ph-1:
                        ph_index = 0
                        sl_index += 1
                    else:
                        ph_index += 1

                if repe_index>=self.dummy_pulses: repe_index_global += 1 # Update the global repe_index
                repe_index+=1 # Update the repe_index after the ETL

            # Turn off the gradients after the end of the batch
            self.endSequence(repe_index*self.repetition_time)

            # Return the output variables
            return(ph_index, sl_index, ln_index, repe_index_global, acq_points)

        # Calibrate frequency
        # if self.freqCal and (not plotSeq) and (not self.demo):
        #     hw.larmorFreq = self.freqCalibration(bw=0.05)
        #     hw.larmorFreq = self.freqCalibration(bw=0.005)
        self.mapVals['larmorFreq'] = hw.larmorFreq

        # Initialize the experiment
        data_full = []
        over_data = []
        noise = []
        n_batches = 0
        repe_index_array = np.array([0])
        repe_index_global = repe_index_array[0]
        ph_index = 0
        sl_index = 0
        ln_index = 0
        acq_points_per_batch = []
        while repe_index_global<n_repetitions:
            n_batches += 1
            if not demo:
                self.expt = ex.Experiment(lo_freq=hw.larmorFreq + self.freqOffset * 1e-6, rx_t=sampling_period,
                                          init_gpa=init_gpa, gpa_fhdo_offset_time=(1 / 0.2 / 3.1))
                sampling_period = self.expt.get_rx_ts()[0]
                bw = 1/sampling_period/hw.oversamplingFactor # MHz
            
            # Time vector for main points
            self.time_vector = np.linspace(-self.nPoints[0] / bw / 2 + 0.5 / bw, self.nPoints[0] / bw / 2 - 0.5 / bw,
                                           self.nPoints[0]) * 1e-6  # s

            self. acq_time = self.nPoints[0]/bw # us
            self.mapVals['bw'] = bw
            ph_index, sl_index, ln_index, repe_index_global, aa = createSequence(ph_index=ph_index,
                                                                                 sl_index=sl_index,
                                                                                 ln_index=ln_index,
                                                                                 repe_index_global=repe_index_global)

            # Save instructions into MaRCoS if not a demo
            if not self.demo:
                if self.floDict2Exp(rewrite=n_batches == 1):
                    print("Sequence waveforms loaded successfully")
                    pass
                else:
                    print("ERROR: sequence waveforms out of hardware bounds")
                    return False
            
            repe_index_array = np.concatenate((repe_index_array, np.array([repe_index_global-1])), axis=0)
            acq_points_per_batch.append(aa)
            
            # Execute current batch nScans times
            for ii in range(self.nScans):
                if not plotSeq:
                    print("Batch %i, scan %i running..." % (n_batches, ii+1))
                    if not self.demo:
                        acq_points = 0
                        while acq_points != (aa * hw.oversamplingFactor):
                            rxd, msgs = self.expt.run()
                            rxd['rx0'] = rxd['rx0'] * hw.adcFactor  # Here I normalize to get the result in mV
                            acq_points = np.size(rxd['rx0'])
                            print("Acquired points = %i" % acq_points)
                            print("Expected points = %i" % (aa * hw.oversamplingFactor))
                        print("Batch %i, scan %i ready!")
                    else:
                        rxd = {}
                        rxd['rx0'] = np.random.randn(aa * hw.oversamplingFactor) + 1j * np.random.randn(
                            aa * hw.oversamplingFactor)
                        print("Batch %i, scan %i ready!" % (n_batches, ii + 1))
                    # Get noise data
                    noise = np.concatenate((noise, rxd['rx0'][0:n_rd * hw.oversamplingFactor]), axis=0)
                    rxd['rx0'] = rxd['rx0'][n_rd * hw.oversamplingFactor::]
                    # Get data
                    over_data = np.concatenate((over_data, rxd['rx0']), axis=0)
            if not demo: self.expt.__del__()
        del aa

        if not plotSeq:
            acq_points_per_batch = (acq_points_per_batch-n_rd)*self.nScans
            self.mapVals['noise_data'] = noise
            self.mapVals['over_data'] = over_data

            # Generate data_full
            data_full = sig.decimate(over_data, hw.oversamplingFactor, ftype='fir', zero_phase=True) ##size 4800 = 60*(60+2*addrdpoints)
            if n_batches>1:
                data_full_a = data_full[0:sum(acq_points_per_batch[0:-1])]
                data_full_b = data_full[sum(acq_points_per_batch[0:-1])::]
                
                        
            # Subtract phase in case of rf spoling or balanced
            if n_batches>1:
                if self.mode==1 or self.mode==3:  # rf spoiling
                    data_full_a = np.reshape(data_full_a, ((n_batches - 1) * self.nScans, -1, n_rd))
                    for repe_index in range(np.size(data_full_a, 1)):
                        data_full_a[:, repe_index, :] *= np.exp(-1j * 117 * np.pi / 180 * repe_index)
                    data_full_a = np.reshape(data_full_a, -1)
                    data_full_b = np.reshape(data_full_b, (self.nScans, -1, n_rd))
                    for repe_index in range(np.size(data_full_b, 1)):
                        data_full_b[:, repe_index, :] *= np.exp(-1j * 117 * np.pi / 180 * repe_index)
                    data_full_b = np.reshape(data_full_b, -1)
                if self.mode==4:
                    data_full_a = np.reshape(data_full_a, ((n_batches - 1) * self.nScans, -1, n_rd))
                    for repe_index in range(np.size(data_full_a, 1)):
                        data_full_a[:, repe_index, :] *= np.exp(-1j * np.pi / 2 *(1 + (-1) ** repe_index))
                    data_full_a = np.reshape(data_full_a, -1)
                    data_full_b = np.reshape(data_full_b, ((n_batches - 1) * self.nScans, -1, n_rd))
                    for repe_index in range(np.size(data_full_b, 1)):
                        data_full_b[:, repe_index, :] *= np.exp(-1j * np.pi / 2 * (1 + (-1) ** repe_index))
                    data_full_b = np.reshape(data_full_b, -1)
            else:
                if self.mode==1 or self.mode==3:  # rf spoiling
                    data_full = np.reshape(data_full, (n_batches * self.nScans, -1, n_rd))
                    for repe_index in range(np.size(data_full, 1)):
                        data_full[:, repe_index, :] *= np.exp(-1j * 117 * np.pi / 180 * repe_index)
                    data_full = np.reshape(data_full, -1)
                if self.mode==4:
                    data_full = np.reshape(data_full, (n_batches * self.nScans, -1, n_rd))
                    for repe_index in range(np.size(data_full, 1)):
                        data_full[:, repe_index, :] *= np.exp(-1j * np.pi / 2 * (1 + (-1) ** repe_index))
                    data_full = np.reshape(data_full, -1)

            # Reorganize data_full
            data_prov = np.zeros([self.nScans,n_sl*n_ph*n_rd], dtype=complex)
            if n_batches>1:
                data_full_a = np.reshape(data_full_a, (n_batches-1, self.nScans, -1, n_rd))
                data_full_b = np.reshape(data_full_b, (1, self.nScans, -1, n_rd))
            else:
                data_full = np.reshape(data_full, (n_batches, self.nScans, -1, n_rd))
            for scan in range(self.nScans):
                if n_batches>1:
                    data_prov[scan, :] = np.concatenate((np.reshape(data_full_a[:,ii,:,:],-1), np.reshape(data_full_b[:,ii,:,:],-1)), axis=0)
                else:
                    data_prov[scan, :] = np.reshape(data_full[:,scan,:,:],-1)
                    
            # Save data_full to save it in .h5
            self.data_full_mat=data_full
            
            
            data_full = np.reshape(data_prov,-1)

            # Get index for krd = 0
            # Average data
            data_prov = np.reshape(data_full, (self.nScans, n_rd*n_ph*n_sl))
            data_prov = np.average(data_prov, axis=0)
            data_prov = np.reshape(data_prov, (n_sl, n_ph, n_rd))
            # Check where is krd = 0
            data_prov = data_prov[int(self.nPoints[2]/2), int(n_ph/2), :]
            indkrd0 = np.argmax(np.abs(data_prov))
            if indkrd0 < n_rd/2-hw.addRdPoints or indkrd0 > n_rd/2+hw.addRdPoints:
                indkrd0 = int(n_rd/2)

            # Get individual images
            data_full = np.reshape(data_full, (self.nScans, n_sl, n_ph, n_rd))
            data_full = data_full[:, :, :, indkrd0-int(self.nPoints[0]/2):indkrd0+int(self.nPoints[0]/2)]
            img_full = data_full*0
            for scan in range(self.nScans):
                img_full[scan, :, :, :] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data_full[scan, :, :, :])))
            self.mapVals['data_full'] = data_full
            self.mapVals['img_full'] = img_full

            # Average data
            data = np.average(data_full, axis=0)

            # Concatenate with k_xyz
            k_xyz = k_ph_sl_xyz + k_rd_xyz
            self.mapVals['sampled_xyz'] = np.concatenate((k_xyz.T, np.reshape(data, (n_sl * n_ph * self.nPoints[0], 1))),
                                                         axis=1)
            data = np.reshape(data, (n_sl, n_ph, self.nPoints[0]))

            # Do zero padding
            dataTemp = np.zeros((self.nPoints[2], self.nPoints[1], self.nPoints[0]), dtype=complex)
            dataTemp[0:n_sl, :, :] = data
            data = np.reshape(dataTemp, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))

            # Fix the position of the sample according to dfov
            kMax = np.array(self.nPoints)/(2*np.array(self.fov))*np.array(axesEnable)
            k_rd = self.time_vector*hw.gammaB*rd_grad_amplitude
            k_ph = np.linspace(-kMax[1],kMax[1],num=self.nPoints[1],endpoint=False)
            k_sl = np.linspace(-kMax[2],kMax[2],num=self.nPoints[2],endpoint=False)
            k_ph, k_sl, k_rd = np.meshgrid(k_ph, k_sl, k_rd)
            k_rd = np.reshape(k_rd, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))
            k_ph = np.reshape(k_ph, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))
            k_sl = np.reshape(k_sl, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))
            d_phase = np.exp(-2*np.pi*1j*(self.dfov[0]*k_rd+self.dfov[1]*k_ph+self.dfov[2]*k_sl))
            data = np.reshape(data*d_phase, (self.nPoints[2], self.nPoints[1], self.nPoints[0]))
            self.mapVals['kSpace3D'] = data
            img=np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
            self.mapVals['image3D'] = img
            data = np.reshape(data, (1, self.nPoints[0]*self.nPoints[1]*self.nPoints[2]))

            # Create sampled data
            k_rd = np.reshape(k_rd, (self.nPoints[0]*self.nPoints[1]*self.nPoints[2], 1))
            k_ph = np.reshape(k_ph, (self.nPoints[0]*self.nPoints[1]*self.nPoints[2], 1))
            k_sl = np.reshape(k_sl, (self.nPoints[0]*self.nPoints[1]*self.nPoints[2], 1))
            data = np.reshape(data, (self.nPoints[0]*self.nPoints[1]*self.nPoints[2], 1))
            self.mapVals['kMax'] = kMax
            self.mapVals['sampled'] = np.concatenate((k_rd, k_ph, k_sl, data), axis=1)
            self.mapVals['sampledCartesian'] = self.mapVals['sampled']
            data = np.reshape(data, (self.nPoints[2], self.nPoints[1], self.nPoints[0]))

        return True

    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        axesEnable = self.mapVals['axesEnable']
        # Get axes in strings
        axes_dict = {'x': 0, 'y': 1, 'z': 2}
        axes_keys = list(axes_dict.keys())
        axes_vals = list(axes_dict.values())
        axes_str = ['', '', '']
        n = 0
        for val in self.axesOrientation:
            index = axes_vals.index(val)
            axes_str[n] = axes_keys[index]
            n += 1

        if (axesEnable[1] == 0 and axesEnable[2] == 0):
            bw = self.mapVals['bw']*1e-3 # kHz
            t_vector = np.linspace(-self.acq_time/2, self.acq_time/2, self.nPoints[0])*1e-3 # ms
            s_vector = self.mapVals['sampled'][:, 3]
            f_vector = np.linspace(-bw/2, bw/2, self.nPoints[0])
            i_vector = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(s_vector)))
            result1 = {'widget': 'curve',
                       'xData': t_vector,
                       'yData': [np.abs(s_vector), np.real(s_vector), np.imag(s_vector)],
                       'xLabel': 'Time (ms)',
                       'yLabel': "Signal amplitude (mV)",
                       'title': "Signal",
                       'legend': ['Magnitude', 'Real', 'Imaginary'],
                       'row': 0,
                       'col': 0}

            result2 = {'widget': 'curve',
                       'xData': f_vector,
                       'yData': [np.abs(i_vector)],
                       'xLabel': "Frequency (kHz)",
                       'yLabel': "Amplitude (a.u.)",
                       'title': "Spectrum",
                       'legend': ['Spectrum magnitude'],
                       'row': 1,
                       'col': 0}

            self.output = [result1, result2]

        else:
            # Plot image
            image = np.abs(self.mapVals['image3D'])
            image = image / np.max(np.reshape(image, -1)) * 100

            if not self.unlock_orientation:  # Image orientation
                if self.axesOrientation[2] == 2:  # Sagittal
                    title = "Sagittal"
                    if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 1:  # OK
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        xLabel = "(-Y) A | PHASE | P (+Y)"
                        yLabel = "(-X) I | READOUT | S (+X)"
                    else:
                        image = np.transpose(image, (0, 2, 1))
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        xLabel = "(-Y) A | READOUT | P (+Y)"
                        yLabel = "(-X) I | PHASE | S (+X)"
                elif self.axesOrientation[2] == 1:  # Coronal
                    title = "Coronal"
                    if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 2:  # OK
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        image = np.flip(image, axis=0)
                        xLabel = "(+Z) R | PHASE | L (-Z)"
                        yLabel = "(-X) I | READOUT | S (+X)"
                    else:
                        image = np.transpose(image, (0, 2, 1))
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        image = np.flip(image, axis=0)
                        xLabel = "(+Z) R | READOUT | L (-Z)"
                        yLabel = "(-X) I | PHASE | S (+X)"
                elif self.axesOrientation[2] == 0:  # Transversal
                    title = "Transversal"
                    if self.axesOrientation[0] == 1 and self.axesOrientation[1] == 2:
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        xLabel = "(+Z) R | PHASE | L (-Z)"
                        yLabel = "(+Y) P | READOUT | A (-Y)"
                    else:  # OK
                        image = np.transpose(image, (0, 2, 1))
                        image = np.flip(image, axis=2)
                        image = np.flip(image, axis=1)
                        xLabel = "(+Z) R | READOUT | L (-Z)"
                        yLabel = "(+Y) P | PHASE | A (-Y)"
            else:
                xLabel = "%s axis" % axesStr[1]
                yLabel = "%s axis" % axesStr[0]
                title = "Image"

            result1 = {}
            result1['widget'] = 'image'
            result1['data'] = image
            result1['xLabel'] = xLabel
            result1['yLabel'] = yLabel
            result1['title'] = title
            result1['row'] = 0
            result1['col'] = 0

            result2 = {}
            result2['widget'] = 'image'
            if self.sl_fraction == 1:
                result2['data'] = np.log10(np.abs(self.mapVals['kSpace3D']))
            else:
                result2['data'] = np.abs(self.mapVals['kSpace3D'])
            result2['xLabel'] = "k%s" % axes_str[1]
            result2['yLabel'] = "k%s" % axes_str[0]
            result2['title'] = "k-Space"
            result2['row'] = 0
            result2['col'] = 1

            # Reset rotation angle and dfov to zero
            self.mapVals['angle'] = 0.0
            self.mapVals['dfov'] = [0.0, 0.0, 0.0]
            hw.dfov = [0.0, 0.0, 0.0]

            # DICOM TAGS
            # Image
            imageDICOM = np.transpose(image, (0, 2, 1))
            # If it is a 3d image
            if len(imageDICOM.shape) > 2:
                # Obtener dimensiones
                slices, rows, columns = imageDICOM.shape
                self.meta_data["Columns"] = columns
                self.meta_data["Rows"] = rows
                self.meta_data["NumberOfSlices"] = slices
                self.meta_data["NumberOfFrames"] = slices
            # if it is a 2d image
            else:
                # Obtener dimensiones
                rows, columns = imageDICOM.shape
                self.meta_data["Columns"] = columns
                self.meta_data["Rows"] = rows
                self.meta_data["NumberOfSlices"] = 1
                self.meta_data["NumberOfFrames"] = 1
            imgAbs = np.abs(imageDICOM)
            imgFullAbs = np.abs(imageDICOM) * (2 ** 15 - 1) / np.amax(np.abs(imageDICOM))
            x2 = np.amax(np.abs(imageDICOM))
            imgFullInt = np.int16(np.abs(imgFullAbs))
            imgFullInt = np.reshape(imgFullInt, (slices, rows, columns))
            arr = np.zeros((slices, rows, columns), dtype=np.int16)
            arr = imgFullInt
            self.meta_data["PixelData"] = arr.tobytes()
            self.meta_data["WindowWidth"] = 26373
            self.meta_data["WindowCenter"] = 13194
            # Sequence parameters
            self.meta_data["RepetitionTime"] = self.mapVals['repetition_time']
            self.meta_data["EchoTime"] = self.mapVals['echo_time']

            # Add results into the output attribute (result1 must be the image to save in dicom)
            self.output = [result1, result2]

        # Save results
        self.saveRawData()
        self.save_ismrmrd()

        if self.mode == 'Standalone':
            self.plotResults()
            
        return self.output
       
       
    def save_ismrmrd(self):
        """
        Save the current instance's data in ISMRMRD format.

        This method saves the raw data, header information, and reconstructed images to an HDF5 file
        using the ISMRMRD (Image Storage and Reconstruction format for MR Data) format.

        Steps performed:
        1. Generate a timestamp-based filename and directory path for the output file.
        2. Initialize the ISMRMRD dataset with the generated path.
        3. Populate the header and write the XML header to the dataset. Informations can be added.
        4. Reshape the raw data matrix and iterate over scans, slices, and phases to write each acquisition. 
        5. Set acquisition flags and properties.
        6. Append the acquisition data to the dataset.
        7. Reshape and save the reconstructed images.
        8. Close the dataset.

        Attribute:
        - self.data_full_mat (numpy.array): Full matrix of raw data to be reshaped and saved.

        Returns:
        None. It creates an HDF5 file with the ISMRMRD format.
        """
        
        directory_rmd = self.directory_rmd
        name = datetime.datetime.now()
        name_string = name.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3]
        self.mapVals['name_string'] = name_string
        if hasattr(self, 'raw_data_name'):
            file_name = "%s.%s" % (self.raw_data_name, name_string)
        else:
            self.raw_data_name = self.mapVals['seqName']
            file_name = "%s.%s" % (self.mapVals['seqName'], name_string)
            
        path= "%s/%s.h5" % (directory_rmd, file_name)
        
        dset = ismrmrd.Dataset(path, f'/dataset', True) # Create the dataset
        
        addRdPoints = hw.addRdPoints
        nScans = self.mapVals['nScans']
        nPoints = np.array(self.mapVals['nPoints'])
        nRD = self.nPoints[0]
        nPH = self.nPoints[1]
        nSL = self.nPoints[2]
        nRep=nPH
        bw = self.mapVals['bw']
        
        axesOrientation = self.axesOrientation
        axesOrientation_list = axesOrientation.tolist()

        read_dir = [0, 0, 0]
        phase_dir = [0, 0, 0]
        slice_dir = [0, 0, 0]

        read_dir[axesOrientation_list.index(0)] = 1
        phase_dir[axesOrientation_list.index(1)] = 1
        slice_dir[axesOrientation_list.index(2)] = 1
        
       # Experimental Conditions field
        exp = ismrmrd.xsd.experimentalConditionsType() 
        magneticFieldStrength = hw.larmorFreq*1e6/hw.gammaB
        exp.H1resonanceFrequency_Hz = hw.larmorFreq

        self.header.experimentalConditions = exp 

        # Acquisition System Information field
        sys = ismrmrd.xsd.acquisitionSystemInformationType() 
        sys.receiverChannels = 1 
        self.header.acquisitionSystemInformation = sys


        # Encoding field can be filled if needed
        encoding = ismrmrd.xsd.encodingType()  
        encoding.trajectory = ismrmrd.xsd.trajectoryType.CARTESIAN
        
        # # encoded and recon spaces 
        # efov = ismrmrd.xsd.fieldOfViewMm()  
        # efov.x =  
        # efov.y =
        # efov.z = 

        # rfov = ismrmrd.xsd.fieldOfViewMm() 
        # rfov.x = 
        # rfov.y = 
        # rfov.z = 

        # ematrix = ismrmrd.xsd.matrixSizeType()
        # rmatrix = ismrmrd.xsd.matrixSizeType()

        # ematrix.x = 
        # ematrix.y = 
        # ematrix.z = 
        # rmatrix.x = 
        # rmatrix.y = 
        # rmatrix.z = 

        # espace = ismrmrd.xsd.encodingSpaceType() 
        # espace.matrixSize = ematrix
        # espace.fieldOfView_mm = efov
        # rspace = ismrmrd.xsd.encodingSpaceType()
        # rspace.matrixSize = rmatrix
        # rspace.fieldOfView_mm = rfov
        
        # encoding.encodedSpace = espace
        # encoding.reconSpace = rspace



        # Encoding limits field can be filled if needed
        
        # limits = ismrmrd.xsd.encodingLimitsType()
        # limits1 = ismrmrd.xsd.limitType()
        # limits1.minimum = 
        # limits1.center = 
        # limits1.maximum = 
        # limits.kspace_encoding_step_1 = limits1

        # limits_rep = ismrmrd.xsd.limitType()
        # limits_rep.minimum = 
        # limits_rep.center = 
        # limits_rep.maximum = 
        # limits.repetition = limits_rep

        # limits_rest = ismrmrd.xsd.limitType()
        # limits_rest.minimum = 
        # limits_rest.center = 
        # limits_rest.maximum =
        # limits.kspace_encoding_step_0 = 
        # limits.slice = 
        # limits.average =
        # limits.contrast =
        # limits.kspaceEncodingStep2 = 
        # limits.phase =  
        # limits.segment = 
        # limits.set =

        # encoding.encodingLimits = limits
        # self.header.encoding.append(encoding)

    
        dset.write_xml_header(self.header.toXML()) 
                
        
        self.data_full_mat = np.reshape(self.data_full_mat, (nScans, nSL, nPH, nRD + 2 * addRdPoints))
        counter = 0
        for average in range(nScans):
            repetition=0  
            for slice_idx in range(nSL):
                for phase_idx in range(nPH):
                        # Extraire la ligne correspondante
                        line = self.data_full_mat[average, slice_idx, phase_idx, :]
                                                
                        # Préparer les données complexes
                        line2d = np.reshape(line, (1, nRD+2*addRdPoints))
                        acq = ismrmrd.Acquisition.from_array(line2d, None)
                        
                        counter += 1
                        repetition += 1 ##verifier
                    
                        acq.idx.repetition = repetition
                        acq.idx.kspace_encode_step_1 = phase_idx + 1
                        acq.idx.slice = slice_idx + 1
                        acq.idx.average = average + 1
                        
                        acq.clearAllFlags()
                        
                        if phase_idx == 0:
                            acq.setFlag(ismrmrd.ACQ_FIRST_IN_PHASE)
                        elif phase_idx == nPH - 1:
                            acq.setFlag(ismrmrd.ACQ_LAST_IN_PHASE)
                        
                        if slice_idx == 0:
                            acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                        elif slice_idx == nSL - 1:
                            acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                        
                        if repetition == 1: 
                            acq.setFlag(ismrmrd.ACQ_FIRST_IN_AVERAGE)
                        elif repetition == nPH*nSL:
                            acq.setFlag(ismrmrd.ACQ_LAST_IN_AVERAGE)
                        
                        
                        acq.scan_counter = counter
                        acq.discard_pre = hw.addRdPoints
                        acq.discard_post = hw.addRdPoints
                        acq.sample_time_us = 1/bw
                        self.dfov = np.array(self.dfov)
                        acq.position = (ctypes.c_float * 3)(*self.dfov.flatten())
                        acq.read_dir = (ctypes.c_float * 3)(*read_dir) 
                        acq.phase_dir = (ctypes.c_float * 3)(*phase_dir)
                        acq.slice_dir = (ctypes.c_float * 3)(*slice_dir)
                        
                        
                        # Ajouter l'acquisition au dataset
                        dset.append_acquisition(acq)
                        
                        
        image=self.mapVals['image3D']
        image_reshaped = np.reshape(image, (nSL, nPH, nRD))
        
        #for scan in range (nScans): ## image3d does not have scan dimension
        for slice_idx in range (nSL):
            
            image_slice = image_reshaped[slice_idx, :, :]
            img = ismrmrd.Image.from_array(image_slice)
            
            
            img.field_of_view = (ctypes.c_float * 3)(*(self.fov)*10) # mm
            img.position = (ctypes.c_float * 3)(*self.dfov)
            img.sample_time_us = 1/bw
            img.image_type = 5 ## COMPLEX
            
            
            
            img.read_dir = (ctypes.c_float * 3)(*read_dir)
            img.phase_dir = (ctypes.c_float * 3)(*phase_dir)
            img.slice_dir = (ctypes.c_float * 3)(*slice_dir)


            
            dset.append_image(f"image_raw", img)
            
        
        dset.close()    

        
if __name__ == '__main__':
    seq = GRE3D()
    seq.sequenceAtributes()
    
    seq.sequenceRun(demo=True, plotSeq=False)
    seq.sequencePlot(standalone=True)
    
    
    seq.sequenceRun(demo=True)
    seq.sequenceAnalysis(mode='Standalone')
