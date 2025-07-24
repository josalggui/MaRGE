"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: mri blank sequence with common methods that will be inherited by any sequence
"""

import os

import bm4d
import numpy as np

import marge.configs.hw_config as hw
from datetime import date, datetime
from scipy.io import savemat, loadmat
import marge.controller.experiment_gui as ex
import scipy.signal as sig
import csv
import ismrmrd
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks
from skimage.measure import shannon_entropy

# Import dicom saver
from marge.manager.dicommanager import DICOMImage
from marge.marge_utils import utils
import shutil
import nibabel as nib

class MRIBLANKSEQ:
    """
    Class for representing MRI sequences.

    This class provides functionality for creating and managing MRI sequences. It includes methods for setting
    parameters, generating sequences, processing data, and plotting results.

    Attributes:
        mapKeys (list): Keys for the maps.
        mapNmspc (dict): Name to show in the GUI.
        mapVals (dict): Values to show in the GUI.
        mapFields (dict): Fields to classify the input parameters.
        mapLen (dict): Length of the input values.
        mapTips (dict): Tips for the input parameters.
        map_units (dict): Units for the input parameters.
        meta_data (dict): Dictionary to save meta data for DICOM files.
        rotations (list): List of rotation matrices.
        dfovs (list): List of displacement field of views.
        fovs (list): List of field of views.
        session (dict): Session information.
        demo (bool): Demo information.
        mode (string): Mode information for 'Standalone' execution.
        flo_dict (dict): Dictionary containing sequence waveforms.

    """

    def __init__(self):
        """
        Constructor method for initializing the MRIBLANKSEQ class instance.

        This method initializes the instance attributes.
        """
        self.mapKeys = []
        self.mapNmspc = {}
        self.mapVals = {}
        self.mapFields = {}
        self.mapLen = {}
        self.mapTips = {}
        self.map_units = {}
        self.meta_data = {}
        self.rotations = hw.rotations
        self.dfovs = hw.dfovs
        self.fovs = hw.fovs
        self.session = {}
        self.demo = None
        self.mode = None
        self.output=[]
        self.raw_data_name="raw_data"
        self.flo_dict = {'g0': [[],[]],
                         'g1': [[],[]],
                         'g2': [[],[]],
                         'rx0': [[],[]],
                         'rx1': [[],[]],
                         'tx0': [[],[]],
                         'tx1': [[],[]],
                         'ttl0': [[],[]],
                         'ttl1': [[],[]],}

        self.addParameter(key='seqName', val='blankSeq')
        self.addParameter(key='angle', val=0)
        self.addParameter(key='rotationAxis', val=[0, 0, 1])
        self.addParameter(key='dfov', val=[0.0, 0.0, 0.0])
        self.addParameter(key='fov', val=[0.0, 0.0, 0.0])
        self.addParameter(key='pypulseq', val=False)

    # *********************************************************************************
    # *********************************************************************************
    # *********************************************************************************

    # Create dictionaries of inputs classified by field (RF, SEQ, IM or OTH)

    @property
    def RFproperties(self):
        """
        Retrieve RF-related properties.

        Automatically selects the inputs related to RF fields and returns them along with their corresponding tips.

        Returns:
            dict: A dictionary containing RF-related properties.
            dict: A dictionary containing tips for RF-related properties.
        """
        out = {}
        tips = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'RF':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
                tips[self.mapNmspc[key]] = [self.mapTips[key]]
        return out, tips

    @property
    def IMproperties(self) -> dict:
        """
        Retrieve IM-related properties.

        Automatically selects the inputs related to IM fields and returns them along with their corresponding tips.

        Returns:
            dict: A dictionary containing IM-related properties.
            dict: A dictionary containing tips for IM-related properties.
        """
        out = {}
        tips = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'IM':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
                tips[self.mapNmspc[key]] = [self.mapTips[key]]
        return out, tips

    @property
    def SEQproperties(self) -> dict:
        """
        Retrieve SEQ-related properties.

        Automatically selects the inputs related to SEQ fields and returns them along with their corresponding tips.

        Returns:
            dict: A dictionary containing SEQ-related properties.
            dict: A dictionary containing tips for SEQ-related properties.
        """
        out = {}
        tips = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'SEQ':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
                tips[self.mapNmspc[key]] = [self.mapTips[key]]
        return out, tips

    @property
    def OTHproperties(self) -> dict:
        """
        Retrieve OTH-related properties.

        Automatically selects the inputs related to OTH fields and returns them along with their corresponding tips.

        Returns:
            dict: A dictionary containing OTH-related properties.
            dict: A dictionary containing tips for OTH-related properties.
        """
        out = {}
        tips = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'OTH':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
                tips[self.mapNmspc[key]] = [self.mapTips[key]]
        return out, tips

    @property
    def PROproperties(self) -> dict:
        """
        Retrieve PRO-related properties.

        Automatically selects the inputs related to PRO fields and returns them along with their corresponding tips.

        Returns:
            dict: A dictionary containing PRO-related properties.
            dict: A dictionary containing tips for PRO-related properties.
        """
        out = {}
        tips = {}
        for key in self.mapKeys:
            if self.mapFields[key] == 'PRO':
                out[self.mapNmspc[key]] = [self.mapVals[key]]
                tips[self.mapNmspc[key]] = [self.mapTips[key]]
        return out, tips

    def rotate_waveforms(self, waveforms):
        # Get the waveforms
        gx = waveforms['grad_vx']
        gy = waveforms['grad_vy']
        gz = waveforms['grad_vz']
        is_x = np.zeros_like(gx[0], dtype=int)
        is_y = np.zeros_like(gy[0], dtype=int) + 1
        is_z = np.zeros_like(gz[0], dtype=int) + 2

        # Concatenate arrays
        time = np.concatenate((gx[0], gy[0], gz[0]))
        ampl = np.concatenate((gx[1] * hw.gFactor[0], gy[1] * hw.gFactor[1], gz[1] * hw.gFactor[2]))  # mT/m
        is_a = np.concatenate((is_x, is_y, is_z))

        # Sort arrays
        idx = np.argsort(time)
        time = time[idx]
        ampl = ampl[idx]
        is_a = is_a[idx]

        # Define new gradient waveforms
        gx_new = [[], []]
        gy_new = [[], []]
        gz_new = [[], []]
        g_new = [[], [], []]

        # Populate new waveform
        w = []
        t = []
        step = 0
        n_steps = 0
        while step < len(time):
            g = [0., 0., 0.]

            # Add time
            gx_new[0].append(time[step])
            gy_new[0].append(time[step])
            gz_new[0].append(time[step])

            next = True
            while next:
                try:
                    # Get amplitude
                    g_new[is_a[step]].append(ampl[step])
                    if time[step + 1] != time[step]:
                        if step == 0:
                            if len(g_new[0]) == 0:
                                g_new[0].append(0.)
                            if len(g_new[0]) == 0:
                                g_new[1].append(0.)
                            if len(g_new[0]) == 0:
                                g_new[2].append(0.)
                        elif step > 0:
                            if len(g_new[0]) == n_steps:
                                g_new[0].append(g_new[0][-1])
                            if len(g_new[1]) == n_steps:
                                g_new[1].append(g_new[1][-1])
                            if len(g_new[2]) == n_steps:
                                g_new[2].append(g_new[2][-1])
                        n_steps += 1
                        next = False
                        gx_new[1].append(g_new[0][-1])
                        gy_new[1].append(g_new[1][-1])
                        gz_new[1].append(g_new[2][-1])
                except:
                    if step == 0:
                        if len(g_new[0]) == 0:
                            g_new[0].append(0.)
                        if len(g_new[0]) == 0:
                            g_new[1].append(0.)
                        if len(g_new[0]) == 0:
                            g_new[2].append(0.)
                    elif step > 0:
                        if len(g_new[0]) == n_steps:
                            g_new[0].append(g_new[0][-1])
                        if len(g_new[1]) == n_steps:
                            g_new[1].append(g_new[1][-1])
                        if len(g_new[2]) == n_steps:
                            g_new[2].append(g_new[2][-1])
                    n_steps += 1
                    next = False
                    gx_new[1].append(g_new[0][-1])
                    gy_new[1].append(g_new[1][-1])
                    gz_new[1].append(g_new[2][-1])
                step += 1
        g_new = np.array(g_new)

        # Rotate the waveforms
        rot = self.getRotationMatrix()
        for step in range(np.size(g_new, axis=1)):
            g_new[:, step] = np.dot(rot, g_new[:, step])
        gx_new[1] = list(g_new[0, :] / hw.gFactor[0])
        gy_new[1] = list(g_new[1, :] / hw.gFactor[1])
        gz_new[1] = list(g_new[2, :] / hw.gFactor[2])

        waveforms['grad_vx'] = gx_new
        waveforms['grad_vy'] = gy_new
        waveforms['grad_vz'] = gz_new

        # Delete last rotation/displacement if plot
        if self.plotSeq:
            self.fovs.pop()
            self.dfovs.pop()
            self.rotations.pop()

        return waveforms

    def runBatches(self, waveforms, n_readouts, n_adc,
                   frequency=hw.larmorFreq,
                   bandwidth=0.03,
                   decimate='Normal',
                   hardware=True,
                   output='',
                   channels=[0],
                   angulation=1,
                   ):
        """
        Execute multiple batches of MRI waveforms, manage data acquisition, and store oversampled data.

        Parameters:
        -----------
        waveforms : dict
            Dictionary containing waveform sequences. Keys represent batch identifiers, and values are
            the corresponding waveform data generated with PyPulseq.
        n_readouts : dict
            Dictionary specifying the number of readout points for each batch. Keys match the batch
            identifiers, and values indicate the number of readout points.
        n_adc : int
            Number of ADC windows. Each window must have the same length.
        frequency : float, optional
            Larmor frequency in MHz for the MRI acquisition. Defaults to the system's Larmor frequency (hw.larmorFreq).
        bandwidth : float, optional
            Bandwidth in MHz used to calculate the sampling period (sampling time = 1 / bandwidth). Defaults to 0.03 MHz.
        decimate : str, optional
            Specifies the decimation method.
            - 'Normal': Decimates the acquired array without preprocessing.
            - 'PETRA': Adjusts the pre-readout points to the desired starting point.
        hardware: bool, optional
            Take into account gradient and ADC delay.
        output: str, optional
            String to add to the output keys saved in the mapVals parameter.
        channels : list, optional
            List of channels used for Rx
        angulation : bool, optional
            Bool parameter to work with angulation (1) or without angulation (0)

        Returns:
        --------
        bool
            True if all batches are executed successfully, False if an error occurs (e.g., waveform constraints exceed hardware limits).

        Notes:
        ------
        - Initializes Red Pitaya hardware unless in demo mode.
        - Converts PyPulseq waveforms to Red Pitaya-compatible format.
        - If `plotSeq` is True, the sequence is plotted instead of executed.
        - In demo mode, simulated random data replaces hardware acquisition.
        - Oversampled data is stored in `self.mapVals['data_over']`.
        - Decimated data is stored in `self.mapVals['data_decimated']`.
        - Handles data loss by repeating batches until the expected points are acquired.
        """
        self.mapVals['n_readouts'] = list(n_readouts.values())
        self.mapVals['n_batches'] = len(n_readouts.values())

        # Initialize a list to hold oversampled data
        data_over = []

        # Iterate through each batch of waveforms
        for seq_num in waveforms.keys():
            # Rotate the waveforms to given reference system
            if angulation:
                waveforms[seq_num] = self.rotate_waveforms(waveforms[seq_num])

            # Initialize the experiment if not in demo mode
            if not self.demo:
                self.expt = ex.Experiment(
                    lo_freq=frequency,  # Larmor frequency in MHz
                    rx_t=1 / bandwidth,  # Sampling time in us
                    init_gpa=False,  # Whether to initialize GPA board (False for now)
                    gpa_fhdo_offset_time=(1 / 0.2 / 3.1),  # GPA offset time calculation
                    auto_leds=True  # Automatic control of LEDs
                )

            # Convert the PyPulseq waveform to the Red Pitaya compatible format
            self.pypulseq2mriblankseq(waveforms=waveforms[seq_num],
                                      shimming=self.shimming,
                                      sampling_period=1/bandwidth,
                                      hardware=hardware,
                                      channels=channels
                                      )

            # Load the waveforms into Red Pitaya
            if not self.floDict2Exp():
                print("ERROR: Sequence waveforms out of hardware bounds")
                return False
            else:
                print("Sequence waveforms loaded successfully")

            # If not plotting the sequence, start scanning
            if not self.plotSeq:
                for scan in range(self.nScans):
                    print(f"Scan {scan + 1}, batch {seq_num.split('_')[-1]}/{len(n_readouts)} running...")
                    acquired_points = 0
                    expected_points = n_readouts[seq_num] * hw.oversamplingFactor  # Expected number of points

                    # Continue acquiring points until we reach the expected number
                    while acquired_points != expected_points:
                        if not self.demo:
                            rxd, msgs = self.expt.run()  # Run the experiment and collect data
                        else:
                            # In demo mode, generate random data as a placeholder
                            rxd = {'rx0': np.random.randn(expected_points) + 1j * np.random.randn(expected_points)}

                        # Update acquired points
                        acquired_points = np.size(rxd['rx0'])

                        # Check if acquired points coincide with expected points
                        if acquired_points != expected_points:
                            print("WARNING: data apoints lost!")
                            print("Repeating batch...")

                    # Concatenate acquired data into the oversampled data array
                    data_over = np.concatenate((data_over, rxd['rx0']), axis=0)
                    print(f"Acquired points = {acquired_points}, Expected points = {expected_points}")
                    print(f"Scan {scan + 1}, batch {seq_num[-1]}/{len(n_readouts)} ready!")

                # Decimate the oversampled data and store it
                if output=='':
                    self.mapVals[f'data_over'] = data_over
                    data = self.decimate(data_over, n_adc=n_adc, option='Normal', remove=False)
                    self.mapVals[f'data_decimated'] = data
                else:
                    self.mapVals[f'data_over_{output}'] = data_over
                    data = self.decimate(data_over, n_adc=n_adc, option='Normal', remove=False)
                    self.mapVals[f'data_decimated_{output}'] = data

            elif self.plotSeq and self.standalone:
                # Plot the sequence if requested and return immediately
                self.sequencePlot(standalone=self.standalone)

            if not self.demo:
                self.expt.__del__()

        return True

    def sequenceInfo(self):
        print("sequenceInfo method is empty."
              "It is recommended to overide this method into your sequence.")

    def sequenceTime(self):
        print("sequenceTime method is empty."
             "It is recommended to overide this method into your sequence.")
        return 0

    def pypulseq2mriblankseq(self, waveforms=None,
                             shimming=np.array([0.0, 0.0, 0.0]),
                             sampling_period=0.0,
                             hardware=True,
                             channels=[0],
                             ):
        """
        Converts PyPulseq waveforms into a format compatible with MRI hardware.

        Parameters:
        -----------
        waveforms : dict, optional
            Dictionary containing PyPulseq waveforms. The keys represent waveform types (e.g., 'tx0', 'rx0_en',
            'grad_vx'), and values are arrays of time and amplitude pairs.
        shimming : numpy.ndarray, optional
            Array of three values representing the shimming currents to apply in the x, y, and z gradients, respectively.
            Defaults to [0.0, 0.0, 0.0].
        sampling_period : float, optional
            Sampling period in seconds, used to account for delays in the CIC filter. Defaults to 0.0.
        hardware: bool, optional
            Take into account gradient and ADC delay
        channels: list, optional
            List of channels used for Rx

        Returns:
        --------
        bool
            Returns True if the conversion is successful.

        Workflow:
        ---------
        1. **Reset flo_dict**:
            Initializes the flo dictionary, which stores gradient, RF, and TTL signals for MRI hardware execution.

        2. **Fill flo_dict**:
            Iterates through the input `waveforms` to populate the flo dictionary. Each key corresponds to a signal
            type, and the waveform data is appended.

        3. **Fill missing keys**:
            Ensures that all keys in `flo_dict` are populated, even if no data exists for certain signals. Unfilled
            keys are set to default arrays with zero values.

        4. **Apply shimming**:
            Adds the shimming values to the corresponding gradient channels (x, y, z).

        5. **Set sequence end**:
            Ensures all signals return to zero at the end of the sequence to finalize waveform execution.

        6. **Add hardware-specific corrections**:
            - Applies gradient latency adjustments.
            - Accounts for CIC filter delays in the receive (rx) signals.

        7. **Revalidate sequence end**:
            Reassesses and ensures all signal channels return to zero with a buffer period.

        Notes:
        ------
        - This method processes and validates input waveform data to ensure compatibility with MRI hardware.
        - Hardware-specific parameters such as gradient delay (`hw.gradDelay`) and CIC filter delay
          (`hw.cic_delay_points`) are applied.
        - Any signal not specified in `waveforms` is initialized with a default value of zero.

        """
        # Reset flo dictionary
        self.flo_dict = {'g0': [[], []],
                         'g1': [[], []],
                         'g2': [[], []],
                         'rx0': [[], []],
                         'rx1': [[], []],
                         'tx0': [[], []],
                         'tx1': [[], []],
                         'ttl0': [[], []],
                         'ttl1': [[], []], }

        # Fill dictionary
        for key in waveforms.keys():
            if key == 'tx0':
                self.flo_dict['tx0'][0] = np.concatenate((self.flo_dict['tx0'][0], waveforms['tx0'][0][0:-1]), axis=0)
                self.flo_dict['tx0'][1] = np.concatenate((self.flo_dict['tx0'][1], waveforms['tx0'][1][0:-1]), axis=0)
            elif key == 'tx1':
                self.flo_dict['tx1'][0] = np.concatenate((self.flo_dict['tx1'][0], waveforms['tx1'][0][0:-1]), axis=0)
                self.flo_dict['tx1'][1] = np.concatenate((self.flo_dict['tx1'][1], waveforms['tx1'][1][0:-1]), axis=0)
            elif key == 'rx0_en':
                self.flo_dict['rx0'][0] = np.concatenate((self.flo_dict['rx0'][0], waveforms['rx0_en'][0][0:-1]), axis=0)
                self.flo_dict['rx0'][1] = np.concatenate((self.flo_dict['rx0'][1], waveforms['rx0_en'][1][0:-1]), axis=0)
            elif key == 'rx1_en':
                self.flo_dict['rx1'][0] = np.concatenate((self.flo_dict['rx1'][0], waveforms['rx1_en'][0][0:-1]), axis=0)
                self.flo_dict['rx1'][1] = np.concatenate((self.flo_dict['rx1'][1], waveforms['rx1_en'][1][0:-1]), axis=0)
            elif key == 'tx_gate':
                self.flo_dict['ttl0'][0] = np.concatenate((self.flo_dict['ttl0'][0], waveforms['tx_gate'][0][0:-1]), axis=0)
                self.flo_dict['ttl0'][1] = np.concatenate((self.flo_dict['ttl0'][1], waveforms['tx_gate'][1][0:-1]), axis=0)
            elif key == 'rx_gate':
                self.flo_dict['ttl1'][0] = np.concatenate((self.flo_dict['ttl1'][0], waveforms['rx_gate'][0][0:-1]), axis=0)
                self.flo_dict['ttl1'][1] = np.concatenate((self.flo_dict['ttl1'][1], waveforms['rx_gate'][1][0:-1]), axis=0)
            elif key == 'grad_vx':
                self.flo_dict['g0'][0] = np.concatenate((self.flo_dict['g0'][0], waveforms['grad_vx'][0][0:-1]), axis=0)
                self.flo_dict['g0'][1] = np.concatenate((self.flo_dict['g0'][1], waveforms['grad_vx'][1][0:-1]), axis=0)
            elif key == 'grad_vy':
                self.flo_dict['g1'][0] = np.concatenate((self.flo_dict['g1'][0], waveforms['grad_vy'][0][0:-1]), axis=0)
                self.flo_dict['g1'][1] = np.concatenate((self.flo_dict['g1'][1], waveforms['grad_vy'][1][0:-1]), axis=0)
            elif key == 'grad_vz':
                self.flo_dict['g2'][0] = np.concatenate((self.flo_dict['g2'][0], waveforms['grad_vz'][0][0:-1]), axis=0)
                self.flo_dict['g2'][1] = np.concatenate((self.flo_dict['g2'][1], waveforms['grad_vz'][1][0:-1]), axis=0)

        # Fill missing keys
        for key in self.flo_dict.keys():
            try:
                is_unfilled = all(not sublist for sublist in self.flo_dict[key])
            except:
                is_unfilled = False
            if is_unfilled:
                self.flo_dict[key] = [np.array([0]), np.array([0])]

        # Add shimming
        self.flo_dict['g0'][1] = self.flo_dict['g0'][1] + shimming[0]
        self.flo_dict['g1'][1] = self.flo_dict['g1'][1] + shimming[1]
        self.flo_dict['g2'][1] = self.flo_dict['g2'][1] + shimming[2]

        # Set everything to zero
        last_times = np.array([value[0][-1] for value in self.flo_dict.values()])
        last_time = np.max(last_times)
        self.endSequence(last_time+10)

        # Add gradient latency and CIC filter delay
        if hardware:
            self.flo_dict['g0'][0][1::] -= hw.gradDelay
            self.flo_dict['g1'][0][1::] -= hw.gradDelay
            self.flo_dict['g2'][0][1::] -= hw.gradDelay
            self.flo_dict['rx0'][0][1::] += hw.cic_delay_points * sampling_period / hw.oversamplingFactor
            self.flo_dict['rx1'][0][1::] += hw.cic_delay_points * sampling_period / hw.oversamplingFactor

        # Set everything to zero (again)
        last_times = np.array([value[0][-1] for value in self.flo_dict.values()])
        last_time = np.max(last_times)
        self.endSequence(last_time + 10)

        return True

    def getFovDisplacement(self):
        """
        Get the displacement to apply in the FFT reconstruction.

        Returns:
            np.ndarray: The displacement vector.
        """

        def rotationMatrix(rotation):
            theta = - rotation[3]
            ux, uy, uz = rotation[:3]
            out = np.zeros((3, 3))
            out[0, 0] = np.cos(theta) + ux ** 2 * (1 - np.cos(theta))
            out[0, 1] = ux * uy * (1 - np.cos(theta)) - uz * np.sin(theta)
            out[0, 2] = ux * uz * (1 - np.cos(theta)) + uy * np.sin(theta)
            out[1, 0] = uy * ux * (1 - np.cos(theta)) + uz * np.sin(theta)
            out[1, 1] = np.cos(theta) + uy ** 2 * (1 - np.cos(theta))
            out[1, 2] = uy * uz * (1 - np.cos(theta)) - ux * np.sin(theta)
            out[2, 0] = uz * ux * (1 - np.cos(theta)) - uy * np.sin(theta)
            out[2, 1] = uz * uy * (1 - np.cos(theta)) + ux * np.sin(theta)
            out[2, 2] = np.cos(theta) + uz ** 2 * (1 - np.cos(theta))

            return out

        dr = np.zeros((3, 1))
        for ii in range(len(self.dfovs)):
            Mii = rotationMatrix(self.rotations[ii])
            rii = np.reshape(np.array(self.dfovs[ii]), (3, 1))
            dr = np.dot(Mii, (dr + rii))

        return dr

    def getRotationMatrix(self):
        """
        Get the rotation matrix to rotate through an arbitrary axis.

        Returns:
            np.ndarray: The rotation matrix.
        """

        def rotationMatrix(rotation):
            theta = rotation[3]  # * pi / 180 ? Check this
            ux, uy, uz = rotation[:3]
            out = np.zeros((3, 3))
            out[0, 0] = np.cos(theta) + ux ** 2 * (1 - np.cos(theta))
            out[0, 1] = ux * uy * (1 - np.cos(theta)) - uz * np.sin(theta)
            out[0, 2] = ux * uz * (1 - np.cos(theta)) + uy * np.sin(theta)
            out[1, 0] = uy * ux * (1 - np.cos(theta)) + uz * np.sin(theta)
            out[1, 1] = np.cos(theta) + uy ** 2 * (1 - np.cos(theta))
            out[1, 2] = uy * uz * (1 - np.cos(theta)) - ux * np.sin(theta)
            out[2, 0] = uz * ux * (1 - np.cos(theta)) - uy * np.sin(theta)
            out[2, 1] = uz * uy * (1 - np.cos(theta)) + ux * np.sin(theta)
            out[2, 2] = np.cos(theta) + uz ** 2 * (1 - np.cos(theta))

            return out

        rotations = [rotationMatrix(rotation) for rotation in self.rotations]
        rotation = rotations[-1]
        for M in rotations[:-1]:
            rotation = np.dot(M, rotation)

        return rotation

    def deleteOutput(self):
        """
        Delete the 'output' attribute from the instance if it exists.
        """
        if hasattr(self, 'output'):
            delattr(self, 'output')

    def saveParams(self):
        """
        Save the parameters in mapVals variable to a CSV file.

        This method performs the following steps:
            1. Resets the `mapVals` variable by calling the `resetMapVals` method. Then only input parameters are accessible.
            2. Ensures that the directory 'experiments/parameterization' exists, creating it if necessary.
            3. Writes the current parameter values stored in `mapVals` to a CSV file named '<seqName>_last_parameters.csv', where <seqName> is the value of the 'seqName' key in `mapVals`.

        The CSV file is saved in the 'experiments/parameterization' directory, and contains the header specified by `mapKeys`.

        Potential exceptions:
            KeyError: If 'seqName' is not a key in `mapVals`.

        Example:
            self.saveParams()

        """
        # Reset the mapVals variable
        self.resetMapVals()

        # Create directory if it does not exist
        if not os.path.exists('experiments/parameterization'):
            os.makedirs('experiments/parameterization')

        # Save csv file with mapVals
        with open('experiments/parameterization/%s_last_parameters.csv' % self.mapVals['seqName'], 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.mapKeys)
            writer.writeheader()
            writer.writerows([self.mapNmspc, self.mapVals])

    def loadParams(self, directory='experiments/parameterization', file=None): ## ca vient d un fichier ou directement de mon action ???
        """
        Load parameter values from a CSV file.

        This method loads parameter values into the `mapVals` attribute from a specified CSV file. The method first
        attempts to load the last saved parameters if no specific file is provided. If a file is provided, it loads
        parameters from the given file. The directory can be either 'experiments/parameterization' or 'calibration', or
        any other specified directory for protocol parameters.

        Args:
            directory (str): The directory where the CSV file is located. Defaults to 'experiments/parameterization'.
            file (str, optional): The specific CSV file to load. If not provided, the method loads the last saved parameters based on `seqName` in `mapVals`.

        Raises:
            KeyError: If 'seqName' is not found in `mapVals` when attempting to load the last parameters.
            FileNotFoundError: If the specified file does not exist in the given directory.

        Example:
            - self.loadParams()
            - self.loadParams(directory='calibration', file='calibration_parameters.csv')

        Notes:
            - This method updates the `mapVals` attribute with the new parameter values from the CSV file.
            - The method handles different data types (str, int, float) for each parameter key and ensures the correct
              type is maintained.
            - If a key is missing in the new parameter values, the old value is retained.

        """
        mapValsOld = self.mapVals
        try:
            if file is None:  # Load last parameters
                with open('%s/%s_last_parameters.csv' % (directory, self.mapVals['seqName']), 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for l in reader:
                        mapValsNew = l
            else:
                try:
                    if directory == 'calibration':  # Load parameters from calibration directory
                        with open('%s/%s_last_parameters.csv' % (directory, file), 'r') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for l in reader:
                                mapValsNew = l
                    else:  # Load parameters from protocol directory
                        with open('%s/%s' % (directory, file), 'r') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for l in reader:
                                mapValsNew = l
                except:
                    print("WARNING: File %s/%s does not exist" % (directory, file))
                    print("WARNING: File %s/%s loaded" % ("experiments/parameterization", self.mapVals['seqName']))
                    with open('%s/%s_last_parameters.csv' % ("experiments/parameterization", self.mapVals['seqName']),
                              'r') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for l in reader:
                            mapValsNew = l

            self.mapVals = {}

            # Get key for corresponding modified parameter
            for key in self.mapKeys:
                dataLen = self.mapLen[key]
                valOld = mapValsOld[key]
                try:
                    valNew = mapValsNew[key]
                except:
                    valNew = str(valOld)
                valNew = valNew.replace('[', '')
                valNew = valNew.replace(']', '')
                valNew = valNew.split(',')
                if type(valOld) == str:
                    valOld = [valOld]
                elif dataLen == 1:
                    valOld = [valOld]
                dataType = type(valOld[0])

                inputNum = []
                for ii in range(dataLen):
                    if dataType == float or dataType == np.float64:
                        try:
                            inputNum.append(float(valNew[ii]))
                        except:
                            inputNum.append(float(valOld[ii]))
                    elif dataType == int:
                        try:
                            inputNum.append(int(valNew[ii]))
                        except:
                            inputNum.append(int(valOld[ii]))
                    else:
                        try:
                            inputNum.append(str(valNew[0]))
                            break
                        except:
                            inputNum.append(str(valOld[0]))
                            break
                if dataType == str:
                    self.mapVals[key] = inputNum[0]
                else:
                    if dataLen == 1:  # Save value into mapVals
                        self.mapVals[key] = inputNum[0]
                    else:
                        self.mapVals[key] = inputNum
        except:
            self.mapVals = self.mapVals

    def resetMapVals(self):
        """
        Reset the `mapVals` attribute to contain only the original keys.

        This method creates a new dictionary (`mapVals2`) that includes only the original keys specified in `mapKeys`
        and retains their corresponding values from the current `mapVals`. This effectively resets `mapVals` to its
        initial state defined by `mapKeys`, discarding any additional keys that may have been added during execution.

        Example:
            - self.resetMapVals()

        Notes:
            - Any additional keys in `mapVals` that are not present in `mapKeys` will be removed.

        """
        mapVals2 = {}
        for key in self.mapKeys:
            mapVals2[key] = self.mapVals[key]
        self.mapVals = mapVals2

    def sequencePlot(self, standalone=False):
        """
        Plot the TX, gradient, RX, and digital I/O sequences.

        This method generates step data plots for TX channels, gradient channels, RX channels, and digital I/O channels
        based on the data stored in `flo_dict` or obtained from an experiment object. If `standalone` is set to True,
        the method will create and display the plots in a new figure window. Otherwise, it returns the plot data for
        further use.

        Args:
            standalone (bool): If True, creates and displays the plots in a new figure window. Defaults to False.

        Returns:
            list: A list of plot data, each element being a list containing:
                - xData: List of time values for the plot.
                - yData: List of amplitude values for the plot.
                - legend: List of legend labels for the plot.
                - title: Title of the plot.

        Example:
            - self.sequencePlot()
            - self.sequencePlot(standalone=True)

        Notes:
            - The method uses a nested function `getStepData` to generate step data for plotting.
            - The method handles different scenarios based on the `demo` attribute and data from `flo_dict` or an
              experiment object.
            - If `demo` is True, it plots the TX, gradient, and RX channels from `flo_dict`.
            - If `demo` is False, it plots the channels using data from an experiment object.
            - The method formats the time data in milliseconds (ms) for plotting.

        """

        def getStepData(data):
            t = data[0]
            s = data[1]
            n = np.size(t)
            tStep = np.zeros(2 * n - 1)
            sStep = np.zeros(2 * n - 1)
            tStep[0::2] = t
            tStep[1::2] = t[1::]
            sStep[0::2] = s
            sStep[1::2] = s[0:-1]
            return [tStep, sStep]

        # Plots
        if self.demo:
            # Plot tx channels
            xData = []
            yData = []
            legend = []

            # tx0_i
            x = self.flo_dict['tx0'][0]
            y = np.real(self.flo_dict['tx0'][1])
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('tx0_i')

            # tx0_q
            x = self.flo_dict['tx0'][0]
            y = np.imag(self.flo_dict['tx0'][1])
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('tx0_q')

            # tx1_i
            x = self.flo_dict['tx1'][0]
            y = np.real(self.flo_dict['tx1'][1])
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('tx1_i')

            # tx1_q
            x = self.flo_dict['tx1'][0]
            y = np.imag(self.flo_dict['tx1'][1])
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('tx1_q')

            plotTx = [xData, yData, legend, 'Tx gate']

            # Plot gradients
            xData = []
            yData = []
            legend = []

            # g0
            x = self.flo_dict['g0'][0]
            y = self.flo_dict['g0'][1]
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('g0')

            # g1
            x = self.flo_dict['g1'][0]
            y = self.flo_dict['g1'][1]
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('g1')

            # g2
            x = self.flo_dict['g2'][0]
            y = self.flo_dict['g2'][1]
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('g2')

            plotGrad = [xData, yData, legend, 'Gradients']

            # Plot readouts
            xData = []
            yData = []
            legend = []

            # rx0
            x = self.flo_dict['rx0'][0]
            y = self.flo_dict['rx0'][1]
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('rx0_en')

            # rx1
            x = self.flo_dict['rx1'][0]
            y = self.flo_dict['rx1'][1]
            data = [x, y]
            dataStep = getStepData(data)
            xData.append(dataStep[0] * 1e-3)
            yData.append(dataStep[1])
            legend.append('rx1_en')

            plotRx = [xData, yData, legend, 'Rx gate']

            outputs = [plotTx, plotGrad, plotRx]
        else:
            # Get instructions from experiment object
            fd = self.expt.get_flodict()

            # Plot tx channels
            xData = []
            yData = []
            legend = []
            for txl in ['tx0_i', 'tx0_q', 'tx1_i', 'tx1_q']:
                try:
                    dataStep = getStepData(fd[txl])
                    xData.append(dataStep[0] * 1e-3)
                    yData.append(dataStep[1])
                    legend.append(txl)
                except KeyError:
                    continue
            plotTx = [xData, yData, legend, 'Tx gate']

            # Plot gradient channels
            xData = []
            yData = []
            legend = []
            for gradl in self.expt.gradb.keys():
                try:
                    dataStep = getStepData(fd[gradl])
                    xData.append(dataStep[0] * 1e-3)
                    yData.append(dataStep[1])
                    legend.append(gradl)
                except KeyError:
                    continue
            plotGrad = [xData, yData, legend, 'Gradients']

            # Plot RX enable channels
            xData = []
            yData = []
            legend = []
            for rxl in ['rx0_en', 'rx1_en']:
                try:
                    dataStep = getStepData(fd[rxl])
                    xData.append(dataStep[0] * 1e-3)
                    yData.append(dataStep[1])
                    legend.append(rxl)
                except KeyError:
                    continue
            plotRx = [xData, yData, legend, 'Rx gate']

            # Plot digital outputs
            xData = []
            yData = []
            legend = []
            for iol in ['tx_gate', 'rx_gate', 'trig_out', 'leds']:
                try:
                    dataStep = getStepData(fd[iol])
                    xData.append(dataStep[0] * 1e-3)
                    yData.append(dataStep[1])
                    legend.append(iol)
                except KeyError:
                    continue
            plotDigital = [xData, yData, legend, 'Digital']

            outputs = [plotTx, plotGrad, plotRx, plotDigital]

        if standalone:
            # Create plot window
            fig, axes = plt.subplots(3, 1, figsize=(10, 5))

            # Insert plots
            plot = 0
            for item in outputs[0:3]:
                plt.subplot(3, 1, plot + 1)
                for ii in range(len(item[0])):
                    plt.plot(item[0][ii], item[1][ii], label=item[2][ii])
                plt.title(item[3])
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (a.u.)')
                plot += 1

            plt.tight_layout()
            plt.show()

        return outputs

    def getIndex(self, etl=1, nPH=1, sweepMode=1):
        """
        Generate an array representing the order to sweep the k-space phase lines along an echo train length.

        The method creates an 'ind' array based on the specified echo train length (ETL), number of phase encoding
        steps (nPH), and sweep mode. The sweep mode determines the order in which the k-space phase lines are traversed.

        Args:
            etl (int): Echo train length. Default is 1.
            nPH (int): Number of phase encoding steps. Default is 1.
            sweepMode (int): Sweep mode for k-space traversal. Default is 1.
                - 0: Sequential from -kMax to kMax (for T2 contrast).
                - 1: Center-out from 0 to kMax (for T1 or proton density contrast).
                - 2: Out-to-center from kMax to 0 (for T2 contrast).
                - 3: Niquist modulated method to reduce ghosting artifact (To be tested).

        Returns:
            numpy.ndarray: An array of indices representing the k-space phase line traversal order.

        """
        n2ETL = int(nPH / 2 / etl)
        ind = []
        if nPH == 1:
            ind = np.array([0])
        else:
            if sweepMode == 0:  # Sequential for T2 contrast
                for ii in range(int(nPH / etl)):
                    ind = np.concatenate((ind, np.linspace(ii, nPH + ii, num=etl, endpoint=False)), axis=0)
                ind = ind[::-1]
            elif sweepMode == 1:  # Center-out for T1 contrast
                if etl == nPH:
                    ind = np.zeros(nPH)
                    ind[0::2] = np.linspace(int(nPH / 2), nPH, num=int(nPH / 2), endpoint=False)
                    ind[1::2] = np.linspace(int(nPH / 2) - 1, -1, num=int(nPH / 2), endpoint=False)
                else:
                    for ii in range(n2ETL):
                        ind = np.concatenate((ind, np.linspace(int(nPH / 2) + ii, nPH + ii, num=etl, endpoint=False)),
                                             axis=0)
                        ind = np.concatenate(
                            (ind, np.linspace(int(nPH / 2) - ii - 1, -ii - 1, num=etl, endpoint=False)), axis=0)
            elif sweepMode == 2:  # Out-to-center for T2 contrast
                if etl == nPH:
                    ind = np.zeros(nPH)
                    ind[0::2] = np.linspace(int(nPH / 2), nPH, num=int(nPH / 2), endpoint=False)
                    ind[1::2] = np.linspace(int(nPH / 2) - 1, -1, num=int(nPH / 2), endpoint=False)
                else:
                    for ii in range(n2ETL):
                        ind = np.concatenate((ind, np.linspace(int(nPH / 2) + ii, nPH + ii, num=etl, endpoint=False)),
                                             axis=0)
                        ind = np.concatenate(
                            (ind, np.linspace(int(nPH / 2) - ii - 1, -ii - 1, num=etl, endpoint=False)), axis=0)
                ind = ind[::-1]
            elif sweepMode == 3:  # Niquist modulated to reduce ghosting artifact
                if etl == nPH:
                    ind = np.arange(0, nPH, 1)
                else:
                    for ii in range(int(n2ETL)):
                        ind = np.concatenate((ind, np.arange(0, nPH, 2 * n2ETL) + 2 * ii), axis=0)
                        ind = np.concatenate((ind, np.arange(nPH - 1, 0, -2 * n2ETL) - 2 * ii), axis=0)

        return np.int32(ind)

    def fixEchoPosition(self, echoes, data0):
        """
        Adjust the position of k=0 in the echo data to the center of the acquisition window.

        This method uses oversampled data obtained with a given echo train length and readout gradient to determine the
        true position of k=0. It then shifts the sampled data to place k=0 at the center of each acquisition window for
        each gradient-spin-echo.

        Args:
            echoes (numpy.ndarray): The echo data array with dimensions [etl, n], where `etl` is the echo train length
                                    and `n` is the number of samples per echo. This echo train is acquired in a dummy
                                    excitation before the sequence using only the readout gradient.
            data0 (numpy.ndarray): The original data array to be adjusted with dimensions [channels, etl, n].

        Returns:
            numpy.ndarray: The adjusted data array with k=0 positioned at the center of each acquisition window.

        """
        etl = np.size(echoes, axis=0)
        n = np.size(echoes, axis=1)
        idx = np.argmax(np.abs(echoes), axis=1)
        idx = idx - int(n / 2)
        data1 = data0 * 0
        for ii in range(etl):
            if idx[ii] > 0:
                idx[ii] = 0
            data1[:, ii, -idx[ii]::] = data0[:, ii, 0:n + idx[ii]]
        return data1

    def decimate(self, data_over, n_adc, option='PETRA', remove=True):
        """
        Decimates oversampled MRI data, with optional preprocessing to manage oscillations and postprocessing
        to remove extra points.

        Parameters:
        -----------
        data_over : numpy.ndarray
            The oversampled data array to be decimated.
        n_adc : int
            The number of adc windows in the dataset, used to reshape and process the data appropriately.
        option : str, optional
            Preprocessing option to handle data before decimation:
            - 'PETRA': Adjusts initial points to avoid oscillations during decimation.
            - 'Normal': Applies no preprocessing (default is 'PETRA').
        remove : bool, optional
            If True, removes `addRdPoints` from the start and end of each readout line after decimation.
            Defaults to True.

        Returns:
        --------
        numpy.ndarray
            The decimated data array, optionally adjusted to remove extra points.

        Workflow:
        ---------
        1. **Preprocess data (optional)**:
            - For 'PETRA' mode, reshapes the data into adc windows and adjusts the first few points of each line
              to avoid oscillations caused by decimation.
            - For 'Normal' mode, no preprocessing is applied.

        2. **Decimate the signal**:
            - Applies a finite impulse response (FIR) filter and decimates the signal by the oversampling factor
              (`hw.oversamplingFactor`).
            - Starts decimation after skipping `(oversamplingFactor - 1) / 2` points to minimize edge effects.

        3. **Postprocess data (if `remove=True`)**:
            - Reshapes the decimated data into adc windows.
            - Removes `hw.addRdPoints` from the start and end of each line.
            - Reshapes the cleaned data back into a 1D array.

        Notes:
        ------
        - This method uses the hardware-specific parameters:
          - `hw.oversamplingFactor`: The oversampling factor applied during data acquisition.
          - `hw.addRdPoints`: The number of additional readout points to include or remove.
        - The 'PETRA' preprocessing mode is tailored for specialized MRI acquisitions that require smoothing of
          initial points to prevent oscillations.
        """

        # Preprocess the signal to avoid oscillations due to decimation
        if option == 'PETRA':
            data_over = np.reshape(data_over, (n_adc, -1))
            for line in range(n_adc):
                data_over[line, 0:hw.addRdPoints * hw.oversamplingFactor] = data_over[
                    line, hw.addRdPoints * hw.oversamplingFactor]
            data_over = np.reshape(data_over, -1)
        elif option == 'Normal':
            pass

        # Decimate the signal after 'fir' filter
        data_decimated = sig.decimate(data_over[int((hw.oversamplingFactor - 1) / 2)::], hw.oversamplingFactor,
                                      ftype='fir', zero_phase=True)

        # Remove addRdPoints
        if remove:
            nPoints = int(data_decimated.shape[0] / n_adc) - 2 * hw.addRdPoints
            data_decimated = np.reshape(data_decimated, (n_adc, -1))
            data_decimated = data_decimated[:, hw.addRdPoints:hw.addRdPoints + nPoints]
            data_decimated = np.reshape(data_decimated, -1)

        return data_decimated

    def rfSincPulse(self, tStart, rfTime, rfAmplitude, rfPhase=0, nLobes=7, channel=0, rewrite=True):
        """
        Generate an RF pulse with a sinc pulse shape and the corresponding deblanking signal. It uses a Hanning window
        to reduce the banding of the frequency profile.

        Args:
            tStart (float): Start time of the RF pulse.
            rfTime (float): Duration of the RF pulse.
            rfAmplitude (float): Amplitude of the RF pulse.
            rfPhase (float): Phase of the RF pulse in radians. Default is 0.
            nLobes (int): Number of lobes in the sinc pulse. Default is 7.
            channel (int): Channel index for the RF pulse. Default is 0.
            rewrite (bool): Whether to rewrite the existing RF pulse. Default is True.

        """
        txTime = np.linspace(tStart, tStart + rfTime, num=100, endpoint=True) + hw.blkTime
        nZeros = (nLobes + 1)
        tx = np.linspace(-nZeros / 2, nZeros / 2, num=100, endpoint=True)
        hanning = 0.5 * (1 + np.cos(2 * np.pi * tx / nZeros))
        txAmp = rfAmplitude * np.exp(1j * rfPhase) * hanning * np.abs(np.sinc(tx))
        txGateTime = np.array([tStart, tStart + hw.blkTime + rfTime])
        txGateAmp = np.array([1, 0])
        self.flo_dict['tx%i' % channel][0] = np.concatenate((self.flo_dict['tx%i' % channel][0], txTime), axis=0)
        self.flo_dict['tx%i' % channel][1] = np.concatenate((self.flo_dict['tx%i' % channel][1], txAmp), axis=0)
        self.flo_dict['ttl0'][0] = np.concatenate((self.flo_dict['ttl0'][0], txGateTime), axis=0)
        self.flo_dict['ttl0'][1] = np.concatenate((self.flo_dict['ttl0'][1], txGateAmp), axis=0)

    def rfRawSincPulse(self, tStart, rfTime, rfAmplitude, rfPhase=0, nLobes=7, channel=0, rewrite=True):
        """
        Generate an RF pulse with a sinc pulse shape. It uses a Hanning window
        to reduce the banding of the frequency profile.

        Args:
            tStart (float): Start time of the RF pulse.
            rfTime (float): Duration of the RF pulse.
            rfAmplitude (float): Amplitude of the RF pulse.
            rfPhase (float): Phase of the RF pulse in radians. Default is 0.
            nLobes (int): Number of lobes in the sinc pulse. Default is 7.
            channel (int): Channel index for the RF pulse. Default is 0.
            rewrite (bool): Whether to rewrite the existing RF pulse. Default is True.

        """
        txTime = np.linspace(tStart, tStart + rfTime, num=100, endpoint=True) + hw.blkTime
        nZeros = (nLobes + 1)
        tx = np.linspace(-nZeros / 2, nZeros / 2, num=100, endpoint=True)
        hanning = 0.5 * (1 + np.cos(2 * np.pi * tx / nZeros))
        txAmp = rfAmplitude * np.exp(1j * rfPhase) * hanning * np.abs(np.sinc(tx))
        txGateTime = np.array([tStart, tStart + hw.blkTime + rfTime])
        txGateAmp = np.array([1, 0])
        self.flo_dict['tx%i' % channel][0] = np.concatenate((self.flo_dict['tx%i' % channel][0], txTime), axis=0)
        self.flo_dict['tx%i' % channel][1] = np.concatenate((self.flo_dict['tx%i' % channel][1], txAmp), axis=0)

    def rfRecPulse(self, tStart, rfTime, rfAmplitude, rfPhase=0, channel=0):
        """
        Generate an RF pulse with a rectangular pulse shape and the corresponding deblanking signal.

        Args:
            tStart (float): Start time of the RF pulse.
            rfTime (float): Duration of the RF pulse.
            rfAmplitude (float): Amplitude of the RF pulse.
            rfPhase (float): Phase of the RF pulse in radians. Default is 0.
            channel (int): Channel index for the RF pulse. Default is 0.

        """
        txTime = np.array([tStart + hw.blkTime, tStart + hw.blkTime + rfTime])
        txAmp = np.array([rfAmplitude * np.exp(1j * rfPhase), 0.])
        txGateTime = np.array([tStart, tStart + hw.blkTime + rfTime])
        txGateAmp = np.array([1, 0])
        self.flo_dict['tx%i' % channel][0] = np.concatenate((self.flo_dict['tx%i' % channel][0], txTime), axis=0)
        self.flo_dict['tx%i' % channel][1] = np.concatenate((self.flo_dict['tx%i' % channel][1], txAmp), axis=0)
        self.flo_dict['ttl0'][0] = np.concatenate((self.flo_dict['ttl0'][0], txGateTime), axis=0)
        self.flo_dict['ttl0'][1] = np.concatenate((self.flo_dict['ttl0'][1], txGateAmp), axis=0)

    def rfRawPulse(self, tStart, rfTime, rfAmplitude, rfPhase=0, channel=0):
        """
        Generate an RF pulse with a rectangular pulse shape.

        Args:
            tStart (float): Start time of the RF pulse.
            rfTime (float): Duration of the RF pulse.
            rfAmplitude (float): Amplitude of the RF pulse.
            rfPhase (float): Phase of the RF pulse in radians. Default is 0.
            channel (int): Channel index for the RF pulse. Default is 0.

        """
        txTime = np.array([tStart, tStart + rfTime])
        txAmp = np.array([rfAmplitude * np.exp(1j * rfPhase), 0.])
        self.flo_dict['tx%i' % channel][0] = np.concatenate((self.flo_dict['tx%i' % channel][0], txTime), axis=0)
        self.flo_dict['tx%i' % channel][1] = np.concatenate((self.flo_dict['tx%i' % channel][1], txAmp), axis=0)

    def rxGate(self, tStart, gateTime, channel=0):
        """
        Open the receiver gate for a specified channel.

        Args:
            tStart (float): Start time of the receiver gate.
            gateTime (float): Duration of the receiver gate.
            channel (int): Channel index for the receiver gate. Default is 0.

        """
        self.flo_dict['rx%i' % channel][0] = \
            np.concatenate((self.flo_dict['rx%i' % channel][0], np.array([tStart, tStart + gateTime])), axis=0)
        self.flo_dict['rx%i' % channel][1] = \
            np.concatenate((self.flo_dict['rx%i' % channel][1], np.array([1, 0])), axis=0)

    def rxGateSync(self, tStart, gateTime, channel=0):
        """
        Open a synchronized receiver gate for a specified channel with additional points to account for the time shift
        and ramp of the CIC filter.

        Args:
            tStart (float): Start time of the receiver gate.
            gateTime (float): Duration of the receiver gate.
            channel (int): Channel index for the receiver gate. Default is 0.

        Notes:
            - This method is designed to work with the Experiment class in the controller, which inherits from Experiment in marcos_client.

        """
        # Generate instructions taking into account the cic filter delay and addRdPoints
        try:
            samplingRate = self.expt.getSamplingRate() / hw.oversamplingFactor  # us
        except:
            samplingRate = self.mapVals['samplingPeriod'] / hw.oversamplingFactor
        t0 = tStart - (hw.addRdPoints * hw.oversamplingFactor - hw.cic_delay_points) * samplingRate  # us
        t1 = tStart + (hw.addRdPoints * hw.oversamplingFactor + hw.cic_delay_points) * samplingRate + gateTime  # us
        self.flo_dict['rx%i' % channel][0] = \
            np.concatenate((self.flo_dict['rx%i' % channel][0], np.array([t0, t1])), axis=0)
        self.flo_dict['rx%i' % channel][1] = \
            np.concatenate((self.flo_dict['rx%i' % channel][1], np.array([1, 0])), axis=0)

    def ttl(self, tStart, ttlTime, channel=0):
        """
        Generate a digital signal for a specified channel.

        Args:
            tStart (float): Start time of the TTL signal.
            ttlTime (float): Duration of the TTL signal.
            channel (int): Channel index for the TTL signal. Default is 0.

        """
        self.flo_dict['ttl%i' % channel][0] = \
            np.concatenate((self.flo_dict['ttl%i' % channel][0], np.array([tStart, tStart + ttlTime])), axis=0)
        self.flo_dict['ttl%i' % channel][1] = \
            np.concatenate((self.flo_dict['ttl%i' % channel][1], np.array([1, 0])), axis=0)

    def gradTrap(self, tStart, gRiseTime, gFlattopTime, gAmp, gSteps, gAxis, shimming):
        """
        Generate a gradient pulse with trapezoidal shape.

        This method generates a gradient pulse with a trapezoidal shape. One step is used to generate a rectangular
        pulse.

        Args:
            tStart (float): Start time of the gradient pulse.
            gRiseTime (float): Rise time of the gradient pulse in microseconds.
            gFlattopTime (float): Flattop time of the gradient pulse in microseconds.
            gAmp (float): Amplitude of the gradient pulse in T/m.
            gSteps (int): Number of steps for the gradient ramps.
            gAxis (int): Axis index for the gradient pulse.
            shimming (list): List of shimming values for each axis in arbitrary units from marcos.

        Notes:
            - Time inputs are in microseconds.
            - Amplitude inputs are in T/m.
            - shimming is in arbitrary units

        """
        tUp = np.linspace(tStart, tStart + gRiseTime, num=gSteps, endpoint=False)
        tDown = tUp + gRiseTime + gFlattopTime
        t = np.concatenate((tUp, tDown), axis=0)

        dAmp = gAmp / gSteps
        aUp = np.linspace(dAmp, gAmp, num=gSteps)
        aDown = np.linspace(gAmp - dAmp, 0, num=gSteps)
        a = np.squeeze(np.concatenate((aUp, aDown), axis=0)) / hw.gFactor[gAxis] + shimming[gAxis]

        self.flo_dict['g%i' % gAxis][0] = np.concatenate((self.flo_dict['g%i' % gAxis][0], t), axis=0)
        self.flo_dict['g%i' % gAxis][1] = np.concatenate((self.flo_dict['g%i' % gAxis][1], a), axis=0)

    def gradTrapMomentum(self, tStart, kMax, gTotalTime, gAxis, shimming, rewrite=True):
        """
        Generate a gradient pulse with trapezoidal shape according to slew rate.

        This method generates a gradient pulse with a trapezoidal shape based on the provided slew rate and maximum
        k-value.

        Args:
            tStart (float): Start time of the gradient pulse.
            kMax (float): Maximum k-value in 1/m.
            gTotalTime (float): Total time of the gradient pulse in microseconds including flattop and ramps.
            gAxis (int): Axis index for the gradient pulse.
            shimming (list): List of shimming values for each axis.
            rewrite (bool, optional): Whether to overwrite existing gradient data. Defaults to True.

        Notes:
            - Time inputs are in microseconds.
            - kMax inputs are in 1/m.
            - shimming is in arbitrary units
            - NOT FULLY TESTED

        """
        kMax = kMax / hw.gammaB * 1e6

        # Changing from Ocra1 units
        slewRate = hw.slewRate / hw.gFactor[gAxis]  # Convert to units [s*m/T]
        stepsRate = hw.stepsRate / hw.gFactor[gAxis]  # Convert to units [steps*m/T]

        # Calculating amplitude
        gAmplitude = (gTotalTime - np.sqrt(gTotalTime ** 2 - 4 * slewRate * kMax)) / (2 * slewRate)

        # Trapezoid characteristics
        gRiseTime = gAmplitude * slewRate
        nSteps = int(np.ceil(gAmplitude * stepsRate))

        # Creating trapezoid
        tRise = np.linspace(tStart, tStart + gRiseTime, nSteps, endpoint=True)
        aRise = np.linspace(0, gAmplitude, nSteps, endpoint=True)
        tDown = np.linspace(tStart + gTotalTime - gRiseTime, tStart + gTotalTime, nSteps, endpoint=True)
        aDown = np.linspace(gAmplitude, 0, nSteps, endpoint=True)
        gTime = np.concatenate((tRise, tDown), axis=0)
        gAmp = np.concatenate((aRise, aDown), axis=0) / hw.gFactor[gAxis]
        if gAxis == 0:
            self.expt.add_flodict({'grad_vx': (gTime, gAmp + shimming[0])}, rewrite)
        elif gAxis == 1:
            self.expt.add_flodict({'grad_vy': (gTime, gAmp + shimming[1])}, rewrite)
        elif gAxis == 2:
            self.expt.add_flodict({'grad_vz': (gTime, gAmp + shimming[2])}, rewrite)

    def setGradientRamp(self, tStart, gradRiseTime, nStepsGradRise, g0, gf, gAxis, shimming, rewrite=True):
        """
        Generate a gradient ramp from 'g0' to 'gf'.

        This method generates a gradient ramp from the initial amplitude 'g0' to the final amplitude 'gf' over the
        specified rise time.

        Args:
            tStart (float): Start time of the gradient ramp.
            gradRiseTime (float): Rise time of the gradient ramp in microseconds.
            nStepsGradRise (int): Number of steps in the gradient ramp.
            g0 (float): Initial gradient amplitude in T/m.
            gf (float): Final gradient amplitude in T/m.
            gAxis (int): Axis index for the gradient ramp.
            shimming (list): List of shimming values for each axis.
            rewrite (bool, optional): Whether to overwrite existing gradient data. Defaults to True.

        Notes:
            - Time inputs are in microseconds.
            - Amplitude inputs are in T/m.
            - shimming is in arbitrary units

        """
        for kk in range(nStepsGradRise):
            tRamp = tStart + gradRiseTime * kk / nStepsGradRise
            gAmp = (g0 + ((gf - g0) * (kk + 1) / nStepsGradRise)) / hw.gFactor[gAxis] + shimming[gAxis]
            self.flo_dict['g%i' % gAxis][0] = np.concatenate((self.flo_dict['g%i' % gAxis][0], np.array([tRamp])),
                                                             axis=0)
            self.flo_dict['g%i' % gAxis][1] = np.concatenate((self.flo_dict['g%i' % gAxis][1], np.array([gAmp])),
                                                             axis=0)

    def gradTrapAmplitude(self, tStart, gAmplitude, gTotalTime, gAxis, shimming, orders, rewrite=True):
        """
        Generate a gradient pulse with trapezoidal shape according to slew rate and specified amplitude.

        This method generates a gradient pulse with a trapezoidal shape according to the specified amplitude and slew
        rate.

        Args:
            tStart (float): Start time of the gradient pulse.
            gAmplitude (float): Amplitude of the gradient pulse in T/m.
            gTotalTime (float): Total duration of the gradient pulse in microseconds.
            gAxis (int): Axis index for the gradient pulse.
            shimming (list): List of shimming values for each axis.
            orders (int): Number of orders.
            rewrite (bool, optional): Whether to overwrite existing gradient data. Defaults to True.

        Notes:
            - Time inputs are in microseconds.
            - gAmplitude input is in T/m.
            - shimming is in arbitrary units
            - OUT OF DATE

        """
        # Changing from Ocra1 units
        slewRate = hw.slewRate / hw.gFactor[gAxis]  # Convert to units [s*m/T]
        stepsRate = hw.stepsRate / hw.gFactor[gAxis]  # Convert to units [steps*m/T]

        # Trapezoid characteristics
        gRiseTime = np.abs(gAmplitude * slewRate)
        nSteps = int(np.ceil(np.abs(gAmplitude * stepsRate)))
        orders = orders + 2 * nSteps

        # Creating trapezoid
        tRise = np.linspace(tStart, tStart + gRiseTime, nSteps, endpoint=True)
        aRise = np.linspace(0, gAmplitude, nSteps, endpoint=True)
        tDown = np.linspace(tStart + gTotalTime - gRiseTime, tStart + gTotalTime, nSteps, endpoint=True)
        aDown = np.linspace(gAmplitude, 0, nSteps, endpoint=True)
        gTime = np.concatenate((tRise, tDown), axis=0)
        gAmp = np.concatenate((aRise, aDown), axis=0) / hw.gFactor[gAxis]
        if gAxis == 0:
            self.expt.add_flodict({'grad_vx': (gTime, gAmp + shimming[0])}, rewrite)
        elif gAxis == 1:
            self.expt.add_flodict({'grad_vy': (gTime, gAmp + shimming[1])}, rewrite)
        elif gAxis == 2:
            self.expt.add_flodict({'grad_vz': (gTime, gAmp + shimming[2])}, rewrite)

    def endSequence(self, tEnd):
        """
        Finalize the sequence by setting the gradients, RX, TX, and TTL signals to zero at the specified end time.

        Args:
            tEnd (float): End time of the sequence in microseconds.

        """
        self.flo_dict['g0'][0] = np.concatenate((self.flo_dict['g0'][0], np.array([tEnd])), axis=0)
        self.flo_dict['g0'][1] = np.concatenate((self.flo_dict['g0'][1], np.array([0])), axis=0)
        self.flo_dict['g1'][0] = np.concatenate((self.flo_dict['g1'][0], np.array([tEnd])), axis=0)
        self.flo_dict['g1'][1] = np.concatenate((self.flo_dict['g1'][1], np.array([0])), axis=0)
        self.flo_dict['g2'][0] = np.concatenate((self.flo_dict['g2'][0], np.array([tEnd])), axis=0)
        self.flo_dict['g2'][1] = np.concatenate((self.flo_dict['g2'][1], np.array([0])), axis=0)
        self.flo_dict['rx0'][0] = np.concatenate((self.flo_dict['rx0'][0], np.array([tEnd])), axis=0)
        self.flo_dict['rx0'][1] = np.concatenate((self.flo_dict['rx0'][1], np.array([0])), axis=0)
        self.flo_dict['rx1'][0] = np.concatenate((self.flo_dict['rx1'][0], np.array([tEnd])), axis=0)
        self.flo_dict['rx1'][1] = np.concatenate((self.flo_dict['rx1'][1], np.array([0])), axis=0)
        self.flo_dict['tx0'][0] = np.concatenate((self.flo_dict['tx0'][0], np.array([tEnd])), axis=0)
        self.flo_dict['tx0'][1] = np.concatenate((self.flo_dict['tx0'][1], np.array([0])), axis=0)
        self.flo_dict['tx1'][0] = np.concatenate((self.flo_dict['tx1'][0], np.array([tEnd])), axis=0)
        self.flo_dict['tx1'][1] = np.concatenate((self.flo_dict['tx1'][1], np.array([0])), axis=0)
        self.flo_dict['ttl0'][0] = np.concatenate((self.flo_dict['ttl0'][0], np.array([tEnd])), axis=0)
        self.flo_dict['ttl0'][1] = np.concatenate((self.flo_dict['ttl0'][1], np.array([0])), axis=0)
        self.flo_dict['ttl1'][0] = np.concatenate((self.flo_dict['ttl1'][0], np.array([tEnd])), axis=0)
        self.flo_dict['ttl1'][1] = np.concatenate((self.flo_dict['ttl1'][1], np.array([0])), axis=0)

    def iniSequence(self, t0, shimming):
        """
        Initialize the sequence by setting the initial values for gradients, RX, TX, and TTL signals at the desired
        time.

        Args:
            t0 (float): Initial time of the sequence in microseconds.
            shimming (list): List of shimming values for each axis in arbitrary units.

        """
        self.flo_dict['g0'][0] = np.array([t0])
        self.flo_dict['g0'][1] = np.array([shimming[0]])
        self.flo_dict['g1'][0] = np.array([t0])
        self.flo_dict['g1'][1] = np.array([shimming[1]])
        self.flo_dict['g2'][0] = np.array([t0])
        self.flo_dict['g2'][1] = np.array([shimming[2]])
        self.flo_dict['rx0'][0] = np.array([t0])
        self.flo_dict['rx0'][1] = np.array([0])
        self.flo_dict['rx1'][0] = np.array([t0])
        self.flo_dict['rx1'][1] = np.array([0])
        self.flo_dict['tx0'][0] = np.array([t0])
        self.flo_dict['tx0'][1] = np.array([0])
        self.flo_dict['tx1'][0] = np.array([t0])
        self.flo_dict['tx1'][1] = np.array([0])
        self.flo_dict['ttl0'][0] = np.array([t0])
        self.flo_dict['ttl0'][1] = np.array([0])
        self.flo_dict['ttl1'][0] = np.array([t0])
        self.flo_dict['ttl1'][1] = np.array([0])

    def setGradient(self, t0, gAmp, gAxis, rewrite=True):
        """
        Set the gradient amplitude to a given value at a specified time.

        Args:
            t0 (float): Time at which the gradient is set, in microseconds.
            gAmp (float): Amplitude of the gradient, in arbitrary units [-1, 1].
            gAxis (int): Axis of the gradient (0 for x, 1 for y, 2 for z).
            rewrite (bool, optional): Whether to overwrite existing values. Defaults to True.

        """
        self.flo_dict['g%i' % gAxis][0] = np.concatenate((self.flo_dict['g%i' % gAxis][0], np.array([t0])), axis=0)
        self.flo_dict['g%i' % gAxis][1] = np.concatenate((self.flo_dict['g%i' % gAxis][1], np.array([gAmp])), axis=0)

    def floDict2Exp(self, rewrite=True, demo=False):
        """
        Check for errors and add instructions to Red Pitaya if no errors are found.

        Args:
            rewrite (bool, optional): Whether to overwrite existing values. Defaults to True.
            demo: If demo is True it just check for errors. Defaults to False.

        Returns:
            bool: True if no errors were found and instructions were successfully added to Red Pitaya; False otherwise.

        """
        # Check errors:
        for key in self.flo_dict.keys():
            item = self.flo_dict[key]
            dt = item[0][1::] - item[0][0:-1]
            if (dt <= 1).any():
                print("ERROR: %s timing error" % key)
                return False
            if (item[1] > 1).any() or (item[1] < -1).any():
                print("ERROR: %s amplitude error" % key)
                return False

        # Add instructions to server
        if not self.demo:
            self.expt.add_flodict({'grad_vx': (self.flo_dict['g0'][0], self.flo_dict['g0'][1]),
                                   'grad_vy': (self.flo_dict['g1'][0], self.flo_dict['g1'][1]),
                                   'grad_vz': (self.flo_dict['g2'][0], self.flo_dict['g2'][1]),
                                   'rx0_en': (self.flo_dict['rx0'][0], self.flo_dict['rx0'][1]),
                                   'rx1_en': (self.flo_dict['rx1'][0], self.flo_dict['rx1'][1]),
                                   'tx0': (self.flo_dict['tx0'][0], self.flo_dict['tx0'][1]),
                                   'tx1': (self.flo_dict['tx1'][0], self.flo_dict['tx1'][1]),
                                   'tx_gate': (self.flo_dict['ttl0'][0], self.flo_dict['ttl0'][1]),
                                   'rx_gate': (self.flo_dict['ttl1'][0], self.flo_dict['ttl1'][1]),
                                   }, rewrite)
        return True

    def saveRawData(self):
        
        """
        Save the rawData.

        This method saves the rawData to various formats including .mat, .csv, .dcm, .nii and .h5.

        The .mat file contains the rawData.
        The .csv file contains only the input parameters.
        The .dcm file is the DICOM image.
        The .nii file is the nifti image
        The .h5 file is the ISMRMRD format.
        

        Returns:
            None

        """
        
        # Get directory
        if 'directory' in self.session.keys():
            directory = self.session['directory']
        else:
            dt2 = date.today()
            date_string = dt2.strftime("%Y.%m.%d")
            directory = 'experiments/acquisitions/%s' % (date_string)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # generate directories for mat, csv and dcm files
        directory_mat = directory + '/mat'
        directory_csv = directory + '/csv'
        directory_dcm = directory + '/dcm'
        directory_nii = directory + '/nii'
        directory_ismrmrd = directory + '/ismrmrd'
        
        if not os.path.exists(directory + '/mat'):
            os.makedirs(directory_mat)
        if not os.path.exists(directory + '/csv'):
            os.makedirs(directory_csv)
        if not os.path.exists(directory + '/dcm'):
            os.makedirs(directory_dcm)
        if not os.path.exists(directory + '/nii'):
            os.makedirs(directory_nii)
        if not os.path.exists(directory + '/ismrmrd'):
            os.makedirs(directory_ismrmrd)

        self.directory_rmd=directory_ismrmrd 
        
        # Generate filename
        name = datetime.now()
        name_string = name.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3]
        self.mapVals['name_string'] = name_string
        if hasattr(self, 'raw_data_name'):
            file_name = "%s.%s" % (self.raw_data_name, name_string)
        else:
            self.raw_data_name = self.mapVals['seqName']
            file_name = "%s.%s" % (self.mapVals['seqName'], name_string)
        self.mapVals['fileName'] = "%s.mat" % file_name
        # Generate filename for ismrmrd
        self.mapVals['fileNameIsmrmrd'] = "%s.h5" % file_name
        
        # Save mat file with the outputs
        savemat("%s/%s.mat" % (directory_mat, file_name), self.mapVals) # au format savemat(chemin_fichier_mat, {"data" : data}), avec data contient les données brute à sauvegarder

        # Save csv with input parameters
        with open('%s/%s.csv' % (directory_csv, file_name), 'w') as csvfile: # ouvrir le fichier csv en mode écriture au format with open(chemin_fichier_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.mapKeys) # mapKeys contient les noms des colonnes du fichier csv que l'on veut sauvegarder 
            writer.writeheader() # écrire l'entete du csv les noms des colonnes dans le fichier csv
            mapVals = {} # stockage de valeurs de données à écrire
            for key in self.mapKeys:  # take only the inputs from mapVals
                mapVals[key] = self.mapVals[key] # copie les donnée de l'acquisition stockées dans self.mapVals dans mapVals
            writer.writerows([self.mapNmspc, mapVals]) # écrire les données dans le fichier csv

        # Save dcm and nifti with the final image
        if (len(self.output) > 0) and (self.output[0]['widget'] == 'image') and (self.mode is None): ##verify if output is an image
            try:
                utils.save_dicom(axes_orientation=self.mapVals['axesOrientation'],
                                 n_points=self.mapVals['nPoints'],
                                 fov=self.mapVals['fov'],
                                 image=self.mapVals['image3D'],
                                 file_path=f"{directory_dcm}/{file_name}.dcm",
                                 meta_data = self.meta_data,
                                 )
                utils.save_nifti(axes_orientation=self.mapVals['axesOrientation'],
                                 n_points=self.mapVals['nPoints'],
                                 fov=self.mapVals['fov'],
                                 dfov=self.mapVals['dfov'],
                                 image=self.mapVals['image3D'],
                                 file_path=f"{directory_nii}/{file_name}.nii"
                                 )
            except:
                pass

        # Move seq files
        self.move_batch_files(destination_folder=directory, file_name=file_name)

    @staticmethod
    def move_batch_files(destination_folder, file_name):
        """
        Move batch_X.seq files from the current working directory to the specified destination folder.

        The method scans all files in the current directory and identifies files with the extension '.seq'.
        It extracts the batch number from the file name (in the format 'batch_X.seq', where 'X' is the batch number).
        Then, it moves these files to a subfolder 'seq' inside the specified destination folder, renaming them based on the provided `file_name` template.

        Args:
        - destination_folder (str): The path to the destination folder where the 'seq' subfolder will be created, and the files will be moved.
        - file_name (str): The prefix used for renaming the files. Files will be renamed in the format 'file_name_X.seq', where 'X' is the extracted batch number from the original file name.

        Example:
            If the file 'batch_1.seq' is found and `file_name='processed'`, it will be moved and renamed to:
            'destination_folder/seq/processed_1.seq'.

        Side Effects:
        - Creates a 'seq' subfolder in the destination folder if it doesn't already exist.
        - Moves and renames the matched '.seq' files from the current directory.

        """
        # List all files in the source folder
        for source_file in os.listdir():
            # Match files with the pattern 'batch_X.seq'
            file_prov = source_file.split('.')
            if file_prov[-1]=='seq' and os.path.isfile(source_file):
                batch_num = file_prov[0].split('_')[-1]

                # Create the destination folder path based on the batch number
                os.makedirs(os.path.join(destination_folder, 'seq'), exist_ok=True)

                # Move the file to the destination folder
                destination_file = os.path.join(destination_folder, 'seq', file_name+'_%s.seq' % batch_num)
                shutil.move(source_file, destination_file)
                print(f'Moved: {file_name} to {destination_folder}')

    def image2Dicom(self, fileName): 
        """
        Save the DICOM image.

        This method saves the DICOM image with the given filename.

        Args:
            fileName (str): The filename to save the DICOM image.

        Returns:
            None

        """
        # Create DICOM object
        dicom_image = DICOMImage()

        # Save image into DICOM object
        try:
            dicom_image.meta_data["PixelData"] = self.meta_data["PixelData"]
        except KeyError:
            image = self.output[0]['data']
            dicom_image.meta_data["PixelData"] = image.astype(np.int16).tobytes()
            # If it is a 3D image
            if len(image.shape) > 2:
                # Get dimensions
                slices, rows, columns = image.shape
                dicom_image.meta_data["Columns"] = columns
                dicom_image.meta_data["Rows"] = rows
                dicom_image.meta_data["NumberOfSlices"] = slices
                dicom_image.meta_data["NumberOfFrames"] = slices
            # If it is a 2D image
            else:
                # Get dimensions
                rows, columns = image.shape
                dicom_image.meta_data["Columns"] = columns
                dicom_image.meta_data["Rows"] = rows
                dicom_image.meta_data["NumberOfSlices"] = 1
                dicom_image.meta_data["NumberOfFrames"] = 1

        # Date and time
        current_time = datetime.now()
        self.meta_data["StudyDate"] = current_time.strftime("%Y%m%d")
        self.meta_data["StudyTime"] = current_time.strftime("%H%M%S")

        # More DICOM tags
        self.meta_data["PatientName"] = self.session["subject_id"]
        self.meta_data["PatientSex"] = " "
        self.meta_data["StudyID"] = self.session["subject_id"]
        self.meta_data["InstitutionName"] = self.session["scanner"]
        self.meta_data["ImageComments"] = " "
        self.meta_data["PatientID"] = self.session["subject_id"]
        self.meta_data["SOPInstanceUID"] = self.mapVals['name_string']
        self.meta_data["SeriesDescription"] = self.raw_data_name
        self.session['seriesNumber'] = self.session['seriesNumber'] + 1
        self.meta_data["SeriesNumber"] = self.session['seriesNumber']
        # Full dynamic window
        # self.meta_data["WindowWidth"] = 26373
        # self.meta_data["WindowCenter"] = 13194

        # Update the DICOM metadata
        dicom_image.meta_data.update(self.meta_data)

        # Save metadata dictionary into DICOM object metadata (Standard DICOM 3.0)
        dicom_image.image2Dicom() 

        # Save DICOM file
        dicom_image.save(fileName)

    def addParameter(self, key='', string='', val=0, units=True, field='', tip=None):
        """
        Add a parameter to the sequence.

        This method adds a parameter to the sequence with the specified key, description string, value, units, field,
        and tip.

        Args:
            key (str): The key of the parameter.
            string (str): The description string of the parameter.
            val (int/float/str/list): The value of the parameter. It can be an integer, a float, a string, or a list.
            units (bool): Indicates the units of the parameter (e.g. cm -> 1e-2, or you can use the config/units.py module).
            field (str): The field of the parameter: 'RF', 'IMG', 'SEQ', 'OTH'.
            tip (str): Additional information or tip about the parameter.

        Returns:
            None

        """
        if key not in self.mapVals.keys():
            self.mapKeys.append(key)
        self.mapNmspc[key] = string
        self.mapVals[key] = val
        self.mapFields[key] = field
        self.mapTips[key] = tip
        self.map_units[key] = units
        try:
            self.mapLen[key] = len(val)
        except TypeError:
            self.mapLen[key] = 1

    def sequenceAtributes(self):
        """
        Add input parameters to the sequence object.

        This method iterates over the input parameters and adds them as attributes to the sequence object (self). It
        multiplies each parameter by its corresponding unit if units are specified (e.g. key = 'f0', val = 3,
        units = 1e-6 will create self.f0 = 3e-6.

        Returns:
            None

        """
        for key in self.mapKeys: 
            if isinstance(self.mapVals[key], list): 
                setattr(self, key, np.array([element * self.map_units[key] for element in self.mapVals[key]]))
            else:
                setattr(self, key, self.mapVals[key] * self.map_units[key])

        # Conversion of variables to non-multiplied units
        if self.pypulseq:
            self.angle = - self.angle * np.pi / 180  # rads
        else:
            self.angle = + self.angle * np.pi / 180

        # Add rotation, dfov and fov to the history
        self.rotation = self.rotationAxis.tolist()
        self.rotation.append(self.angle)
        self.rotations.append(self.rotation)
        self.dfovs.append(self.dfov.tolist())
        self.fovs.append(self.fov.tolist())

    def plotResults(self):
        """
        Plot results in a standalone window.

        This method generates plots based on the output data provided. It creates a plot window, inserts each plot
        according to its type (image or curve), sets titles and labels, and displays the plot.

        Returns:
            None

        """
        # Determine the number of columns and rows for subplots
        cols = 1
        rows = 1
        for item in self.output:
            if item['row'] + 1 > rows:
                rows = item['row'] + 1
            if item['col'] + 1 > cols:
                cols = item['col'] + 1

        # Create the plot window
        fig, axes = plt.subplots(rows, cols, figsize=(10, 5))

        # Insert plots
        plot = 0
        for item in self.output:
            if item['widget'] == 'image':
                nz, ny, nx = item['data'].shape
                plt.subplot(rows, cols, plot + 1)
                plt.imshow(item['data'][int(nz / 2), :, :], cmap='gray')
                plt.title(item['title'])
                plt.xlabel(item['xLabel'])
                plt.ylabel(item['yLabel'])
            elif item['widget'] == 'curve':
                plt.subplot(rows, cols, plot + 1)
                n = 0
                for y_data in item['yData']:
                    if isinstance(item['xData'], list):
                        plt.plot(item['xData'][n], y_data, label=item['legend'][n])
                    else:
                        plt.plot(item['xData'], y_data, label=item['legend'][n])
                    n += 1
                plt.title(item['title'])
                plt.xlabel(item['xLabel'])
                plt.ylabel(item['yLabel'])
                plt.legend()
            plot += 1

        # Set the figure title
        plt.suptitle(self.mapVals['fileName'])

        # Adjust the layout to prevent overlapping titles
        plt.tight_layout()

        # Show the plot
        plt.show()

    def getParameter(self, key):
        """
        Get the value of a parameter.

        Args:
            key (str): The key corresponding to the parameter.

        Returns:
            Any: The value of the parameter associated with the given key.

        """
        return self.mapVals[key]

    def setParameter(self, key=True, string=True, val=True, unit=True):
        """
        Set the value of a parameter.

        Args:
            key (str): The key corresponding to the parameter.
            string (str): String that will be shown in the GUI
            val (Any): The new value to be assigned to the parameter.
            unit (bool): The unit of the parameter.

        Returns:
            None

        """
        self.mapVals[key] = val
        self.mapNmspc[key] = string
        self.map_units[key] = unit

    def autoProcessing(self, sampled, k_space):
        """
        Perform automated processing on k-space data.

        This method performs a series of processing steps on the k-space data to generate an image.
        The steps include inverse FFT, BM4D filtering, direct FFT, Cosbell filtering, zero-padding, and inverse FFT.

        Args:
            sampled (ndarray): The sampled k-space data.
            k_space (ndarray): The k-space data to be processed.

        Returns:
            ndarray: The processed image generated from the k-space data after automated processing.
        """
        # Perform inverse FFT to reconstruct the image in the spatial domain
        image = self.runIFFT(k_space)

        # Apply the BM4D filter to denoise the image
        image = self.runBm4dFilter(np.abs(image))

        # Perform direct FFT to transform the denoised image back to k-space
        k_sp = self.runDFFT(image)

        # Apply the Cosbell filter to k-space data in three directions
        k_sp_cb = self.runCosbellFilter(sampled, k_sp, 0.5)

        # Perform zero-padding on the Cosbell-filtered k-space data
        k_sp_zp = self.runZeroPadding(k_sp_cb, np.array([2, 2, 2]))

        # Perform inverse FFT to reconstruct the final image
        image = self.runIFFT(k_sp_zp)

        return image


