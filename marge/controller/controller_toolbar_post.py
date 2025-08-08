import copy
import os
import sys 
import ctypes
import scipy as sp
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QLabel, QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, \
    QWidget, QTabWidget
from scipy.interpolate import griddata
from marge.widgets.widget_toolbar_post import ToolBarWidgetPost
from controller.controller_plot3d import Plot3DController as Spectrum3DPlot
from PyQt5 import QtCore
from scipy.io import loadmat
import h5py
import ismrmrd
import pyqtgraph as pg
import marge.configs.hw_config as hw # Import the scanner hardware config


class ToolBarControllerPost(ToolBarWidgetPost):
    """
    Controller class for the ToolBarWidget.

    Inherits from ToolBarWidget to provide additional functionality for managing toolbar actions.

    Attributes:
        k_space_raw (ndarray): Raw k-space data loaded from a .mat file.
        mat_data (dict): Data loaded from a .mat file.
        nPoints (ndarray): Array containing the number of points in each dimension.
        k_space (ndarray): Processed k-space data.
        image_loading_button: QPushButton for loading the file and getting the k-space.
    """

    def __init__(self, main_window, *args, **kwargs):
        """
        Initialize the ToolBarController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(ToolBarControllerPost, self).__init__(*args, **kwargs)
        

        # Connect the image_loading_button clicked signal to the rawDataLoading method
        self.main_window = main_window
        self.k_space_raw = None
        self.mat_data = None
        self.nPoints = None
        self.k_space = None
        self.image_data = None
        self.action_load.triggered.connect(self.rawDataLoading)
        self.action_loadrmd.triggered.connect(self.mrdDataLoading)
        self.action_printrmd.triggered.connect(self.mrdDataShow) 
        self.action_convert.triggered.connect(self.convert)
        self.acq = ismrmrd.Acquisition()
        self.img = ismrmrd.Image()
        self.header= ismrmrd.xsd.ismrmrdHeader() 
        self.current_slice = 0
        
        
    def rawDataLoading(self, file_path=None, file_name=None):
        """
        Load raw data from a .mat file and update the image view widget.
        """
        # self.clearCurrentImage()
        # Prompt the user to select a .mat file
        if not file_path:
            file_path = self.loadmatFile()
            file_name = os.path.basename(file_path)
        else:
            file_path = file_path+file_name
        self.main.file_name = file_name
        self.mat_data = sp.io.loadmat(file_path)
        self.nPoints = np.reshape(self.mat_data['nPoints'], -1)

        if self.mat_data['seqName'] == 'PETRA':
            print("Executing regridding...")

            kCartesian = self.mat_data['kCartesian']
            self.k_space_raw = self.mat_data['kSpaceRaw']

            kxOriginal = np.reshape(np.real(self.k_space_raw[:, 0]), -1)
            kyOriginal = np.reshape(np.real(self.k_space_raw[:, 1]), -1)
            kzOriginal = np.reshape(np.real(self.k_space_raw[:, 2]), -1)
            kxTarget = np.reshape(kCartesian[:, 0], -1)
            kyTarget = np.reshape(kCartesian[:, 1], -1)
            kzTarget = np.reshape(kCartesian[:, 2], -1)
            valCartesian = griddata((kxOriginal, kyOriginal, kzOriginal), np.reshape(self.k_space_raw[:, 3], -1),
                                    (kxTarget, kyTarget, kzTarget), method="linear", fill_value=0, rescale=False)

            self.k_space = np.reshape(valCartesian, (self.nPoints[2], self.nPoints[1], self.nPoints[0]))

        else:  # Cartesian
            # Extract the k-space data from the loaded .mat file
            self.k_space_raw = self.mat_data['sampled'] ## on recupere pas kspace3d ni dataFull mais sampled
            self.k_space = np.reshape(self.k_space_raw[:, 3], self.nPoints[-1::-1])

            # Clear the console, history widget, history controller, and history dictionaries
            self.main.visualisation_controller.clear2DImage()

        # Update the main matrix of the image view widget with the k-space data
        self.main.image_view_widget.main_matrix = self.k_space

        # Update the image view widget to display the new main matrix
        try:
            image = np.log10(np.abs(self.main.image_view_widget.main_matrix))
            image[image == -np.inf] = np.inf
            val = np.min(image[:])
            image[image == np.inf] = val
        except:
            image = np.abs(self.main.image_view_widget.main_matrix)

        # Create figure widget
        image2show, x_label, y_label, title = self.fixImage(image, orientation=self.mat_data['axesOrientation'][0])
        image = Spectrum3DPlot(main=self.main,
                               data=image2show,
                               x_label=x_label,
                               y_label=y_label,
                               title=title)

        # Set window title
        self.main.setWindowTitle(self.mat_data['fileName'][0])

        # Delete all widgets from image_view_widget
        self.main.image_view_widget.clearFiguresLayout()

        # Create label widget
        label = QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("background-color: black;color: white")
        self.main.image_view_widget.addWidget(label, row=0, col=0, colspan=2)
        label.setText(file_path)

        # Add widgets to the figure layout
        self.main.image_view_widget.addWidget(label, row=0, col=0)
        self.main.image_view_widget.addWidget(image, row=1, col=0)

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="KSpace .mat",
                                          image=self.main.image_view_widget.main_matrix,
                                          orientation=self.mat_data['axesOrientation'][0],
                                          operation="KSpace",
                                          space="k")

    def loadmatFile(self):
        """
        Open a file dialog to select a .mat file and return its path.

        Returns:
            str: The path of the selected .mat file.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        default_dir = "C:/Users/Portatil PC 6/PycharmProjects/pythonProject1/Results"

        # Open the file dialog and prompt the user to select a .mat file
        file_name, _ = QFileDialog.getOpenFileName(self, "Select a .mat file", default_dir, "MAT Files (*.mat)",
                                                   options=options)

        return file_name
    
    def loadrmdFile(self):
    
        """
        Open a file dialog to select a .h5 file and return its path.

        Returns:
            str: The path of the selected .h5 file.
        """
        
        
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        default_dir = "C:/Users/Portatil PC 6/PycharmProjects/pythonProject1/Results"

        # Open the file dialog and prompt the user to select a .mat file
        file_name, _ = QFileDialog.getOpenFileName(self, "Select a .h5 file", default_dir, "HDF5 Files (*.h5)",
                                                   options=options)

        return file_name
    
    def mrdDataShow(self, file_path_rmd=None, file_name_rmd=None):
        """
        Load all data from a .h5 file and show the information about k_space and image.

        Returns : 
        None
        """
        
        if not file_path_rmd:
            file_path_rmd = self.loadrmdFile()
            file_name_rmd = os.path.basename(file_path_rmd)
        else:
            file_path_rmd = file_path_rmd+file_name_rmd
        self.main.file_name_rmd = file_name_rmd
        
        self.main_window.show()
        
        self.load_data(file_path_rmd)
        self.load_image_data(file_path_rmd)
        
        self.main_window.tableWidget2.cellClicked.connect(self.main_window.display_image)
         
    def mrdDataLoading(self, file_path_rmd=None, file_name_rmd=None):
        """
        Load raw data from a .h5 file and update the image view widget.

        Note : 
        This code will verify if the .h5 file is from a conversion (.mat to .h5) or directly from an acquisition because both contains different information and data.
        """
        
        if not file_path_rmd:
            file_path_rmd = self.loadrmdFile()
            file_name_rmd = os.path.basename(file_path_rmd)
        else:
            file_path_rmd = file_path_rmd+file_name_rmd
        self.main.file_name_rmd = file_name_rmd
        
        self.load_data(file_path_rmd) #load data_rearranged
        data=self.data #put data_rearanged into data
        
        if "from_mat" in file_name_rmd : # file created after a conversion
            nPhases=self.ntotPhases
            nSlices=self.ntotSlices
            data3d=np.zeros((nSlices, nPhases, data[0].shape[0]))
            count=0
            while count<data.shape[0]:
                for slice in range(nSlices):
                    for phase in range(nPhases):
                        data3d[slice, phase, :]=data[count]
                        count+=1
                            
            self.data3draw=data3d
            nReadout = data3d.shape[2]
            complex_data3d=np.zeros((nSlices, nPhases, nReadout//2), dtype=complex)
            
            for slice in range(nSlices):
                for phase in range(nPhases):
                    for readout in range(nReadout // 2):
                        real_part = data3d[slice, phase, 2 * readout]
                        imag_part = data3d[slice, phase, 2 * readout + 1]
                        complex_data3d[slice, phase, readout] = complex(real_part, imag_part)
            self.data3d = complex_data3d             
            self.data3dabs = np.abs(complex_data3d)    
                       
                            
        else : #file created after an acquisition
            nPhases=self.ntotPhases
            nSlices=self.ntotSlices
            nScans=self.ntotScans
            
            data4d=np.zeros((nScans, nSlices, nPhases, data[0].shape[0]))
            count=0
            while count<data.shape[0]:
                for scan in range (nScans):
                    for slice in range(nSlices):
                        for phase in range(nPhases):
                            data4d[scan, slice, phase, :]=data[count]
                            count+=1
                            
            self.data4d=data4d
            nReadout = data4d.shape[3]
            addRdPoints = hw.addRdPoints
            complex_data4d=np.zeros((nScans, nSlices, nPhases, (nReadout-2*addRdPoints)//2), dtype=complex)
            ## Delete addRdPoints before and after
            for scan in range(nScans):
                for slice in range(nSlices):
                    for phase in range(nPhases):
                        for readout in range((nReadout-2*addRdPoints) // 2):
                            real_part = data4d[scan, slice, phase, 2 * (readout + addRdPoints)]
                            imag_part = data4d[scan, slice, phase, 2 * (readout + addRdPoints) + 1]
                            complex_data4d[scan, slice, phase, readout] = complex(real_part, imag_part)
            
            self.complex_data4d=complex_data4d
            self.data3d=np.mean(complex_data4d, axis=0) #mean over scans
            self.data3dabs = np.abs(self.data3d) #abs value
            
            
        self.main.image_view_widget.main_matrix = self.data3d ## not abs
        
        image2show, x_label, y_label, title = self.fixImage(self.data3dabs) #abs
        image = Spectrum3DPlot(main=self.main,
                        data=image2show,
                        x_label=x_label,
                        y_label=y_label,
                        title=title)
        self.main.setWindowTitle("ISMRMRD data")

        # Delete all widgets from image_view_widget
        self.main.image_view_widget.clearFiguresLayout()

        # Create label widget
        label = QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("background-color: black;color: white")
        self.main.image_view_widget.addWidget(label, row=0, col=0, colspan=2)
        label.setText(file_path_rmd)

        # Add widgets to the figure layout
        self.main.image_view_widget.addWidget(label, row=0, col=0)
        self.main.image_view_widget.addWidget(image, row=1, col=0)
        self.main.history_list.addNewItem(stamp="KSpace .h5",
                                        image=self.main.image_view_widget.main_matrix,
                                        orientation=None,
                                        operation="KSpace",
                                        space="k")
       
    def load_data(self, file_name): 

        """
        Load MRI data and header information from the specified HDF5 file,
        rearranges the data if necessary, and populates a table in the main window with the loaded data.
    
        Returns:
        None
        """
        
        f = h5py.File(file_name, 'r')
        group = f['dataset']  
        
        header_data = group['data']['head'][()]
        data = group['data']['data'][()]
        
        
        if "from_mat" in file_name : 
            phase_ind = np.zeros(len(header_data), dtype=int)
            nSlices = np.zeros(len(header_data), dtype=int)
            for i in range(len(header_data)):
                phase_ind[i] = header_data[i][21][0]
                nSlices[i] = header_data[i][21][3]
            
            ntotSlices = max(nSlices)
            ntotPhases = max(phase_ind) 
            
            self.ntotSlices = ntotSlices
            self.ntotPhases = ntotPhases
            
            self.data = data
            self.header_data = header_data
              
            # No need to rearrange data because kSpace3D is already rearranged

        
        else : 
            phase_ind = np.zeros(len(header_data), dtype=int)
            nSlices = np.zeros(len(header_data), dtype=int)
            nScans = np.zeros(len(header_data), dtype=int)
            for i in range(len(header_data)):
                phase_ind[i] = header_data[i][21][0]
                nSlices[i] = header_data[i][21][3]
                nScans[i] = header_data[i][21][2]
            
            ntotSlices = max(nSlices)
            ntotScans = max(nScans)
            ntotPhases = max(phase_ind) 
            
            self.ntotSlices = ntotSlices
            self.ntotScans = ntotScans
            self.ntotPhases = ntotPhases
            
            # Rearrange data if necessary
            
            data_rearranged = np.zeros_like(data)
            header_rearranged = np.zeros_like(header_data)
            phase_counters = [0] * ntotPhases*ntotSlices*ntotScans
            
            for scan in range(ntotScans):
                for slice in range(ntotSlices):
                    for phase in range(ntotPhases):
                        ## suppr les addrdpoints premiers et derniers
                        index = phase + ntotPhases * (slice + ntotSlices * scan)
                        phase_index = phase_ind[phase]-1
                        position = phase_index + ntotPhases * (slice + ntotSlices * scan)
                        data_rearranged[position] = data[index]
                        header_rearranged[position] = header_data[index]
                        phase_counters[position] += 1
            
            
            self.data = data_rearranged
            self.header_data = header_rearranged
            
        
        fields = [field[0] for field in ismrmrd.AcquisitionHeader._fields_]
        
        f.close()
        
        self.populate_table(self.main_window.tableWidget1, self.header_data, self.data, fields)
        
    def load_image_data(self, file_name): 

        """
        Load MRI image data and header information from the specified HDF5 file,
        and populates a table in the main window with the loaded data.
    
        Returns:
        None
        """
        
        f = h5py.File(file_name, 'r')
        group = f['dataset']  
        header_image = group['image_raw']['header'][()]
        image = group['image_raw']['data'][()]
        
        
        
        ntotSlices = image.shape[0]
        ntotPhases = image.shape[3]
        ntotReadouts = image.shape[4]
        
        image_reshaped = np.reshape(image, (ntotSlices, ntotPhases, ntotReadouts))
        image_data = np.zeros_like(image_reshaped)
        
        for slice in range(ntotSlices):
            image_data[slice, : , :] = image_reshaped[slice, :,:]
            
        
        self.image_data=image_data
        self.main_window.initialize(self.image_data)
                   
        fields = [field[0] for field in ismrmrd.ImageHeader._fields_]
        f.close()
        
        self.populate_table(self.main_window.tableWidget2, header_image, image_data, fields)
    
    def populate_table(self, tableWidget, headers, data, fields):
        """
        Populate a QTableWidget with MRI header and data information.
    
        This method sets up the table widget to display MRI header information and corresponding data.
        Each row represents a header entry and its associated data.
    
        Parameters:
        - headers (array-like): The header information to display.
        - data (array-like): The data corresponding to the headers (data from k-space and from image)
        - fields (list): List of field names for the headers.
    
        Returns:
        None
        """
        
        tableWidget.setRowCount(len(headers))
        tableWidget.setColumnCount(len(headers[0]) + 1)  # +1 for data
    
        
        tableWidget.setHorizontalHeaderLabels(list(fields) + ['Data'])
        
        
        for row in range(len(headers)):
            for col in range(len(headers[row])):
                tableWidget.setItem(row, col, QTableWidgetItem(str(headers[row][col])))
            tableWidget.setItem(row, len(headers[row]), QTableWidgetItem(str(data[row])))
    
    def fixImage(self, matrix3d, orientation=None):
        matrix = copy.copy(matrix3d)
        axes = orientation
        if axes is None : # No orientation for h5 test 
            title = "No orientation"
            matrix=matrix
            x_label = "X"
            y_label = "Y"
        elif axes[2] == 2:  # Sagittal
            title = "Sagittal"
            if axes[0] == 0 and axes[1] == 1:  # OK
                matrix = np.flip(matrix, axis=2)
                matrix = np.flip(matrix, axis=1)
                x_label = "(-Y) A | PHASE | P (+Y)"
                y_label = "(-X) I | READOUT | S (+X)"
            else:
                matrix = np.transpose(matrix, (0, 2, 1))
                matrix = np.flip(matrix, axis=2)
                matrix = np.flip(matrix, axis=1)
                x_label = "(-Y) A | READOUT | P (+Y)"
                y_label = "(-X) I | PHASE | S (+X)"
        elif axes[2] == 1:  # Coronal
            title = "Coronal"
            if axes[0] == 0 and axes[1] == 2:  # OK
                matrix = np.flip(matrix, axis=2)
                matrix = np.flip(matrix, axis=1)
                matrix = np.flip(matrix, axis=0)
                x_label = "(+Z) R | PHASE | L (-Z)"
                y_label = "(-X) I | READOUT | S (+X)"
            else:
                matrix = np.transpose(matrix, (0, 2, 1))
                matrix = np.flip(matrix, axis=2)
                matrix = np.flip(matrix, axis=1)
                matrix = np.flip(matrix, axis=0)
                x_label = "(+Z) R | READOUT | L (-Z)"
                y_label = "(-X) I | PHASE | S (+X)"
        elif axes[2] == 0:  # Transversal
            title = "Transversal"
            if axes[0] == 1 and axes[1] == 2:
                matrix = np.flip(matrix, axis=2)
                matrix = np.flip(matrix, axis=1)
                x_label = "(+Z) R | PHASE | L (-Z)"
                y_label = "(+Y) P | READOUT | A (-Y)"
            else:  # OK
                matrix = np.transpose(matrix, (0, 2, 1))
                matrix = np.flip(matrix, axis=2)
                matrix = np.flip(matrix, axis=1)
                x_label = "(+Z) R | READOUT | L (-Z)"
                y_label = "(+Y) P | PHASE | A (-Y)"
       
        
        return matrix, x_label, y_label, title
    
   
    def convert (self, file_path = None, file_name =None):
        
        
        """
        Convert .mat to .h5
        WARNING : 
        - h5 file will contain less information than if it were created during acquisition (kSpace3D instead of dataFull)
        - Different cases for RARE and GRE3D, if a new seq is added, please update this code with an elif case.

        Returns : 
        None. An .h5 file is created 
        """
        
        if not file_path:
            file_path = self.loadmatFile()
            file_name = os.path.basename(file_path)
        else:
            file_path = file_path+ file_name
        
        mat = loadmat(file_path)
        mat = {k: np.squeeze(v) for k, v in mat.items()}
        
        mat_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        file_parts = file_name.split('.')
        file_parts.insert(1, 'from_mat')
        file_parts.remove('mat')
        new_file_name = '.'.join(file_parts)
        
        h5_file_path = os.path.join(mat_dir, new_file_name + '.h5')
        sequence_type = file_name.split('.')[0]
        
        if os.path.exists(h5_file_path):
            os.remove(h5_file_path)
        
        dset = ismrmrd.Dataset(h5_file_path, f'/dataset', True)  # Créer le dataset
        
        nRD, nPH, nSL = mat['nPoints'][0], mat['nPoints'][1], mat['nPoints'][2]
        nScans = int(mat['nScans'])
        bw =float(mat['bw'])
        
        if sequence_type == 'RARE':
            etl = int(mat['etl'])
            ind = self.getIndex(etl, nPH, int(mat['sweepMode']))
            nRep = (nPH // etl) * nSL
            
        axesOrientation = mat['axesOrientation']
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
              
        
        dset.write_xml_header(self.header.toXML()) # Write the header to the dataset
        addRdPoints = int(mat['addRdPoints'])       
        
        if sequence_type == 'RARE':
            new_data = np.zeros((nPH * nSL, nRD)) 
            new_data = np.reshape(mat['kSpace3D'], (nSL, nPH, nRD))
        
            counter=0  
            
            for slice_idx in range(nSL):
                for phase_idx in range(nPH):
                    
                    line = new_data[slice_idx, phase_idx, :]
                    line2d = np.reshape(line, (1, nRD))
                    acq = ismrmrd.Acquisition.from_array(line2d, None)
                    
                    index_in_repetition = phase_idx % etl
                    current_repetition = (phase_idx // etl) + (slice_idx * (nPH // etl))
                    
                    acq.clearAllFlags()
                    
                    if index_in_repetition == 0: 
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_CONTRAST)
                    elif index_in_repetition == etl - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_CONTRAST)
                    
                    if phase_idx== 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_PHASE)
                    elif phase_idx == nPH - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_PHASE)
                    
                    if slice_idx == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                    elif slice_idx == nSL - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                        
                    if int(current_repetition) == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_REPETITION)
                    elif int(current_repetition) == nRep - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)
                    
                    
                    counter += 1 
                    
                    # +1 to start at 1 instead of 0
                    acq.idx.repetition = int(current_repetition + 1)
                    acq.idx.kspace_encode_step_1 = phase_idx+1 # phase
                    acq.idx.slice = slice_idx + 1
                    acq.idx.contrast = index_in_repetition + 1
                    
                    acq.scan_counter = counter
                    # acq.discard_pre = addRdPoints
                    # acq.discard_post = addRdPoints #######""
                    acq.sample_time_us = 1/bw
                    acq.position=(ctypes.c_float * 3)(*mat['dfov']) 
                    
                    acq.read_dir = (ctypes.c_float * 3)(*read_dir)
                    acq.phase_dir = (ctypes.c_float * 3)(*phase_dir)
                    acq.slice_dir = (ctypes.c_float * 3)(*slice_dir)
                    
                    dset.append_acquisition(acq) # Append the acquisition to the dataset   
                
        elif sequence_type == 'GRE3D':
            new_data = np.reshape(mat['kSpace3D'], (nSL, nPH, nRD)) 
            counter = 0
        
            for slice_idx in range(nSL):
                for phase_idx in range(nPH):
                        line = new_data[slice_idx, phase_idx, :]
                        line2d = np.reshape(line, (1, nRD))
                        acq = ismrmrd.Acquisition.from_array(line2d, None)
                        
                        counter += 1
                    
                        acq.idx.repetition = counter
                        acq.idx.kspace_encode_step_1 = phase_idx + 1
                        acq.idx.slice = slice_idx + 1
                
                        
                        acq.clearAllFlags()
                        
                        if phase_idx == 0:
                            acq.setFlag(ismrmrd.ACQ_FIRST_IN_PHASE)
                        elif phase_idx == nPH - 1:
                            acq.setFlag(ismrmrd.ACQ_LAST_IN_PHASE)
                        
                        if slice_idx == 0:
                            acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                        elif slice_idx == nSL - 1:
                            acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                        
                        if counter == 1:
                            acq.setFlag(ismrmrd.ACQ_FIRST_IN_AVERAGE)
                        elif counter == nPH*nSL:
                            acq.setFlag(ismrmrd.ACQ_LAST_IN_AVERAGE)
                        
                        
                        acq.scan_counter = counter
                        # acq.discard_pre = hw.addRdPoints
                        # acq.discard_post = hw.addRdPoints
                        acq.sample_time_us = 1/bw
                        acq.position=(ctypes.c_float * 3)(*mat['dfov'])  
                        acq.read_dir = (ctypes.c_float * 3)(*read_dir) 
                        acq.phase_dir = (ctypes.c_float * 3)(*phase_dir)
                        acq.slice_dir = (ctypes.c_float * 3)(*slice_dir)
                        
                        
                        # Ajouter l'acquisition au dataset
                        dset.append_acquisition(acq)
        
        image=mat['image3D']
        image_reshaped = np.reshape(image, (nSL, nPH, nRD))
        
        #for scan in range (nScans): ## image3d does not have scan dimension
        for slice_idx in range (nSL):
            
            image_slice = image_reshaped[slice_idx, :, :]
            img = ismrmrd.Image.from_array(image_slice)
            
            
            img.field_of_view = (ctypes.c_float * 3)(*(mat['fov'])*10) # mm
            img.position = (ctypes.c_float * 3)(*mat['dfov'])
            img.sample_time_us = 1/bw
            img.image_type = 5 ## COMPLEX
            
            
            
            img.read_dir = (ctypes.c_float * 3)(*read_dir)
            img.phase_dir = (ctypes.c_float * 3)(*phase_dir)
            img.slice_dir = (ctypes.c_float * 3)(*slice_dir)


            
            dset.append_image(f"image_raw", img)
            
        
        dset.close()       
        
        
        
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
######################################################################################################
    
    
class MainWindow_toolbar(QMainWindow):
    def __init__(self):
        super().__init__()    
        self.setWindowTitle("Show ISMRMRD data")
        self.setGeometry(100, 100, 800, 600)
        
        self.tabWidget = QTabWidget()
        self.setCentralWidget(self.tabWidget)
        
        self.tableWidget1 = QTableWidget()
        self.tableWidget2 = QTableWidget()
        
        self.tabWidget.addTab(self.tableWidget1, "k-Space data")
        self.tabWidget.addTab(self.tableWidget2, "Image data")
                
        
        self.rawPlot = pg.PlotWidget()
        self.rawPlot.hide()
        self.trajPlot = pg.PlotWidget()
        self.trajPlot.hide()
        
        self.imagePlot = pg.ImageView()
        self.annotationLabel = QLabel()
    
        
        self.imageWidget = QWidget()
        self.imageLayout = QVBoxLayout()
        self.imageLayout.addWidget(self.imagePlot)
        self.imageLayout.addWidget(self.annotationLabel)
        self.imageWidget.setLayout(self.imageLayout)
        
        self.tabWidget.addTab(self.imageWidget, "Image")
        
        
    def initialize(self, data_from_other_class):
        self.image_data = data_from_other_class
        
    def display_image(self, row, column):

            """
            Display the MRI image data for a specific slice selected by clicking in the array.
        
            Returns:
            None. It displays an image of the slice selected in an oter tab.
            """
            
            data= self.image_data[row]
            nReadout = data.shape[1]
            ntotPhases = data.shape[0]
            
            real_parts = np.zeros((ntotPhases, nReadout))
            imag_parts = np.zeros((ntotPhases, nReadout))
            image = np.zeros((ntotPhases, nReadout))
            
            # Re and Im parts
            for i in range(ntotPhases):
                for j in range(nReadout):
                    real_parts[i, j] = data[i, j][0]
                    imag_parts[i, j] = data[i, j][1]
            
            for i in range(ntotPhases):
                for j in range(nReadout):
                    image[i, j] = np.abs(complex(real_parts[i,j], imag_parts[i,j]))
            
            # Display the image
            self.imagePlot.setImage(image)
            self.annotationLabel.setText(f'Slice {row + 1}')
    
######################################################################################################
   
    
if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = MainWindow_toolbar()
    

    toolbar_controller = ToolBarControllerPost(main_window)
    
    sys.exit(app.exec_())
    
    
    
    
    
    
    
    
