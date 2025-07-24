import threading
import numpy as np
from marge.widgets.widget_preprocessing import PreProcessingTabWidget


class PreProcessingTabController(PreProcessingTabWidget):
    """
    Controller class for the pre-processing tab widget.

    Inherits from PreProcessingTabWidget.

    Attributes:
        partial_reconstruction_button (QPushButton): QPushButton for applying partial reconstruction.
        image_cosbell_button (QPushButton): QPushButton for applying Cosbell filter.
        phase_center_button (QPushButton): QPushButton for getting the phase center.
        image_padding_button (QPushButton): QPushButton for applying zero padding.
        new_fov_button (QPushButton): QPushButton for changing the field of view.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the PreProcessingTabController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(PreProcessingTabController, self).__init__(*args, **kwargs)

        # Connect the button click signal to the corresponding methods
        self.scan_button.clicked.connect(self.selectScans)
        self.image_cosbell_button.clicked.connect(self.cosbellFilter)
        self.image_padding_button.clicked.connect(self.zeroPadding)
        self.new_fov_button.clicked.connect(self.fovShifting)

    def selectScans(self):
        # Define method to get the indexes
        def parse_indexes(index_string, max_index):
            if index_string.strip().lower() == "all":
                return list(range(max_index))

            indexes = []

            for part in index_string.split(','):
                part = part.strip()
                if ':' in part:
                    start, end = part.split(':')
                    start = int(start.strip())
                    end = int(end.strip())
                    indexes.extend(range(start, end))
                else:
                    indexes.append(int(part))

            return indexes

        # Get the mat data
        mat_data = self.main.toolbar_image.mat_data

        # Get the desired scans
        n_scans = mat_data['nScans'][0][0]
        scans = self.scan_field.text()
        print(" Selected scans: " + str(scans))
        scans = parse_indexes(scans, n_scans)

        # Generate artifitial data_full for RareDoubleImage
        if mat_data['seqName'] == 'RareDoubleImage':
            data_odd = mat_data['data_full_odd_echoes']
            data_even = mat_data['data_full_even_echoes']
            s_scans, n_sl, n_ph, n_rd = np.shape(data_odd)
            n_points = np.reshape(mat_data['nPoints'], -1)

            # Process odd data
            data_odd_temp = np.zeros((len(scans), n_points[2], n_points[1], n_points[0]), dtype=complex)
            data_odd_temp[:, 0:n_sl, :, :] = data_odd[scans, :, :, :]
            data_odd = np.average(data_odd_temp, axis=0)
            img_odd = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(data_odd)))

            # Process even data
            data_even_temp = np.zeros((len(scans), n_points[2], n_points[1], n_points[0]), dtype=complex)
            data_even_temp[:, 0:n_sl, :, :] = data_even[scans, :, :, :]
            data_even = np.average(data_even_temp, axis=0)
            img_even = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(data_even)))

            # Get average k-space from odd and even image
            img = (np.abs(img_odd) + np.abs(img_even)) / 2
            data = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))
        else:
            # Get the selected scans from dataFull and average
            data_full = mat_data['dataFull'][scans, :, :, :]
            s_scans, n_sl, n_ph, n_rd = np.shape(data_full)
            n_points = np.reshape(mat_data['nPoints'], -1)
            data_temp = np.zeros((len(scans), n_points[2], n_points[1], n_points[0]), dtype=complex)
            data_temp[:, 0:n_sl, :, :] = data_full
            data_temp = np.average(data_temp, axis=0)
            data = np.reshape(data_temp, (1, n_points[0] * n_points[1] * n_points[2]))

        # Input the resulting data into the k-space
        self.main.toolbar_image.k_space_raw[:, 3] = np.reshape(data, -1)
        self.main.toolbar_image.k_space = np.reshape(self.main.toolbar_image.k_space_raw[:, 3], n_points[-1::-1])

        # Update the main matrix of the image view widget with the cosbell data
        self.main.image_view_widget.main_matrix = self.main.toolbar_image.k_space.copy()

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="Selected scans",
                                          image=self.main.image_view_widget.main_matrix,
                                          orientation=self.main.toolbar_image.mat_data['axesOrientation'][0],
                                          operation="Scans: " + str(scans),
                                          space="k",
                                          image_key=self.main.image_view_widget.image_key)


    def cosbellFilter(self):
        """
        Apply the Cosbell filter operation using threading.

        Starts a new thread to execute the runCosbellFilter method.
        """

        thread = threading.Thread(target=self.runCosbellFilter)
        thread.start()

    def runCosbellFilter(self):
        """
        Run the Cosbell filter operation.

        Retrieves the necessary parameters and performs the Cosbell filtering on the loaded image.
        Updates the main matrix of the image view widget with the filtered data, adds the operation to the history widget,
        and updates the operations history.
        """
        # Cosbell filter order from the text field
        cosbell_order = float(self.cosbell_order_field.text())

        # Get the mat data from the loaded .mat file in the main toolbar controller
        mat_data = self.main.toolbar_image.mat_data
        sampled = self.main.toolbar_image.k_space_raw
        data = self.main.image_view_widget.main_matrix.copy()
        nPoints = np.reshape(mat_data['nPoints'], -1)

        # Check which checkboxes are selected
        theta = None
        text = "Cosbell -"
        if self.readout_checkbox.isChecked():
            k = np.reshape(sampled[:, 0], nPoints[-1::-1])
            kmax = np.max(np.abs(k[:]))
            text += ' RD,'
            theta = k / kmax
            data *= (np.cos(theta * (np.pi / 2)) ** cosbell_order)
        if self.phase_checkbox.isChecked():
            k = np.reshape(sampled[:, 1], nPoints[-1::-1])
            kmax = np.max(np.abs(k[:]))
            text += ' PH,'
            theta = k / kmax
            data *= (np.cos(theta * (np.pi / 2)) ** cosbell_order)
        if self.slice_checkbox.isChecked():
            k = np.reshape(sampled[:, 2], nPoints[-1::-1])
            kmax = np.max(np.abs(k[:]))
            text += ' SL,'
            theta = k / kmax
            data *= (np.cos(theta * (np.pi / 2)) ** cosbell_order)

        # Update the main matrix of the image view widget with the cosbell data
        self.main.image_view_widget.main_matrix = data.copy()

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="Cosbell",
                                          image=data,
                                          orientation=self.main.toolbar_image.mat_data['axesOrientation'][0],
                                          operation=text + " Order: " + str(cosbell_order),
                                          space="k",
                                          image_key=self.main.image_view_widget.image_key)

    def zeroPadding(self):
        """
        Apply the zero-padding operation using threading.

        Starts a new thread to execute the runZeroPadding method.
        """

        thread = threading.Thread(target=self.runZeroPadding)
        thread.start()

    def runZeroPadding(self):
        """
        Run the zero-padding operation.

        Retrieves the necessary parameters and performs the zero-padding on the loaded image.
        Updates the main matrix of the image view widget with the padded image, adds the operation to the history
        widget, and updates the operations history.
        """
        # Get the k_space data and its shape
        k_space = self.main.image_view_widget.main_matrix.copy()
        shape_0 = k_space.shape

        # Determine the new shape after zero-padding
        matrix_size = self.zero_padding_order_field.text().split(',')
        n_rd = int(matrix_size[0]) * shape_0[2]
        n_ph = int(matrix_size[1]) * shape_0[1]
        n_sl = int(matrix_size[2]) * shape_0[0]
        shape_1 = n_sl, n_ph, n_rd

        # Create an image matrix filled with zeros
        image_matrix = np.zeros(shape_1, dtype=complex)

        # Calculate the centering offsets for each dimension
        offset_0 = (shape_1[0] - shape_0[0]) // 2
        offset_1 = (shape_1[1] - shape_0[1]) // 2
        offset_2 = (shape_1[2] - shape_0[2]) // 2

        # Calculate the start and end indices to center the k_space within the new image_matrix
        new_start_0 = offset_0 if offset_0 >= 0 else 0
        new_start_1 = offset_1 if offset_1 >= 0 else 0
        new_start_2 = offset_2 if offset_2 >= 0 else 0
        new_end_0 = new_start_0 + shape_0[0] if offset_0 > 0 else shape_1[0]
        new_end_1 = new_start_1 + shape_0[1] if offset_1 > 0 else shape_1[1]
        new_end_2 = new_start_2 + shape_0[2] if offset_2 > 0 else shape_1[2]

        # Calculate the start and end indices of old matrix
        old_start_0 = 0 if offset_0 >= 0 else -offset_0
        old_start_1 = 0 if offset_1 >= 0 else -offset_1
        old_start_2 = 0 if offset_2 >= 0 else -offset_2
        old_end_0 = shape_0[0] if offset_0 >=0 else old_start_0 + shape_1[0]
        old_end_1 = shape_0[1] if offset_1 >=0 else old_start_1 + shape_1[1]
        old_end_2 = shape_0[2] if offset_2 >=0 else old_start_2 + shape_1[2]

        # Copy the k_space into the image_matrix at the center
        image_matrix[new_start_0:new_end_0, new_start_1:new_end_1, new_start_2:new_end_2] = k_space[
                                                                                            old_start_0:old_end_0,
                                                                                            old_start_1:old_end_1,
                                                                                            old_start_2:old_end_2
                                                                                            ]

        # Update the main matrix of the image view widget with the padded image
        self.main.image_view_widget.main_matrix = image_matrix.copy()

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="Zero Padding",
                                          image=self.main.image_view_widget.main_matrix,
                                          orientation=self.main.toolbar_image.mat_data['axesOrientation'][0],
                                          operation="Zero Padding - RD: " + str(n_rd) + ", PH: "
                                                    + str(n_ph) + ", SL: " + str(n_sl),
                                          space="k",
                                          image_key=self.main.image_view_widget.image_key)

    def fovShifting(self):
        """
        Perform the FOV change operation using threading.

        Starts a new thread to execute the runFovShifting method.
        """
        thread = threading.Thread(target=self.runFovShifting)
        thread.start()

    def runFovShifting(self):
        """
        Run the FOV change operation.

        Retrieves the necessary parameters and performs the FOV change on the loaded image.
        Updates the main matrix of the image view widget with the new FOV image, adds the operation to the history
        widget, and updates the operations history.
        """
        # Get the k_space data and its shape
        k = self.main.toolbar_image.k_space_raw.copy()
        nPoints = self.main.toolbar_image.nPoints

        # Factors for FOV change from the text field
        factors = self.change_fov_field.text().split(',')

        # Extract the k_space components
        krd = k[:, 0]
        kph = k[:, 1]
        ksl = k[:, 2]
        k = np.column_stack((krd, kph, ksl))

        # Convert factors to spatial shifts
        delta_rd = (float(factors[0])) * 1e-3
        delta_ph = (float(factors[1])) * 1e-3
        delta_sl = (float(factors[2])) * 1e-3
        delta_r = np.array([delta_rd, delta_ph, delta_sl])

        # Calculate the phase shift using the spatial shifts
        phi = np.exp(-1j * 2 * np.pi * k @ np.reshape(delta_r, (3, 1)))
        self.main.image_view_widget.main_matrix *= np.reshape(phi, nPoints[-1::-1])

        # Add the "New FOV" operation to the history widget with a timestamp
        self.main.history_list.addNewItem(stamp="FOV shift",
                                          image=self.main.image_view_widget.main_matrix,
                                          orientation=self.main.toolbar_image.mat_data['axesOrientation'][0],
                                          operation="Fov shift",
                                          space="k",
                                          image_key=self.main.image_view_widget.image_key)
