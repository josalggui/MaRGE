import threading
import numpy as np
from widgets.widget_preprocessing import PreProcessingTabWidget


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
        self.image_cosbell_button.clicked.connect(self.cosbellFilter)
        self.image_padding_button.clicked.connect(self.zeroPadding)
        self.new_fov_button.clicked.connect(self.fovShifting)

    def cosbellFilter(self):
        """
        Apply the Cosbell filter operation using threading.

        Starts a new thread to execute the runCosbellFilter method.
        """
        # Send printed text to the corresponding console
        self.main.console.setup_console()

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
        # Send printed text to the corresponding console
        self.main.console.setup_console()

        thread = threading.Thread(target=self.runZeroPadding)
        thread.start()

    def runZeroPadding(self):
        """
        Run the zero-padding operation.

        Retrieves the necessary parameters and performs the zero-padding on the loaded image.
        Updates the main matrix of the image view widget with the padded image, adds the operation to the history
        widget, and updates the operations history.
        """
        # Zero-padding order for each dimension from the text field
        zero_padding_order = self.zero_padding_order_field.text().split(',')
        rd_order = int(zero_padding_order[0])
        ph_order = int(zero_padding_order[1])
        sl_order = int(zero_padding_order[2])

        # Get the k_space data and its shape
        k_space = self.main.image_view_widget.main_matrix
        current_shape = k_space.shape

        # Determine the new shape after zero-padding
        new_shape = current_shape[0] * sl_order, current_shape[1] * ph_order, current_shape[2] * rd_order

        # Create an image matrix filled with zeros
        image_matrix = np.zeros(new_shape, dtype=complex)

        # Get the dimensions of the current image
        image_height = current_shape[0]
        image_width = current_shape[1]
        image_depth = current_shape[2]

        # Calculate the centering offsets for each dimension
        col_offset = (new_shape[0] - image_height) // 2
        row_offset = (new_shape[1] - image_width) // 2
        depth_offset = (new_shape[2] - image_depth) // 2

        # Calculate the start and end indices to center the k_space within the image_matrix
        col_start = col_offset
        col_end = col_start + image_height
        row_start = row_offset
        row_end = row_start + image_width
        depth_start = depth_offset
        depth_end = depth_start + image_depth

        # Copy the k_space into the image_matrix at the center
        image_matrix[col_start:col_end, row_start:row_end, depth_start:depth_end] = k_space

        # Update the main matrix of the image view widget with the padded image
        self.main.image_view_widget.main_matrix = image_matrix

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="Zero Padding",
                                          image=self.main.image_view_widget.main_matrix,
                                          orientation=self.main.toolbar_image.mat_data['axesOrientation'][0],
                                          operation="Zero Padding - RD: " + str(rd_order) + ", PH: "
                                                    + str(ph_order) + ", SL: " + str(sl_order),
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
