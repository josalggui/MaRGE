import bm4d
import threading
import numpy as np
from scipy.ndimage import gaussian_filter
from widgets.widget_postpocessing import PostProcessingTabWidget


class PostProcessingTabController(PostProcessingTabWidget):
    """
    Controller class for the post-processing tab widget.

    Inherits from PostProcessingTabWidget.

    Attributes:
        image_data (np.ndarray): The input image data.
        denoised_image (np.ndarray): The denoised image data.
        run_filter_button (QPushButton): QPushButton for applying BM4D filter.
        run_gaussian_button (QPushButton): QPushButton for applying Gaussian filter.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the PostProcessingTabController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(PostProcessingTabController, self).__init__(*args, **kwargs)

        # Connect the buttons to their corresponding methods
        self.run_filter_button.clicked.connect(self.bm4dFilter)
        self.run_gaussian_button.clicked.connect(self.gaussianFilter)

        self.image_data = None
        self.denoised_image = None

    def bm4dFilter(self):
        """
        Perform the BM4D filter operation using threading.

        Starts a new thread to execute the runBm4dFilter method.
        """
        thread = threading.Thread(target=self.runBm4dFilter)
        thread.start()

    def runBm4dFilter(self):
        """
        Run the BM4D filter operation.

        Retrieves the image data, performs rescaling, computes median and median absolute deviation (MAD),
        applies the BM4D filter to the rescaled image, restores the denoised image to its original dimensions,
        updates the main matrix of the image view widget, adds the operation to the history widget,
        and updates the operations history.
        """
        # Get the absolute value of the main image matrix and convert it to float
        image_data = np.abs(self.main.image_view_widget.main_matrix).astype(float)

        # Rescale the image data to the range (0, 100)
        image_rescaled = np.interp(image_data, (np.min(image_data), np.max(image_data)), (0, 100))

        # Compute the median and median absolute deviation (MAD) of the rescaled image
        med = np.median(image_rescaled)
        mad = np.median(np.abs(image_rescaled - med))

        # Calculate the standard deviation (sigma_psd) for BM4D filter
        if self.auto_checkbox.isChecked():
            sigma_psd = (1.4826 * mad) / 2
        else:
            std_value = float(self.std_text_field.text())
            sigma_psd = std_value

        self.main.console.print('BM4D is loading')

        # Create a BM4D profile and set the stage argument and blockmatches options
        profile = bm4d.BM4DProfile()
        stage_arg = bm4d.BM4DStages.ALL_STAGES
        blockmatches = (False, False)

        # Apply the BM4D filter to the rescaled image
        denoised_rescaled = bm4d.bm4d(image_rescaled, sigma_psd=sigma_psd, profile=profile, stage_arg=stage_arg,
                                      blockmatches=blockmatches)

        # Rescale the denoised image back to its original dimensions
        denoised_image = np.interp(denoised_rescaled, (np.min(denoised_rescaled), np.max(denoised_rescaled)),
                                   (np.min(image_data), np.max(image_data)))

        # Update the main image view widget with the denoised image
        self.main.image_view_widget.main_matrix = denoised_image

        # Add the operation to the history widget with a timestamp
        self.main.history_list.addItemWithTimestamp("BM4D")

        # Update the history dictionary with the denoised image data
        self.main.history_list.hist_dict[self.main.history_list.matrix_infos] = \
            self.main.image_view_widget.main_matrix

        # Update the operations history with the BM4D operation details
        self.main.history_list.updateOperationsHist(self.main.history_list.matrix_infos,
                                                          "BM4D - Standard deviation: " + str(sigma_psd))
        self.main.console.print('BM4D filter has been applied')

        # Update the space dictionary
        self.main.history_list.space[self.main.history_list.matrix_infos] = 'i'

    def gaussianFilter(self):
        """
        Perform the Gaussian filter operation using threading.

        Starts a new thread to execute the runGaussianFilter method.
        """
        thread = threading.Thread(target=self.runGaussianFilter)
        thread.start()

    def runGaussianFilter(self):
        """
        Run the Gaussian filter operation.

        Retrieves the image data, applies the Gaussian filter, restores the filtered image to its original dimensions,
        updates the main matrix of the image view widget, adds the operation to the history widget,
        and updates the operations history.
        """
        # Get the absolute value of the main image matrix and convert it to float
        image_data = np.abs(self.main.image_view_widget.main_matrix).astype(float)

        # Get the standard deviation (sigma) for the Gaussian filter
        sigma = float(self.gaussian_text_field.text())

        # Rescale the image data to the range (0, 100)
        image_rescaled = np.interp(image_data, (np.min(image_data), np.max(image_data)), (0, 100))

        # Apply the Gaussian filter to the rescaled image
        filtered_image = gaussian_filter(image_data, sigma=sigma)

        # Rescale the filtered image back to its original dimensions
        gaussian_image = np.interp(filtered_image, (np.min(filtered_image), np.max(filtered_image)),
                                   (np.min(image_data), np.max(image_data)))

        # Update the main image view widget with the filtered image
        self.main.image_view_widget.main_matrix = gaussian_image

        # Add the operation to the history widget with a timestamp
        self.main.history_list.addItemWithTimestamp("Gaussian")

        # Update the history dictionary with the filtered image data
        self.main.history_list.hist_dict[self.main.history_list.matrix_infos] = \
            self.main.image_view_widget.main_matrix

        # Update the operations history with the Gaussian operation details
        self.main.history_list.updateOperationsHist(self.main.history_list.matrix_infos,
                                                          "Gaussian - Standard deviation: " + str(sigma))

        # Update the space dictionary
        self.main.history_list.space[self.main.history_list.matrix_infos] = 'i'
