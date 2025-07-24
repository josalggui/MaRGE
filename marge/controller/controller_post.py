import bm4d
import threading
import numpy as np
from scipy.ndimage import gaussian_filter
from marge.widgets.widget_post import PostProcessingTabWidget
from skimage.util import view_as_blocks
from skimage.measure import shannon_entropy


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
        print('BM4D is loading')

        # Get the absolute value of the main image matrix and convert it to float
        image_data = np.abs(self.main.image_view_widget.main_matrix).astype(float)
        # Rescale the image data to the range (0, 100)
        reference = np.max(image_data)
        image_rescaled = image_data/reference*100

        # Calculate the standard deviation (sigma_psd) for BM4D filter
        if self.auto_checkbox.isChecked():
            # Quantize image
            num_bins = 1000
            image_quantized = np.digitize(image_rescaled, bins=np.linspace(0, 1, num_bins + 1)) - 1


            # Divide the image into blocks
            n_multi = (np.array(image_quantized.shape) / 5).astype(int) * 5
            blocks_q = view_as_blocks(image_quantized[0:n_multi[0], 0:n_multi[1], 0:n_multi[2]], block_shape=(5, 5, 5))
            blocks_r = view_as_blocks(image_rescaled[0:n_multi[0], 0:n_multi[1], 0:n_multi[2]], block_shape=(5, 5, 5))

            # Calculate the standard deviation for each block
            block_std_devs = np.std(blocks_r, axis=(3, 4, 5))

            # Calculate the average value for each block
            block_mean = np.mean(blocks_r, axis=(3, 4, 5))

            # Calculate entropy for each block
            block_entropies = np.zeros_like(blocks_q[:, :, :, 0, 0, 0], dtype=np.float32)
            for ii in range(blocks_q.shape[0]):
                for jj in range(blocks_q.shape[1]):
                    for kk in range(blocks_q.shape[2]):
                        block = blocks_q[ii, jj, kk, :, :, :]
                        entropy = shannon_entropy(block)
                        block_entropies[ii, jj, kk] = entropy

            # Find the indices of the block with the highest entropy
            max_entropy_index = np.unravel_index(np.argmax(block_entropies), block_entropies.shape)

            # Find the indices of the block with the minimum mean
            min_mean_index = np.unravel_index(np.argmin(block_mean), block_mean.shape)

            # Extract the block with the highest entropy from the block_std_devs array
            std = 3 * block_std_devs[min_mean_index]
            print("Standard deviation for BM4D: %0.2f" % std)

        else:
            std = float(self.std_text_field.text())

        # Create a BM4D profile and set the stage argument and blockmatches options
        profile = bm4d.BM4DProfile()
        stage_arg = bm4d.BM4DStages.ALL_STAGES
        blockmatches = (False, False)

        # Apply the BM4D filter to the rescaled image
        denoised_rescaled = bm4d.bm4d(image_rescaled, sigma_psd=std, profile=profile, stage_arg=stage_arg,
                                      blockmatches=blockmatches)

        # Rescale the denoised image back to its original dimensions
        denoised_image = denoised_rescaled/100*reference

        # Update the main image view widget with the denoised image
        self.main.image_view_widget.main_matrix = denoised_image

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="BM4D",
                                          image=self.main.image_view_widget.main_matrix,
                                          orientation=self.main.toolbar_image.mat_data['axesOrientation'][0],
                                          operation="BM4D - Standard deviation: %0.2f" % std,
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)

        print('BM4D filter has been applied')

        # Update the space dictionary
        self.main.history_list.space[self.main.history_list.image_key] = 'i'

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

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="Gaussian",
                                          image=self.main.image_view_widget.main_matrix,
                                          orientation=self.main.toolbar_image.mat_data['axesOrientation'][0],
                                          operation="Gaussian - Standard deviation: " + str(sigma),
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)

        # Update the space dictionary
        self.main.history_list.space[self.main.history_list.image_key] = 'i'
