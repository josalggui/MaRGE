import threading
import bm4d
import numpy as np

from widgets.postpocessing_tab_widget import PostProcessingTabWidget


class PostProcessingTabController(PostProcessingTabWidget):
    """
    Controller class for the post-processing tab widget.

    Inherits from PostProcessingTabWidget.

    Attributes:
        image_data: The input image data.
        denoised_image: The denoised image data.
        run_filter_button: QPushButton for applying BM4D filter

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the PostProcessingTabController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(PostProcessingTabController, self).__init__(*args, **kwargs)

        self.run_filter_button.clicked.connect(self.bm4dFilter)
        self.image_data = None
        self.denoised_image = None

    def bm4dFilter(self):
        """
        Perform the BM4D filter operation using threading.

        Starts a new thread to execute the RunBm4dFilter method.
        """
        thread = threading.Thread(target=self.RunBm4dFilter)
        thread.start()

    def RunBm4dFilter(self):
        """
        Run the BM4D filter operation.

        Retrieves the image data, performs rescaling, computes median and median absolute deviation (MAD),
        applies the BM4D filter to the rescaled image, restores the denoised image to its original dimensions,
        updates the main matrix of the image view widget, adds the operation to the history widget,
        and updates the operations history.
        """
        image_data = np.abs(self.main.image_view_widget.main_matrix).astype(float)

        image_rescaled = np.interp(image_data, (np.min(image_data), np.max(image_data)), (0, 100))

        med = np.median(image_rescaled)
        mad = np.median(np.abs(image_rescaled - med))

        if self.auto_checkbox.isChecked():
            sigma_psd = (1.4826 * mad)/2
        else:
            std_value = float(self.std_text_field.text())
            sigma_psd = std_value

        print('BM4D is loading')

        profile = bm4d.BM4DProfile()
        stage_arg = bm4d.BM4DStages.ALL_STAGES
        blockmatches = (False, False)

        denoised_rescaled = bm4d.bm4d(image_rescaled, sigma_psd=sigma_psd, profile=profile, stage_arg=stage_arg,
                                      blockmatches=blockmatches)

        denoised_image = np.interp(denoised_rescaled, (np.min(denoised_rescaled), np.max(denoised_rescaled)),
                                   (np.min(image_data), np.max(image_data)))

        self.main.image_view_widget.main_matrix = denoised_image

        self.main.history_controller.addItemWithTimestamp("BM4D")

        self.main.history_controller.hist_dict[self.main.history_controller.matrix_infos] = \
            self.main.image_view_widget.main_matrix

        self.main.history_controller.updateOperationsHist(self.main.history_controller.matrix_infos, "BM4D - Standard "
                                                                                                     "deviation : " +
                                                          str(sigma_psd))
        print('BM4D filter has been applied')
