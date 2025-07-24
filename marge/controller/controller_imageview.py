from marge.widgets.widget_imageview import ImageViewWidget


class ImageViewController(ImageViewWidget):
    """
    Controller class for the image view widget.

    Inherits from ImageViewWidget.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the ImageViewController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(ImageViewController, self).__init__(*args, **kwargs)

        self.main_matrix = None  # Variable to store the main image matrix
