import numpy as np
from PyQt5.QtWidgets import QGridLayout
from marge.widgets.widget_visualisation import VisualisationTabWidget
from controller.controller_plot3d import Plot3DController as Spectrum3DPlot


class VisualisationTabController(VisualisationTabWidget):
    """
    Controller class for the VisualisationTabWidget.

    Inherits from VisualisationTabWidget to provide additional functionality for managing visualisation tab actions.

    Attributes:
        visualisation_button: QPushButton for showing 2D slices.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the VisualisationTabController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(VisualisationTabController, self).__init__(*args, **kwargs)

        self.layout = QGridLayout()
        # Connect the visualisation_button clicked signal to the imageVisualisation method
        self.visualisation_button.clicked.connect(self.imageVisualisation)
        self.image_view = None

    def imageVisualisation(self):
        """
        Perform image visualisation on the selected slices of the main matrix.

        Display the selected slices as images using a grid layout.
        """

        image = self.main.image_view_widget.main_matrix

        slices = self.range_text_field.text().split(',')
        n0 = int(slices[0])
        n_end = int(slices[1])
        # slices_number = n_end - n0 + 1
        selected_slices = image[n0:n_end + 1]

        rows_columns = self.column_text_field.text().split(',')
        rows_number = int(rows_columns[0])
        columns_number = int(rows_columns[1])

        slice_height, slice_width = image.shape[1], image.shape[2]

        image_matrix = np.zeros((slice_height * columns_number, slice_width * rows_number), dtype=np.complex128)

        i = 0
        while i < len(selected_slices):
            row = i // columns_number
            col = i % columns_number
            row_start = row * slice_height
            row_end = row_start + slice_height
            col_start = col * slice_width
            col_end = col_start + slice_width

            image_matrix[col_start:col_end, row_start:row_end] = image[i]

            i += 1

        # Clean the image_view_widget
        self.main.image_view_widget.clearFiguresLayout()

        # Create new widget
        image = Spectrum3DPlot(main=self.main,
                               data=np.abs(np.reshape(image_matrix, (1, image_matrix.shape[0], image_matrix.shape[1]))),
                               x_label='',
                               y_label='',
                               title='')
        self.main.image_view_widget.addWidget(image)

    def clear2DImage(self):
        """
        Clear the second image view.
        """
        if self.image_view is not None:
            self.image_view.close()
            self.image_view = None
