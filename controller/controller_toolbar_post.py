import copy

import scipy as sp
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QLabel, QSizePolicy
from scipy.interpolate import griddata
from widgets.widget_toolbar_post import ToolBarWidgetPost
from controller.controller_plot3d import Plot3DController as Spectrum3DPlot
from PyQt5 import QtCore


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

    def __init__(self, *args, **kwargs):
        """
        Initialize the ToolBarController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(ToolBarControllerPost, self).__init__(*args, **kwargs)

        # Connect the image_loading_button clicked signal to the rawDataLoading method
        self.k_space_raw = None
        self.mat_data = None
        self.nPoints = None
        self.k_space = None
        self.action_load.triggered.connect(self.rawDataLoading)

    def rawDataLoading(self, file_path=None):
        """
        Load raw data from a .mat file and update the image view widget.
        """
        # Prompt the user to select a .mat file
        if not file_path:
            file_path = self.loadFile()
        self.mat_data = sp.io.loadmat(file_path)
        self.nPoints = np.reshape(self.mat_data['nPoints'], -1)

        if self.mat_data['seqName'] == 'PETRA':
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
            self.k_space_raw = self.mat_data['sampled']
            self.k_space = np.reshape(self.k_space_raw[:, 3], self.nPoints[-1::-1])

            # Clear the console, history widget, history controller, and history dictionaries
            self.main.console.console.clear()
            self.main.visualisation_controller.clear2DImage()

        # Update the main matrix of the image view widget with the k-space data
        self.main.image_view_widget.main_matrix = self.k_space

        # Update the image view widget to display the new main matrix
        try:
            image = np.log10(np.abs(self.main.image_view_widget.main_matrix))
        except:
            image = np.abs(self.main.image_view_widget.main_matrix)

        # Create figure widget
        image2show, x_label, y_label, title = self.fixImage(image)
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

        # Add the "KSpace" operation to the history
        self.main.history_list.addItemWithTimestamp("KSpace")

        # Update the history dictionary with the new main matrix for the current matrix info
        self.main.history_list.hist_dict[self.main.history_list.matrix_infos] = \
            self.main.image_view_widget.main_matrix

        # Update the operations history
        self.main.history_list.updateOperationsHist(self.main.history_list.matrix_infos, "KSpace")

        # Update the space dictionary
        self.main.history_list.space[self.main.history_list.matrix_infos] = 'k'

    def loadFile(self):
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

    def fixImage(self, matrix3d):
        matrix = copy.copy(matrix3d)
        axes = self.mat_data['axesOrientation'][0]
        if axes[2] == 2:  # Sagittal
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