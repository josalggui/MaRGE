from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QGroupBox, QWidget


class ReconstructionTabWidget(QWidget):
    """
    ReconstructionTabWidget class for displaying a tab widget for image reconstruction options.

    Inherits from QTabWidget provided by PyQt5 to display a tab widget for image reconstruction options.

    Attributes:
        main: The parent widget.
    """

    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the ReconstructionTabWidget.

        Args:
            parent: The parent widget.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(ReconstructionTabWidget, self).__init__(*args, **kwargs)

        # The 'main' attribute represents the parent widget, which is used to access the main window or controller.
        self.main = parent

        # ART
        self.image_art_button = QPushButton('ART')

        # Labels
        self.niter_label = QLabel('Number of iterations')
        self.lambda_label = QLabel('Lambda')

        # Text Fields
        self.niter_text_field = QLineEdit()
        self.niter_text_field.setText('1')
        self.lambda_text_field = QLineEdit()
        self.lambda_text_field.setText('1')

        # Layouts
        self.order_layout = QHBoxLayout()
        self.order_layout.addWidget(self.niter_label)
        self.order_layout.addWidget(self.niter_text_field)
        self.order_layout.addWidget(self.lambda_label)
        self.order_layout.addWidget(self.lambda_text_field)

        self.art_layout = QVBoxLayout()
        self.art_layout.addLayout(self.order_layout)
        self.art_layout.addWidget(self.image_art_button)

        self.art_group = QGroupBox('ART')
        self.art_group.setLayout(self.art_layout)

        # FFT
        self.ifft_button = QPushButton('k-space -> i-space')
        self.dfft_button = QPushButton('i-space -> k-space')

        self.fft_layout = QVBoxLayout()
        self.fft_layout.addWidget(self.ifft_button)
        self.fft_layout.addWidget(self.dfft_button)

        self.fft_group = QGroupBox('FFT')
        self.fft_group.setLayout(self.fft_layout)

        # *****************
        # POCS
        # *****************
        # Partial fourier factor
        self.partial_reconstruction_label = QLabel('Factor')
        self.partial_reconstruction_factor = QLineEdit()
        self.partial_reconstruction_factor.setPlaceholderText("Readout, Phase, Slice")
        self.partial_reconstruction_factor.setStatusTip('To apply partial acquisition use numbers between'
                                                        '0.5 and 1, otherwise use 0')

        self.factor_layout = QHBoxLayout()
        self.factor_layout.addWidget(self.partial_reconstruction_label)
        self.factor_layout.addWidget(self.partial_reconstruction_factor)

        self.nb_points_label = QLabel('Number of points')
        self.nb_points_text_field = QLineEdit()
        self.nb_points_text_field.setText('2')

        self.nb_points_layout = QHBoxLayout()
        self.nb_points_layout.addWidget(self.nb_points_label)
        self.nb_points_layout.addWidget(self.nb_points_text_field)

        self.threshold_label = QLabel('Correlation threshold')
        self.threshold_text_field = QLineEdit()
        self.threshold_text_field.setText('1e-6')

        self.threshold_layout = QHBoxLayout()
        self.threshold_layout.addWidget(self.threshold_label)
        self.threshold_layout.addWidget(self.threshold_text_field)

        self.pocs_button = QPushButton('Run POCS')
        self.zero_button = QPushButton('Run iFFT with zero padding')

        self.pocs_layout = QVBoxLayout()
        self.pocs_layout.addLayout(self.factor_layout)
        self.pocs_layout.addLayout(self.nb_points_layout)
        self.pocs_layout.addLayout(self.threshold_layout)
        self.pocs_layout.addWidget(self.pocs_button)
        self.pocs_layout.addWidget(self.zero_button)

        self.pocs_group = QGroupBox("Partial Reconstruction")
        self.pocs_group.setLayout(self.pocs_layout)

        # Main layout
        self.reconstruction_layout = QVBoxLayout()
        self.reconstruction_layout.addWidget(self.art_group)
        self.reconstruction_layout.addWidget(self.fft_group)
        self.reconstruction_layout.addWidget(self.pocs_group)
        self.reconstruction_layout.addStretch()
        self.setLayout(self.reconstruction_layout)
