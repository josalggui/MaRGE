from PyQt5.QtWidgets import QPushButton, QTabWidget, QWidget, QVBoxLayout, QLabel, QCheckBox, QLineEdit, QHBoxLayout, \
    QGroupBox


class PostProcessingTabWidget(QWidget):
    """
    PostProcessingTabWidget class for displaying a tab widget for post-processing options.

    Inherits from QTabWidget provided by PyQt5 to display a tab widget for post-processing options.

    Attributes:
        main: The parent widget.
    """

    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the PostProcessingTabWidget.

        Args:
            parent: The parent widget.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(PostProcessingTabWidget, self).__init__(*args, **kwargs)

        # The 'main' attribute represents the parent widget, which is used to access the main window or controller.
        self.main = parent

        # Layout for BM4D tab
        self.std_label = QLabel('Standard deviation')
        self.run_filter_button = QPushButton('Run filter')
        self.auto_checkbox = QCheckBox('Auto')
        self.auto_checkbox.setChecked(True)
        self.std_text_field = QLineEdit()
        self.bm4d_layout = QVBoxLayout()
        self.std_layout = QHBoxLayout()
        self.bm4d_layout.addLayout(self.std_layout)
        self.std_layout.addWidget(self.std_label)
        self.std_layout.addWidget(self.std_text_field)
        self.bm4d_layout.addWidget(self.auto_checkbox)
        self.bm4d_layout.addWidget(self.run_filter_button)

        self.bm4d_group = QGroupBox('BM4D')
        self.bm4d_group.setLayout(self.bm4d_layout)

        # Layout for Gaussian tab
        self.gaussian_layout = QVBoxLayout()
        self.gaussian_std_layout = QHBoxLayout()
        self.gaussian_label = QLabel('Standard deviation')
        self.gaussian_text_field = QLineEdit()
        self.gaussian_std_layout.addWidget(self.gaussian_label)
        self.gaussian_std_layout.addWidget(self.gaussian_text_field)
        self.run_gaussian_button = QPushButton('Run filter')
        self.gaussian_layout.addLayout(self.gaussian_std_layout)
        self.gaussian_layout.addWidget(self.run_gaussian_button)

        self.gauss_group = QGroupBox('Gaussian filter')
        self.gauss_group.setLayout(self.gaussian_layout)

        self.post_layout = QVBoxLayout()
        self.post_layout.addWidget(self.bm4d_group)
        self.post_layout.addWidget(self.gauss_group)
        self.post_layout.addStretch()
        self.setLayout(self.post_layout)