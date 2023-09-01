from PyQt5.QtWidgets import QPushButton, QTabWidget, QWidget, QVBoxLayout, QLabel, QCheckBox, QLineEdit, QHBoxLayout


class PostProcessingTabWidget(QTabWidget):
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

        # Create separate tabs for BM4D and Gaussian options
        bm4d_tab = QWidget()
        gaussian_tab = QWidget()

        # Add the tabs to the QTabWidget
        self.addTab(bm4d_tab, 'BM4D')
        self.addTab(gaussian_tab, 'Gaussian')

        # Labels
        self.std_label = QLabel('Standard deviation')

        # Buttons
        self.run_filter_button = QPushButton('Run filter')

        # CheckBox
        self.auto_checkbox = QCheckBox('Auto')
        self.auto_checkbox.setChecked(True)

        # Text Fields
        self.std_text_field = QLineEdit()

        # Layout for BM4D tab
        self.bm4d_layout = QVBoxLayout(bm4d_tab)
        self.std_layout = QHBoxLayout()
        self.bm4d_layout.addLayout(self.std_layout)
        self.std_layout.addWidget(self.std_label)
        self.std_layout.addWidget(self.std_text_field)
        self.bm4d_layout.addWidget(self.auto_checkbox)
        self.bm4d_layout.addWidget(self.run_filter_button)

        # Layout for Gaussian tab
        self.gaussian_layout = QVBoxLayout(gaussian_tab)
        self.gaussian_std_layout = QHBoxLayout()
        self.gaussian_label = QLabel('Standard deviation')
        self.gaussian_text_field = QLineEdit()
        self.gaussian_std_layout.addWidget(self.gaussian_label)
        self.gaussian_std_layout.addWidget(self.gaussian_text_field)
        self.run_gaussian_button = QPushButton('Run filter')
        self.gaussian_layout.addLayout(self.gaussian_std_layout)
        self.gaussian_layout.addWidget(self.run_gaussian_button)
