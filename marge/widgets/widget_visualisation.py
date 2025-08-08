from PyQt5.QtWidgets import QPushButton, QLabel, QLineEdit, QHBoxLayout, QVBoxLayout, QWidget


class VisualisationTabWidget(QWidget):
    """
    VisualisationTabWidget class for displaying a tab widget for visualization settings.

    Inherits from QTabWidget provided by PyQt5 to display a tab widget for visualization settings.

    Attributes:
        main: The parent widget.
    """

    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the VisualisationTabWidget.

        Args:
            parent: The parent widget.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(VisualisationTabWidget, self).__init__(*args, **kwargs)
        self.main = parent

        # Button
        self.visualisation_button = QPushButton('Show slices')

        # Labels
        self.range_label = QLabel('Slices')
        self.column_label = QLabel('Rows and columns')

        # Text Fields
        self.range_text_field = QLineEdit()
        self.range_text_field.setPlaceholderText('First, Last')

        self.column_text_field = QLineEdit()
        self.column_text_field.setPlaceholderText('Rows, Columns')

        # Layouts
        self.number_layout = QHBoxLayout()
        self.number_layout.addWidget(self.range_label)
        self.number_layout.addWidget(self.range_text_field)

        self.column_layout = QHBoxLayout()
        self.column_layout.addWidget(self.column_label)
        self.column_layout.addWidget(self.column_text_field)

        self.visualisation_layout = QVBoxLayout()
        self.visualisation_layout.addLayout(self.number_layout)
        self.visualisation_layout.addLayout(self.column_layout)
        self.visualisation_layout.addWidget(self.visualisation_button)
        self.visualisation_layout.addStretch()

        self.setLayout(self.visualisation_layout)
