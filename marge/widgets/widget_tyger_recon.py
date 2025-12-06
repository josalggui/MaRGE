import sys

from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QGroupBox, QWidget, QApplication, \
    QTextEdit, QRadioButton, QComboBox


class TygerTabWidget(QWidget):
    def __init__(self, parent, *args, **kwargs):
        super(TygerTabWidget, self).__init__(*args, **kwargs)
        self.main = parent

        #**************************************************************
        # RARE PK recon here
        #**************************************************************
        self.rare_recon_layout = QVBoxLayout()

        # Radio buttons
        self.rare_radio_layout = QHBoxLayout()
        self.rare_method2_radio = QRadioButton("CP")
        self.rare_method1_radio = QRadioButton("ART")
        self.rare_method2_radio.setChecked(True)  # Optional default selection
        self.rare_radio_layout.addWidget(self.rare_method2_radio)
        self.rare_radio_layout.addWidget(self.rare_method1_radio)

        # Combobox
        # self.rare_kspace_combo = QComboBox()
        # self.rare_kspace_combo.addItems(["Raw k-space", "Denoised k-space"])

        # lineedit and recon button
        self.rare_recon_text = QLineEdit()
        label_bo = QLabel("Path to B0 file:")
        self.rare_recon_text.setPlaceholderText("Path to B0 file")
        self.rare_run_pk_recon_button = QPushButton('Run PK reconstruction')

        # Fill layout
        self.rare_recon_layout.addLayout(self.rare_radio_layout)
        # self.rare_recon_layout.addWidget(self.rare_kspace_combo)
        self.rare_recon_layout.addWidget(label_bo)
        self.rare_recon_layout.addWidget(self.rare_recon_text)
        self.rare_recon_layout.addWidget(self.rare_run_pk_recon_button)

        # #**************************************************************
        # # PETRA PK recon here
        # #**************************************************************
        # self.petra_recon_layout = QVBoxLayout()

        # # lineedit and recon button
        # # self.petra_recon_text = QLineEdit()
        # # self.petra_recon_text.setPlaceholderText("Path to B0 file")
        # self.petra_run_pk_recon_button = QPushButton('Run ART')

        # # Fill layout
        # # self.petra_recon_layout.addWidget(self.petra_recon_text)
        # self.petra_recon_layout.addWidget(self.petra_run_pk_recon_button)

        #**************************************************************
        # SNRAware here
        #**************************************************************
        # Crear label + combo
        label = QLabel("If RareDoubleImage, select:")
        self.snraware_double = QComboBox()
        self.snraware_double.addItems(["odd", "even", "all"])
        self.run_snraware_button = QPushButton('Run SNRAware')
        self.snraware_layout = QVBoxLayout()
        self.snraware_layout.addWidget(label)
        self.snraware_layout.addWidget(self.snraware_double)
        self.snraware_layout.addWidget(self.run_snraware_button)

        self.rare_recon_group = QGroupBox('RARE - PK reconstruction')
        self.rare_recon_group.setLayout(self.rare_recon_layout)

        # self.petra_recon_group = QGroupBox('PETRA - reconstruction')
        # self.petra_recon_group.setLayout(self.petra_recon_layout)

        self.snraware_group = QGroupBox('RARE - Denoising')
        self.snraware_group.setLayout(self.snraware_layout)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.snraware_group)
        self.main_layout.addWidget(self.rare_recon_group)
        # self.main_layout.addWidget(self.petra_recon_group)
        self.main_layout.addStretch()
        self.setLayout(self.main_layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = TygerTabWidget(parent=None)
    gui.show()
    sys.exit(app.exec_())


