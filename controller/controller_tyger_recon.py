import os
import sys
import time
import threading
import numpy as np
from PyQt5.QtWidgets import QApplication

from widgets.widget_tyger_recon import TygerTabWidget
from marge_utils import utils
try:
    import cupy as cp
except ImportError:
    pass


class TygerTabController(TygerTabWidget):

    def __init__(self, *args, **kwargs):
        super(TygerTabController, self).__init__(*args, **kwargs)

        self.rare_run_pk_recon_button.clicked.connect(self.run_rare_pk_recon)
        self.run_snraware_button.clicked.connect(self.run_snraware_recon)
        self.petra_run_pk_recon_button.clicked.connect(self.run_petra_pk_recon)


    def run_rare_pk_recon(self):

        thread = threading.Thread(target=self.rare_pk_recon)
        thread.start()

    def rare_pk_recon(self):
        print("----- PK Reconstruction Inputs -----")

        # Radio buttons
        if self.rare_method1_radio.isChecked():
            print("Selected method:", self.rare_method1_radio.text())
        elif self.rare_method2_radio.isChecked():
            print("Selected method:", self.rare_method2_radio.text())

        # Combo box
        print("K-space type:", self.rare_kspace_combo.currentText())

        # Path text field
        print("B0 file path:", self.rare_recon_text.text())

        print("----- End -----")

    def run_petra_pk_recon(self):

        thread = threading.Thread(target=self.petra_pk_recon)
        thread.start()

    def petra_pk_recon(self):
        pass

    def run_snraware_recon(self):

        thread = threading.Thread(target=self.snraware_recon)
        thread.start()

    def snraware_recon(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TygerTabController(parent=None)
    window.show()
    sys.exit(app.exec_())