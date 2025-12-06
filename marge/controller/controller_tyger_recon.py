import sys
import threading
import numpy as np
from PyQt5.QtWidgets import QApplication

from marge.widgets.widget_tyger_recon import TygerTabWidget
try:
    import cupy as cp
except ImportError:
    pass
from marge.marge_tyger import tyger_rare
from marge.marge_tyger import tyger_denoising
from marge.marge_tyger import tyger_denoising_double
import marge.marge_tyger.tyger_config as tyger_conf
from marge.marge_tyger import tyger_petra

class TygerTabController(TygerTabWidget):

    def __init__(self, *args, **kwargs):
        super(TygerTabController, self).__init__(*args, **kwargs)

        self.rare_run_pk_recon_button.clicked.connect(self.run_rare_pk_recon)
        self.run_snraware_button.clicked.connect(self.run_snraware_recon)
        # self.petra_run_pk_recon_button.clicked.connect(self.run_petra_pk_recon)


    def run_rare_pk_recon(self):

        thread = threading.Thread(target=self.rare_pk_recon)
        thread.start()

    def rare_pk_recon(self):
        if self.main.seq_name == 'RarePyPulseq' or self.main.seq_name == 'RareDoubleImage':
            rawData_path = self.main.file_path
            # print("----- PK Reconstruction Inputs -----")
            if self.rare_method1_radio.isChecked():
                # print("Selected method:", self.rare_method1_radio.text())
                recon_type = 'artpk'
                output_field = 'post_imgTygerARTPK'
            elif self.rare_method2_radio.isChecked():
                # print("Selected method:", self.rare_method2_radio.text())
                recon_type = 'cp'
                output_field = 'post_imgTygerCP'
            # print("K-space type:", self.rare_kspace_combo.currentText())
            # print("B0 file path:", self.rare_recon_text.text())
            boFit_path = self.rare_recon_text.text()
            sign_rarepp = [-1,-1,-1,1,1,1,1,1, tyger_conf.cp_batchsize_RARE]
            try: 
                input_field = self.main.tyger_denoising
                output_field = output_field + '_den'
            except:
                input_field = ''
            try:
                imgTyger = tyger_rare.reconTygerRARE(rawData_path, recon_type, boFit_path, sign_rarepp, output_field, input_field)
                imageTyger = np.abs(imgTyger[0])
                imageTyger = imageTyger/np.max(np.reshape(imageTyger,-1))*100
                self.main.image_view_widget.main_matrix = imageTyger
                figure = imageTyger
                orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
                self.main.history_list.addNewItem(stamp="PKrecon",
                                        image=figure,
                                        orientation=orientation,
                                        operation="PKrecon",
                                        space="i",
                                        image_key=self.main.image_view_widget.image_key)
            except Exception as e:
                    print('Tyger reconstruction failed.')
                    print(f'Error: {e}')
        else:
            print('Reconstruction only available for RarePyPulseq and  RareDoubleImage sequences.')
        

    def run_petra_pk_recon(self):

        thread = threading.Thread(target=self.petra_pk_recon)
        thread.start()

    def petra_pk_recon(self):
        if self.main.seq_name == 'PETRA':
            try:
                rawData_path = self.main.file_path
                output_field = 'post_imgTygerART'
                imgTyger = tyger_petra.reconTygerPETRA(rawData_path, output_field)
                imageTyger = np.abs(imgTyger[0])
                imageTyger = imageTyger/np.max(np.reshape(imageTyger,-1))*100
                self.main.image_view_widget.main_matrix = imageTyger
                figure = imageTyger
                orientation=None
                self.main.history_list.addNewItem(stamp="ART",
                                        image=figure,
                                        orientation=orientation,
                                        operation="ART",
                                        space="i",
                                        image_key=self.main.image_view_widget.image_key)
            except Exception as e:
                    print('Tyger reconstruction failed.')
                    print(f'Error: {e}')
        else:
            print('Reconstruction only available for PETRA sequence.')

    def run_snraware_recon(self):

        thread = threading.Thread(target=self.snraware_recon)
        thread.start()

    def snraware_recon(self):
        if self.main.seq_name == 'RarePyPulseq' or self.main.seq_name == 'RareDoubleImage':
            rawData_path = self.main.file_path
            out_field = 'post_image3D_den'
            out_field_k = 'post_kSpace3D_den'
            if self.main.seq_name == 'RareDoubleImage':
                try:
                    input_echoes = self.snraware_double.currentText()
                    imgTyger = tyger_denoising_double.denoisingTyger_double(rawData_path, out_field, out_field_k, input_echoes)
                    imageTyger = np.abs(imgTyger[0])
                    imageTyger = imageTyger/np.max(np.reshape(imageTyger,-1))*100
                    if input_echoes == 'even': 
                        self.main.tyger_denoising = out_field_k + '_even'
                    elif input_echoes == 'odd': 
                        self.main.tyger_denoising = out_field_k + '_odd'
                    elif input_echoes == 'all': 
                        self.main.tyger_denoising = out_field_k + '_all'
                    self.main.image_view_widget.main_matrix = imageTyger
                    figure = imageTyger
                    orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
                    self.main.history_list.addNewItem(stamp="SNRAware",
                                            image=figure,
                                            orientation=orientation,
                                            operation="SNRAware",
                                            space="i",
                                            image_key=self.main.image_view_widget.image_key)
                except Exception as e:
                    print('Tyger reconstruction failed.')
                    print(f'Error: {e}')
            elif self.main.seq_name == 'RarePyPulseq':
                try:
                    imgTyger = tyger_denoising.denoisingTyger(rawData_path, out_field, out_field_k)
                    imageTyger = np.abs(imgTyger[0])
                    imageTyger = imageTyger/np.max(np.reshape(imageTyger,-1))*100
                    self.main.tyger_denoising = out_field_k
                    self.main.image_view_widget.main_matrix = imageTyger
                    figure = imageTyger
                    orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
                    self.main.history_list.addNewItem(stamp="SNRAware",
                                            image=figure,
                                            orientation=orientation,
                                            operation="SNRAware",
                                            space="i",
                                            image_key=self.main.image_view_widget.image_key)
                except Exception as e:
                    print('Tyger reconstruction failed.')
                    print(f'Error: {e}')
        else:
            print('Reconstruction only available for RarePyPulseq and  RareDoubleImage sequences.')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TygerTabController(parent=None)
    window.show()
    sys.exit(app.exec_())