"""
session_controller.py
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import time

from widgets.widget_toolbar_marcos import MarcosToolBar
import subprocess
import experiment as ex
import numpy as np
import shutil
import configs.hw_config as hw


class MarcosController(MarcosToolBar):
    def __init__(self, *args, **kwargs):
        super(MarcosController, self).__init__(*args, **kwargs)

        # Copy relevant files from marcos_extras
        shutil.copy("../marcos_extras/copy_bitstream.sh", "../PhysioMRI_GUI")
        shutil.copy("../marcos_extras/marcos_fpga_rp-122.bit", "../PhysioMRI_GUI")
        shutil.copy("../marcos_extras/marcos_fpga_rp-122.bit.bin", "../PhysioMRI_GUI")
        shutil.copy("../marcos_extras/marcos_fpga_rp-122.dtbo", "../PhysioMRI_GUI")
        shutil.copy("../marcos_extras/readme.org", "../PhysioMRI_GUI")

        self.action_server.setCheckable(True)
        self.action_start.triggered.connect(self.startMaRCoS)
        self.action_server.triggered.connect(self.controlMarcosServer)
        self.action_copybitstream.triggered.connect(self.copyBitStream)
        self.action_gpa_init.triggered.connect(self.initgpa)

    def startMaRCoS(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: execute startRP.sh: copy_bitstream.sh & marcos_server
        """
        if not self.demo:
            subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
            subprocess.run([hw.bash_path, "--", "./startRP.sh", hw.rp_ip_address, hw.rp_version])
            self.initgpa()
            print("\nMaRCoS updated, server connected, gpa initialized.")
        else:
            print("\nThis is a demo")
        self.action_server.setChecked(True)
        self.main.toolbar_sequences.serverConnected()

    def controlMarcosServer(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: connect to marcos_server
        """
        if not self.demo:
            if not self.action_server.isChecked():
                self.action_server.setStatusTip('Connect to marcos server')
                self.action_server.setToolTip('Connect to marcos server')
                subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
                print("\nServer disconnected")
            else:
                self.action_server.setStatusTip('Kill marcos server')
                self.action_server.setToolTip('Kill marcos server')
                subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
                subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "~/marcos_server"])
                print("\nServer connected")
        else:
            print("\nThis is a demo")

    def copyBitStream(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: execute copy_bitstream.sh
        """
        if not self.demo:
            subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
            subprocess.run([hw.bash_path, '--', './copy_bitstream.sh', '192.168.1.101', 'rp-122'])
            print("\nMaRCoS updated")
        else:
            print("\nThis is a demo.")
        self.action_server.setChecked(False)
        self.main.toolbar_sequences.serverConnected()

    def initgpa(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: initialize the gpa board
        """
        if self.action_server.isChecked():
            if not self.demo:
                link = False
                while link==False:
                    try:
                        expt = ex.Experiment(init_gpa=True)
                        expt.add_flodict({
                            'grad_vx': (np.array([100]), np.array([0])),
                        })
                        expt.run()
                        expt.__del__()
                        link = True
                        print("\nGPA init done!")
                    except:
                        link = False
                        time.sleep(1)
        else:
            print("\nNo connection to the server")
            print("Please, connect to MaRCoS server first")
