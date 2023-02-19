"""
session_controller.py
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from widgets.widget_toolbar_marcos import MarcosToolBar
import os
import platform
import experiment as ex
import numpy as np


class MarcosController(MarcosToolBar):
    def __init__(self, *args, **kwargs):
        super(MarcosController, self).__init__(*args, **kwargs)

        self.action_server.setCheckable(True)
        self.action_server.triggered.connect(self.controlMarcosServer)
        self.action_copybitstream.triggered.connect(self.copyBitStream)
        self.action_gpa_init.triggered.connect(self.initgpa)

    def controlMarcosServer(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: connect to marcos_server
        """

        if not self.action_server.isChecked():
            self.action_server.setStatusTip('Connect to marcos server')
            self.action_server.setToolTip('Connect to marcos server')
            if not self.demo:
                os.system('ssh root@192.168.1.101 "killall marcos_server"')
            print("\nServer disconnected.")
        else:
            self.action_server.setStatusTip('Kill marcos server')
            self.action_server.setToolTip('Kill marcos server')
            if platform.system() == 'Windows' and not self.demo:
                os.system('ssh root@192.168.1.101 "killall marcos_server"')
                os.system('start ssh root@192.168.1.101 "~/marcos_server"')
            elif platform.system() == 'Linux' and not self.demo:
                os.system('ssh root@192.168.1.101 "killall marcos_server"')
                os.system('ssh root@192.168.1.101 "~/marcos_server" &')
            print("\nServer connected.")

    def copyBitStream(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: execute copy_bitstream.sh
        """
        if self.action_server.isChecked():
            if not self.demo:
                os.system('ssh root@192.168.1.101 "killall marcos_server"')
                if platform.system() == 'Windows':
                    os.system('..\marcos_extras\copy_bitstream.sh 192.168.1.101 rp-122')
                elif platform.system() == 'Linux':
                    os.system('../marcos_extras/copy_bitstream.sh 192.168.1.101 rp-122')
            print("\nMaRCoS updated")
        else:
            print("\nNo connection to the server")
            print("Please, connect to MaRCoS server first")

    def initgpa(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: initialize the gpa board
        """
        if self.action_server.isChecked():
            if not self.demo:
                expt = ex.Experiment(init_gpa=True)
                expt.add_flodict({
                    'grad_vx': (np.array([100]), np.array([0])),
                })
                expt.run()
                expt.__del__()
            print("\nGPA init done!")
        else:
            print("\nNo connection to the server")
            print("Please, connect to MaRCoS server first")
