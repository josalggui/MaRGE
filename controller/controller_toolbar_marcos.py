"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import time

from widgets.widget_toolbar_marcos import MarcosToolBar
import subprocess
import experiment as ex
import numpy as np
import shutil
import configs.hw_config as hw
import autotuning.autotuning as autotuning # Just to use an arduino


class MarcosController(MarcosToolBar):
    """
    Controller class for managing MaRCoS (Magnetic Resonance Compatible
    Optical Stimulation) functionality.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Inherits:
        MarcosToolBar: Base class for the MaRCoS toolbar.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the MarcosController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(MarcosController, self).__init__(*args, **kwargs)

        # Copy relevant files from marcos_extras
        shutil.copy("../marcos_extras/copy_bitstream.sh", "../MaRGE")
        shutil.copy("../marcos_extras/marcos_fpga_rp-122.bit", "../MaRGE")
        shutil.copy("../marcos_extras/marcos_fpga_rp-122.bit.bin", "../MaRGE")
        shutil.copy("../marcos_extras/marcos_fpga_rp-122.dtbo", "../MaRGE")
        shutil.copy("../marcos_extras/readme.org", "../MaRGE")

        self.action_server.setCheckable(True)
        self.action_start.triggered.connect(self.startMaRCoS)
        self.action_server.triggered.connect(self.controlMarcosServer)
        self.action_copybitstream.triggered.connect(self.copyBitStream)
        self.action_gpa_init.triggered.connect(self.initgpa)

        # Arduino to control the interlock
        self.arduino = autotuning.Arduino(baudrate=19200, name="interlock", serial_number=hw.ard_sn_interlock)
        self.arduino.connect()

        # Enable remote communication with power modules
        if self.arduino:
            # Remote communication with GPA
            self.arduino.send("GPA_SPC:CTL 1;")  # Activate remote control
            received_string = self.arduino.receive()  # Wait to arduino response
            if received_string != ">OK;":  # If wrong response
                print("Error enabling GPA remote control.")
            else:  # If good response
                print("GPA remote communication succeed.")

            # Remote communication with RFPA
            self.arduino.send("RFPA_SPC:CTL 1;")
            received_string = self.arduino.receive()
            if received_string != ">OK;":
                print("Error enabling RFPA remote control.")
            else:
                print("RFPA remote communication succeed.")

    def startMaRCoS(self):
        """
        Starts the MaRCoS system.

        Executes startRP.sh: copy_bitstream.sh & marcos_server.
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
        Controls the MaRCoS server connection.

        Connects or disconnects from the MaRCoS server.
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
        Copies the MaRCoS bitstream to the remote platform.

        Executes copy_bitstream.sh.
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
        Initializes the GPA board.
        """
        if self.action_server.isChecked():
            if not self.demo:
                link = False
                while link==False:
                    try:
                        # Disable power module
                        self.arduino.send("GPA_ON 0;")
                        self.arduino.receive()
                        self.arduino.send("RFPA_ON 0;")
                        self.arduino.receive()

                        # Run init_gpa sequence
                        expt = ex.Experiment(init_gpa=True)
                        expt.add_flodict({
                            'grad_vx': (np.array([100]), np.array([0])),
                        })
                        expt.run()
                        expt.__del__()
                        link = True
                        print("\nGPA init done!")

                        # Enable power modules
                        # Enable GPA module
                        self.arduino.send("GPA_ON 1;")  # Enable power module
                        received_string = self.arduino.receive()
                        if received_string != ">OK;":  # If wrong response
                            print("Error activating GPA power module.")
                        else:  # If good reponse
                            print("GPA power enabled.")

                        # Enable RFPA module
                        self.arduino.send("RPFA_ON 1;")
                        received_string = self.arduino.receive()
                        if received_string != ">OK;":
                            print("Error activating RFPA power module.")
                        else:
                            print("RFPA power enabled.")

                    except:
                        link = False
                        time.sleep(1)
        else:
            print("\nNo connection to the server")
            print("Please, connect to MaRCoS server first")
