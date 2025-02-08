"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import time

from configs.hw_config import rp_ip_list
from widgets.widget_toolbar_marcos import MarcosToolBar
import subprocess
import controller.controller_device as device
from mimo_devices import mimo_dev_run
import numpy as np
import shutil
import configs.hw_config as hw
import autotuning.autotuning as autotuning # Just to use an arduino
import threading
import multiprocessing as mp


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
        shutil.copy(f"../marcos_extras/copy_bitstream.sh", "../MaRGE")
        shutil.copy(f"../marcos_extras/marcos_fpga_{hw.rp_version}.bit", "../MaRGE")
        shutil.copy(f"../marcos_extras/marcos_fpga_{hw.rp_version}.bit.bin", "../MaRGE")
        shutil.copy(f"../marcos_extras/marcos_fpga_{hw.rp_version}.dtbo", "../MaRGE")
        shutil.copy("../marcos_extras/readme.org", "../MaRGE")

        self.action_server.setCheckable(True)
        self.action_start.triggered.connect(self.startMaRCoS)
        self.action_server.triggered.connect(self.controlMarcosServer)
        self.action_copybitstream.triggered.connect(self.copyBitStream)
        self.action_gpa_init.triggered.connect(self.initgpa)

        # Arduino to control the interlock
        self.arduino = autotuning.Arduino(baudrate=19200, name="interlock", serial_number=hw.ard_sn_interlock)
        self.arduino.connect()

    def startMaRCoS(self):
        """
        Starts the MaRCoS system.

        Executes startRP.sh: copy_bitstream.sh & marcos_server.
        """
        if not self.main.demo:
            try:
                subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
                subprocess.run([hw.bash_path, "--", "./startRP.sh", hw.rp_ip_address, hw.rp_version])
                self.initgpa()
                print("READY: MaRCoS updated, server connected, gpa initialized.")
            except:
                print("ERROR: Server connection not found! Please verify if the blue LED is illuminated on the Red Pitaya.")
        else:
            print("This is a demo\n")
        self.action_server.setChecked(True)
        self.main.toolbar_sequences.serverConnected()

    def controlMarcosServer(self):
        """
        Controls the MaRCoS server connection.

        Connects or disconnects from the MaRCoS server.
        """
        def connect_server():
            ip = self.ip
            subprocess.run([hw.bash_path, "--", "./communicateRP.sh", ip, "killall marcos_server"])
            result = subprocess.run([hw.bash_path, "--", "./communicateRP.sh", ip, "~/marcos_server"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                print(f"READY: {ip} server connected!")
            else:
                print(f"ERROR: {ip} server not connected!")

        def disconnect_server():
            ip = self.ip
            subprocess.run([hw.bash_path, "--", "./communicateRP.sh", ip, "killall marcos_server"])
            self.action_server.setStatusTip('Connect to marcos server')
            self.action_server.setToolTip('Connect to marcos server')
            print("Server disconnected")

        if not self.main.demo:
            if not self.action_server.isChecked():
                for self.ip in hw.rp_ip_list:
                    thread = threading.Thread(target=disconnect_server)
                    thread.start()
                    time.sleep(0.1)
            else:
                try:
                    for self.ip in hw.rp_ip_list:
                        thread = threading.Thread(target=connect_server)
                        thread.start()
                        time.sleep(0.1)
                    self.action_server.setStatusTip('Kill marcos server')
                    self.action_server.setToolTip('Kill marcos server')

                    # TODO: Test connection to server
                    # device = dev.Device(init_gpa=False)
                    # device.add_flodict({
                    #     'grad_vx': (np.array([100]), np.array([0])),
                    # })
                    # device.run()
                    # device.__del__()

                except Exception as e:
                    print("ERROR: Server not connected!")
                    print("ERROR: Try to connect to the server again.")
                    print(e)
        else:
            print("This is a demo\n")

    def copyBitStream(self):
        """
        Copies the MaRCoS bitstream to the remote platform.

        Executes copy_bitstream.sh.
        """
        def do_copy_bitstream():
            ip = self.ip
            subprocess.run([hw.bash_path, "--", "./communicateRP.sh", ip, "killall marcos_server"])
            start_time = time.time()
            subprocess.run([hw.bash_path, '--', './copy_bitstream.sh', ip, hw.rp_version], timeout=10)
            if time.time() - start_time<10:
                print(f"READY: {ip} initialized!")
            else:
                print(f"WARNING: {ip} timeout.")

        if not self.main.demo:
            try:
                for self.ip in hw.rp_ip_list:
                    thread = threading.Thread(target=do_copy_bitstream)
                    thread.start()
                    time.sleep(0.1)
            except subprocess.TimeoutExpired as e:
                print("ERROR: MaRCoS init timeout")
                print(e)
        else:
            print("This is a demo\n.")
        self.action_server.setChecked(False)
        self.main.toolbar_sequences.serverConnected()

    def initgpa(self):
        """
        Initializes the GPA board.
        """

        def init_gpa():
            if self.action_server.isChecked():
                if not self.main.demo:
                    link = False
                    while not link:
                        try:
                            # Check if GPA available
                            received_string = self.arduino.send("GPA_VERB 1;").decode()
                            if received_string[0:4] != ">OK;":
                                print("WARNING: GPA not available.")
                            else:
                                print("READY: GPA available.")

                            # Remote communication with GPA
                            received_string = self.arduino.send("GPA_SPC:CTL 1;").decode()  # Activate remote control
                            if received_string[0:4] != ">OK;":  # If wrong response
                                print("WARNING: Error enabling GPA remote control.")
                            else:  # If good response
                                print("READY: GPA remote communication succeed.")

                            # Disable Interlock
                            received_string = self.arduino.send("GPA_ERRST;").decode()  # Activate remote control
                            if received_string[0:4] != ">OK;":  # If wrong response
                                print("WARNING: Interlock reset.")
                            else:  # If good response
                                print("READY: Interlock reset done.")

                            # Check if RFPA available
                            received_string = self.arduino.send("RFPA_VERB 1;").decode()
                            if received_string[0:4] != ">OK;":
                                print("WARNING: RFPA not available.")
                            else:
                                print("READY: RFPA available.")

                            # Remote communication with RFPA
                            received_string = self.arduino.send("RFPA_SPC:CTL 1;").decode()
                            if received_string[0:4] != ">OK;":
                                print("WARNING: Error enabling RFPA remote control.")
                            else:
                                print("READY: RFPA remote communication succeed.")

                            # Disable power module
                            self.arduino.send("GPA_ON 0;")
                            self.arduino.send("RFPA_RF 0;")

                            # Define device arguments
                            dev_kwargs = {
                                "init_gpa": True,
                            }

                            # Define master arguments
                            master_kwargs = {
                                'mimo_master': True,
                                'trig_output_time': 1e5,
                                'slave_trig_latency': 6.079
                            }

                            # Create list of devices
                            master_device = device.Device(
                                ip_address=rp_ip_list[0], port=hw.rp_port, **(master_kwargs | dev_kwargs))

                            # Run init_gpa sequence
                            master_device.add_flodict({
                                'grad_vx': (np.array([100]), np.array([0])),
                            })
                            mpl = [(master_device, 0)]
                            with mp.Pool(1) as p:
                                p.map(mimo_dev_run, mpl)
                            master_device.__del__()  # manual destructor needed
                            link = True
                            print("READY: GPA init done!")

                            # Enable power modules
                            # Enable GPA module
                            received_string = self.arduino.send("GPA_ON 1;").decode()  # Enable power module
                            if received_string[0:4] != ">OK;":  # If wrong response
                                print("WARNING: Error activating GPA power module.")
                            else:  # If good reponse
                                print("READY: GPA power enabled.")

                            # Enable RFPA module
                            received_string = self.arduino.send("RFPA_RF 1;").decode()
                            if received_string[0:4] != ">OK;":
                                print("WARNING: Error activating RFPA power module.")
                            else:
                                print("READY: RFPA power enabled.")

                        except:
                            link = False
                            time.sleep(1)
            else:
                print("ERROR: No connection to the server")
                print("Please, connect to MaRCoS server first")

        thread = threading.Thread(target=init_gpa)
        thread.start()


