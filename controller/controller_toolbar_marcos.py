"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import time

from widgets.widget_toolbar_marcos import MarcosToolBar
import subprocess
import platform
import experiment as ex
import numpy as np
import shutil
import configs.hw_config as hw
import autotuning.autotuning as autotuning # Just to use an arduino
import threading


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

        thread = threading.Thread(target=self.search_sdrlab)
        thread.start()

        # Arduino to control the interlock
        self.arduino = autotuning.Arduino(baudrate=19200, name="interlock", serial_number=hw.ard_sn_interlock)
        self.arduino.connect()

    def search_sdrlab(self):
        # Get the IP of the SDRLab
        if not self.demo:
            try:
                hw.rp_ip_address = self.get_sdrlab_ip()[0]
            except:
                print("No SDRLab found.")
                try:
                    hw.rp_ip_address = self.get_sdrlab_ip()[0]
                except:
                    print("No communication with SDRLab.")
                    print("Try manually.")



    @staticmethod
    def get_sdrlab_ip():
        print("\nSearching for SDRLabs...")

        ip_addresses = []
        subnet = '192.168.1.'
        timeout = 0.1  # Adjust timeout value as needed

        for i in range(101, 132):  # Scan IP range 192.168.1.101 to 192.168.1.132
            ip = subnet + str(i)
            try:
                if platform.system() == 'Linux':
                    result = subprocess.run(['ping', '-c', '1', ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=timeout)
                elif platform.system() == 'Windows':
                    result = subprocess.run(['ping', '-n', '1', ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=timeout)
                if result.returncode == 0:
                    print(f"Checking ip {ip}...")
                    # Attempt SSH connection without authentication
                    ssh_command = ['ssh', '-o', 'BatchMode=yes', '-o', f'ConnectTimeout={5}',
                                   f'{"root"}@{ip}', 'exit']
                    ssh_result = subprocess.run(ssh_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    if ssh_result.returncode == 0:  # SSH was successful
                        ip_addresses.append(ip)
                    else:
                        print(f"No SDRLab found at ip {ip}.")
            except:
                pass

        for ip in ip_addresses:
            print("SDRLab found at IP " + ip)

        return ip_addresses

    def startMaRCoS(self):
        """
        Starts the MaRCoS system.

        Executes startRP.sh: copy_bitstream.sh & marcos_server.
        """
        if not self.demo:

            try:
                subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
                subprocess.run([hw.bash_path, "--", "./startRP.sh", hw.rp_ip_address, hw.rp_version])
                self.initgpa()
                print("\nMaRCoS updated, server connected, gpa initialized.")
            except:
                print("\nERROR: Server connection not found! Please verify if the blue LED is illuminated on the Red Pitaya.")
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
                subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
                self.action_server.setStatusTip('Connect to marcos server')
                self.action_server.setToolTip('Connect to marcos server')
                print("\nServer disconnected")
            else:
                subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
                time.sleep(1.5)
                subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "~/marcos_server"])
                time.sleep(1.5)
                self.action_server.setStatusTip('Kill marcos server')
                self.action_server.setToolTip('Kill marcos server')

                # Run tiny sequence to check connection to the server
                try:
                    expt = ex.Experiment(init_gpa=False)
                    expt.add_flodict({
                        'grad_vx': (np.array([100]), np.array([0])),
                    })
                    expt.run()
                    expt.__del__()
                    print("\nServer connected!")
                except Exception as e:
                    print("\nServer not connected!")
                    print("Try to connect to the server again.")
                    print("Error details:", e)
        else:
            print("\nThis is a demo")

    def copyBitStream(self):
        """
        Copies the MaRCoS bitstream to the remote platform.

        Executes copy_bitstream.sh.
        """
        if not self.demo:
            try:
                subprocess.run([hw.bash_path, "--", "./communicateRP.sh", hw.rp_ip_address, "killall marcos_server"])
                subprocess.run([hw.bash_path, '--', './copy_bitstream.sh', hw.rp_ip_address, 'rp-122'], timeout=10)
                print("\nMaRCoS updated")
            except subprocess.TimeoutExpired as e:
                print("\nTimeout error.")
                print("Error details:", e)
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
                while not link:
                    try:
                        # Check if GPA available
                        received_string = self.arduino.send("GPA_VERB 1;").decode()
                        if received_string[0:4] != ">OK;":
                            print("GPA not available.")
                        else:
                            print("GPA available.")

                        # Remote communication with GPA
                        received_string = self.arduino.send("GPA_SPC:CTL 1;").decode()  # Activate remote control
                        if received_string[0:4] != ">OK;":  # If wrong response
                            print("Error enabling GPA remote control.")
                        else:  # If good response
                            print("GPA remote communication succeed.")

                        # Check if RFPA available
                        received_string = self.arduino.send("RFPA_VERB 1;").decode()
                        if received_string[0:4] != ">OK;":
                            print("RFPA not available.")
                        else:
                            print("RFPA available.")

                        # Remote communication with RFPA
                        received_string = self.arduino.send("RFPA_SPC:CTL 1;").decode()
                        if received_string[0:4] != ">OK;":
                            print("Error enabling RFPA remote control.")
                        else:
                            print("RFPA remote communication succeed.")

                        # Disable power module
                        self.arduino.send("GPA_ON 0;")
                        self.arduino.send("RFPA_RF 0;")

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
                        received_string = self.arduino.send("GPA_ON 1;").decode()  # Enable power module
                        if received_string[0:4] != ">OK;":  # If wrong response
                            print("Error activating GPA power module.")
                        else:  # If good reponse
                            print("GPA power enabled.")

                        # Enable RFPA module
                        received_string = self.arduino.send("RFPA_RF 1;").decode()
                        if received_string[0:4] != ">OK;":
                            print("Error activating RFPA power module.")
                        else:
                            print("RFPA power enabled.")

                    except:
                        link = False
                        time.sleep(1)
        else:
            print("\nNo connection to the server")
            print("Please, connect to MaRCoS server first")
