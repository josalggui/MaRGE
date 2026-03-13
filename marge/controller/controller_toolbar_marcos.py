"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain
"""

import os
import time
import shutil
import platform
import subprocess
import threading
import numpy as np

from marge.widgets.widget_toolbar_marcos import MarcosToolBar
import marge.configs.hw_config as hw
if hw.marcos_version=="MaRCoS":
    import marge.marcos.marcos_client.experiment as ex
elif hw.marcos_version=="MIMO":
    import marge.controller.controller_device as device
import marge.marge_tyger.tyger_config as tyger
import multiprocessing as mp
from marge.mimo.marcos_client.mimo_devices import mimo_dev_run


class MarcosController(MarcosToolBar):
    """
    Controller class for managing MaRCoS functionality.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the MarcosController.
        """
        super(MarcosController, self).__init__(*args, **kwargs)

        # Copy relevant files from marcos_extras
        if hw.marcos_version=="MaRCoS":
            extras_path = os.path.join(os.path.dirname(__file__), "..", "marcos", "marcos_extras")
        elif hw.marcos_version=="MIMO":
            extras_path = os.path.join(os.path.dirname(__file__), "..", "mimo", "marcos_extras")
        dst = os.getcwd()
        os.makedirs(dst, exist_ok=True)
        files_to_copy = ["copy_bitstream.sh", "marcos_fpga_rp-122.bit", "marcos_fpga_rp-122.bit.bin",
            "marcos_fpga_rp-122.dtbo", "readme.org"]
        for fname in files_to_copy:
            src_file = os.path.join(extras_path, fname)
            if os.path.exists(src_file):
                shutil.copy(src_file, dst)
            else:
                print(f"[WARNING] File not found and not copied: {src_file}")

        # Communicate with RP
        comm_path = os.path.dirname(__file__)
        src_file = os.path.join(comm_path, "../communicateRP.sh")
        try:
            shutil.copy(src_file, dst)
        except:
            pass

        # MaRCoS installer
        src_file = os.path.join(comm_path, "../marcos_install.sh")
        try:
            shutil.copy(src_file, dst)
        except:
            pass

        self.action_server.setCheckable(True)
        self.action_marcos_install.triggered.connect(self.marcos_install)
        self.action_server.triggered.connect(self.controlMarcosServer)
        self.action_copybitstream.triggered.connect(self.copyBitStream)
        self.action_gpa_init.triggered.connect(self.initgpa)
        self.action_tyger_init.triggered.connect(self.init_tyger)

        # Unable action buttons
        if not self.main.demo:
            self.action_server.setEnabled(False)
            self.action_copybitstream.setEnabled(False)
            self.action_gpa_init.setEnabled(False)

        thread = threading.Thread(target=self.search_sdrlab)
        thread.start()

    @staticmethod
    def init_tyger():
        print(tyger.tyger_server)

        print("Initializing Tyger with server...")
        result = subprocess.run(
            ["tyger", "login", tyger.tyger_server],
            capture_output=True,
            text=True
        )

        print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr)

    def search_sdrlab(self):
        # Get the IP of the SDRLab
        if not self.main.demo:
            try:
                self.get_sdrlab_ip()
            except:
                print("ERROR: No SDRLab found.")
                try:
                    self.get_sdrlab_ip()
                except:
                    print("ERROR: No communication with SDRLab.")
                    print("ERROR: Try manually.")

    def get_sdrlab_ip(self):
        print("Searching for SDRLabs...")
        ip_addresses = []
        timeout = 0.1

        for ip in hw.rp_ip_list:
            try:
                print(f"Checking ip {ip}...")
                if platform.system() == 'Linux':
                    result = subprocess.run(['ping', '-c', '1', ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
                elif platform.system() == 'Windows':
                    result = subprocess.run(['ping', '-n', '1', ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
                else:
                    continue

                if result.returncode == 0:
                    print(f"ping to ip {ip} succeeded.")
                    ssh_command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', f'root@{ip}', 'exit']
                    ssh_result = subprocess.run(ssh_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if ssh_result.returncode == 0:
                        print(f"ssh to ip {ip} succeeded.")
                        ip_addresses.append(ip)
                    else:
                        print(f"ERROR: ssh to {ip} failed.")
            except:
                print(f"ERROR: ping to ip {ip} failed.")
                continue

        # Check SDRLabs found
        n = 0
        for ip in hw.rp_ip_list:
            if ip in ip_addresses:
                n+=1
                print(f"SDRLab at ip {ip} ready!")
            else:
                print(f"SDRLab at ip {ip} not ready!")
        if n==len(hw.rp_ip_list):
            print("READY: All SDRLab ready!")
        else:
            print("ERROR: At least one SDRLab is not ready")

        self.action_copybitstream.setEnabled(True)
        self.action_gpa_init.setEnabled(True)
        self.action_server.setEnabled(True)

        return ip_addresses

    def marcos_install(self):
        try:
            subprocess.run([
                "gnome-terminal", "--",
                "bash", "-c", f"sudo ./marcos_install.sh; exec bash"
            ])
        except:
            print("ERROR: Something went wrong.")

    def controlMarcosServer(self):
        """
        Controls the MaRCoS server connection.

        Connects or disconnects from the MaRCoS server.
        """

        def connect_server(ip):
            subprocess.run([hw.bash_path, "--", "./communicateRP.sh", ip, "killall marcos_server"])
            result = subprocess.run([hw.bash_path, "--", "./communicateRP.sh", ip, "~/marcos_server"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                print(f"READY: {ip} server connected!")
            else:
                print(f"ERROR: {ip} server not connected!")

        def disconnect_server(ip):
            subprocess.run([hw.bash_path, "--", "./communicateRP.sh", ip, "killall marcos_server"])
            self.action_server.setStatusTip('Connect to marcos server')
            self.action_server.setToolTip('Connect to marcos server')
            print("Server disconnected")

        if not self.main.demo:
            if not self.action_server.isChecked():
                for ip in hw.rp_ip_list:
                    thread = threading.Thread(target=disconnect_server, args=(ip,))
                    thread.start()
                    time.sleep(0.1)
                    if hw.marcos_version=="MaRCoS":
                        break
                self.action_server.setStatusTip('Connect to marcos server')
                self.action_server.setToolTip('Connect to marcos server')
            else:
                try:
                    for ip in hw.rp_ip_list:
                        thread = threading.Thread(target=connect_server, args=(ip, ))
                        thread.start()
                        time.sleep(0.1)
                        if hw.marcos_version=="MaRCoS":
                            break
                    self.action_server.setStatusTip('Kill marcos server')
                    self.action_server.setToolTip('Kill marcos server')

                    # TODO: Test connection to server
                    # expt = ex.Experiment(init_gpa=False, assert_errors=True)
                    # expt.add_flodict({'grad_vx': (np.array([100]), np.array([0]))})
                    # expt.run()
                    # expt.__del__()

                except Exception as e:
                    print("ERROR: Server not connected!")
                    print("ERROR: Try to connect to the server again.")
                    print(e)
        else:
            print("This is a demo\n")

    def copyBitStream(self):
        """
        Copies the MaRCoS bitstream to the Red Pitaya.

        Executes copy_bitstream.sh.
        """
        def do_copy_bitstream(ip):
            subprocess.run([hw.bash_path, "--", "./communicateRP.sh", ip, "killall marcos_server"])
            start_time = time.time()
            subprocess.run([hw.bash_path, '--', './copy_bitstream.sh', ip, hw.rp_version], timeout=10)
            if time.time() - start_time<10:
                print(f"READY: communication with FPGA from {ip} established")
            else:
                print(f"WARNING: {ip} timeout.")

        if not self.main.demo:
            try:
                for ip in hw.rp_ip_list:
                    thread = threading.Thread(target=do_copy_bitstream, args=(ip, ))
                    thread.start()
                    time.sleep(0.1)
                    if hw.marcos_version == "MaRCoS":
                        break
            except subprocess.TimeoutExpired as e:
                print("ERROR: MaRCoS init timeout")
                print(e)
        else:
            print("This is a demo\n")

        self.action_server.setChecked(False)
        self.main.toolbar_sequences.serverConnected()

    def initgpa(self):
        """
        Initializes the GPA (Gradient Power Amplifier) hardware.
        """

        def init_gpa():
            if self.action_server.isChecked():
                if not self.main.demo:
                    self.action_gpa_init.setEnabled(False)
                    link = False
                    while not link:
                        try:
                            gpa_code = []  # It appends 0 or 1 depending on success on GPA process
                            rfpa_code = []  # It appends 0 or 1 depending on success on RFPA process
                            # Initialize communication with gpa
                            if hw.gpa_model=="Barthel":
                                # Check if GPA available
                                received_string = self.main.arduino_interlock.send("GPA_VERB 1;").decode()
                                if received_string[0:4] != ">OK;":
                                    print("WARNING: GPA not available.")
                                    gpa_code.append(0)
                                else:
                                    print("GPA available.")
                                    gpa_code.append(1)

                                # Remote communication with GPA
                                received_string = self.main.arduino_interlock.send("GPA_SPC:CTL 1;").decode()  # Activate remote control
                                if received_string[0:4] != ">OK;":  # If wrong response
                                    print("WARNING: Error enabling GPA remote control.")
                                    gpa_code.append(0)
                                else:  # If good response
                                    print("GPA remote communication succeeded.")
                                    gpa_code.append(1)

                                # Disable Interlock
                                received_string = self.main.arduino_interlock.send("GPA_ERRST;").decode()  # Activate remote control
                                if received_string[0:4] != ">OK;":  # If wrong response
                                    print("WARNING: Interlock reset.")
                                    gpa_code.append(0)
                                else:  # If good response
                                    print("Interlock reset done.")
                                    gpa_code.append(1)

                                # Disable power module
                                received_string = self.main.arduino_interlock.send("GPA_ON 0;").decode()
                                if received_string[0:4] != ">OK;":  # If wrong response
                                    gpa_code.append(0)
                                else:  # If good response
                                    gpa_code.append(1)

                            # Initialize communication with rfpa
                            if hw.rfpa_model == "Barthel":
                                # Check if RFPA available
                                received_string = self.main.arduino_interlock.send("RFPA_VERB 1;").decode()
                                if received_string[0:4] != ">OK;":
                                    print("WARNING: RFPA not available.")
                                    rfpa_code.append(0)
                                else:
                                    print("RFPA available.")
                                    rfpa_code.append(1)

                                # Remote communication with RFPA
                                received_string = self.main.arduino_interlock.send("RFPA_SPC:CTL 1;").decode()
                                if received_string[0:4] != ">OK;":
                                    print("WARNING: Error enabling RFPA remote control.")
                                    rfpa_code.append(0)
                                else:
                                    print("RFPA remote communication succeeded.")
                                    rfpa_code.append(1)

                                # Disable power module
                                self.main.arduino_interlock.send("RFPA_RF 0;")
                                if received_string[0:4] != ">OK;":
                                    rfpa_code.append(0)
                                else:
                                    rfpa_code.append(1)

                            # Run init_gpa sequence
                            if hw.grad_board == "ocra1":
                                if hw.marcos_version == "MaRCoS":
                                    expt = ex.Experiment(init_gpa=True)
                                    expt.add_flodict({
                                        'grad_vx': (np.array([100]), np.array([0])),
                                    })
                                    expt.run()
                                    expt.__del__()
                                elif hw.marcos_version == "MIMO":
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
                                        ip_address=hw.rp_ip_list[0], port=hw.rp_port[0], **(master_kwargs | dev_kwargs))

                                    # Run init_gpa sequence
                                    master_device.add_flodict({
                                        'grad_vx': (np.array([100]), np.array([0])),
                                    })
                                    mpl = [(master_device, 0)]
                                    with mp.Pool(1) as p:
                                        p.map(mimo_dev_run, mpl)
                                    master_device.__del__()  # manual destructor needed
                                link = True
                                print("READY: OCRA1 init done!")
                            elif hw.grad_board == "gpa-fhdo":
                                link = True
                                print("READY: FHDO init done!")

                            # Enable gpa power modules
                            if hw.gpa_model == "Barthel":
                                # Enable GPA module
                                received_string = self.main.arduino_interlock.send("GPA_ON 1;").decode()  # Enable power module
                                if received_string[0:4] != ">OK;":  # If wrong response
                                    print("WARNING: Error activating GPA power module.")
                                    gpa_code.append(0)
                                else:  # If good response
                                    print("GPA power enabled.")
                                    gpa_code.append(1)

                            # Enable rfpa power module
                            if hw.rfpa_model == "Barthel":
                                received_string = self.main.arduino_interlock.send("RFPA_RF 1;").decode()
                                if received_string[0:4] != ">OK;":
                                    print("WARNING: Error activating RFPA power module.")
                                    rfpa_code.append(1)
                                else:
                                    print("RFPA power enabled.")
                                    rfpa_code.append(1)

                            # Check
                            if sum(gpa_code) == 5:
                                print("READY: GPA init done!")
                            elif sum(gpa_code) != 5 and hw.gpa_model == "Barthel":
                                print(f"ERROR: GPA init failed. Error code: {gpa_code}")
                            if sum(rfpa_code) == 4:
                                print("READY: RFPA init done!")
                            elif sum(rfpa_code) != 4 and hw.rfpa_model == "Barthel":
                                print(f"ERROR: RPFA init failed. Error code: {rfpa_code}")

                        except Exception as e:
                            print(e)
                            link = False
                            time.sleep(1)
                    self.action_gpa_init.setEnabled(True)
            else:
                print("ERROR: No connection to the server")
                print("Please, connect to MaRCoS server first")

        thread = threading.Thread(target=init_gpa)
        thread.start()


