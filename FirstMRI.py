"""FirstMRI.pFirstMRI.pyy
Startup Code

#@author:    Yolanda Vives

"""
import os
import sys
import time
import pyautogui
import platform
from PyQt5.QtWidgets import QApplication
import cgitb
#*****************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************
from controller.sessionviewer_controller import SessionViewerController

cgitb.enable(format = 'text')

VERSION = "0.2.0"
AUTHORA = "Yolanda Vives"
AUTHORB = "J.M. Algarín"
print("****************************************************************************************")
print("Graphical User Interface for MaRCoS                                                    *")
print("Dr. Y. Vives, Department of Electronic Engineering, Universitat de València, Spain     *")
print("Dr. J.M. Algarín, mriLab @ i3M, CSIC, Spain                                            *")
print("https://www.i3m-stim.i3m.upv.es/research/magnetic-resonance-imaging-laboratory-mrilab/ *")
print("https://github.com/yvives/PhysioMRI_GUI                                                *")
print("****************************************************************************************")

# Start the red pitaya -> ./copy_bitstream and ssh root@192.168.1.101 "marcos_server"
if platform.system()=='Windows'
    os.system('start startRP.sh')
    time.sleep(4)  # Wait while seting up red pitaya
    # Close shell
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'c')
elif platform.system()=='Linux':
    os.system('nohup ./startRP.sh')     # For linux
    time.sleep(4)  # Wait while seting up red pitaya
    # Close shell
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'c')

# Run the gui
app = QApplication(sys.argv)
gui = SessionViewerController('')
gui.show()
sys.exit(app.exec_())

    
    
