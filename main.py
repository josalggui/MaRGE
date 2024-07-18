"""
Main file to run MaRGE
"""
import os
import sys
from PyQt5.QtWidgets import QApplication
from configs import sys_config

# *****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory (one level up)
parent_directory = os.path.dirname(main_directory)

# Define the subdirectories you want to add to sys.path
subdirs = ['marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
# ******************************************************************************

import experiment as ex

print("****************************************************************************************")
print("Graphical User Interface for MaRCoS                                                    *")
print("Dr. J.M. Algarín, mriLab @ i3M, CSIC, Spain                                            *")
print("https://www.i3m-stim.i3m.upv.es/research/magnetic-resonance-imaging-laboratory-mrilab/ *")
print("https://github.com/mriLab-i3M/MaRGE                                                    *")
print("****************************************************************************************")

# Add folders
if not os.path.exists('experiments/parameterization'):
    os.makedirs('experiments/parameterization')
if not os.path.exists('calibration'):
    os.makedirs('calibration')
if not os.path.exists('protocols'):
    os.makedirs('protocols')
if not os.path.exists(sys_config.screenshot_folder):
    os.makedirs(sys_config.screenshot_folder)

from controller.controller_session import SessionController

# Run the gui
app = QApplication(sys.argv)
gui = SessionController()
sys.exit(app.exec_())

