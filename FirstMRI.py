"""FirstMRI.pFirstMRI.pyy
Startup Code

#@author:    Yolanda Vives

"""

import os
import sys
# sys.path.append('../marcos_client')
from PyQt5.QtWidgets import QApplication
#from controller.mainviewcontroller import MainViewController,1
from controller.sessionviewer_controller import SessionViewerController
import cgitb 
cgitb.enable(format = 'text')

VERSION = "0.1.0"
AUTHOR = "Yolanda Vives"

print("Graphical User Interface for Magnetic Resonance Imaging")

#******************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        # sys.path.append(path[0:ii])
        print("Path: ",path[0:ii+1])
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************
   
app = QApplication(sys.argv)
#    gui = MainViewController()
gui = SessionViewerController('')
gui.show()
sys.exit(app.exec_())

    
    
