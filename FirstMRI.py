"""FirstMRI.pFirstMRI.pyy
Startup Code

#@author:    Yolanda Vives

"""
#*****************************************************************************
# Add path to the working directory
import os
import sys
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************

from PyQt5.QtWidgets import QApplication
from controller.sessionviewer_controller import SessionViewerController
import cgitb 
cgitb.enable(format = 'text')


VERSION = "0.2.0"
AUTHORA = "Yolanda Vives"
AUTHORB = "J.M. Algarín"
print("****************************************************************************************")
print("Graphical User Interface for MaRCoS                                                    *")
print("Dr. Y. Vives, and Dr. J.M. Algarín                                                     *")
print("mriLab @ i3M, CSIC, Spain                                                              *")
print("https://www.i3m-stim.i3m.upv.es/research/magnetic-resonance-imaging-laboratory-mrilab/ *")
print("https://github.com/yvives/PhysioMRI_GUI                                                *")
print("****************************************************************************************")

# import seq.sequencesCalibration as seqs
# rabi = seqs.RabiFlops()
# rabi.sequenceRun()

app = QApplication(sys.argv)
gui = SessionViewerController('')
gui.show()
sys.exit(app.exec_())

    
    
