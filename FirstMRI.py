"""FirstMRI.py
Startup Code

#@author:    Yolanda Vives

"""

import sys
sys.path.append('../marcos_client')
from PyQt5.QtWidgets import QApplication
#from controller.mainviewcontroller import MainViewController,1
from controller.sessionviewer_controller import SessionViewerController
import cgitb 
cgitb.enable(format = 'text')


VERSION = "0.1.0"
AUTHOR = "Yolanda Vives"

print("Graphical User Interface for Magnetic Resonance Imaging")
    
   
app = QApplication(sys.argv)
#    gui = MainViewController()
gui = SessionViewerController('')
gui.show()
sys.exit(app.exec_())

    
    
