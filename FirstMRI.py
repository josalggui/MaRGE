"""
Startup Code

@author:    Yolanda Vives

"""

import sys
sys.path.append('../marcos_client')
from PyQt5.QtWidgets import QApplication
from controller.mainviewcontroller import MainViewController

VERSION = "0.1.0"
AUTHOR = "Yolanda Vives"

if __name__ == '__main__':
    print("Graphical User Interface for Magnetic Resonance Imaging {} by {}".format(VERSION, AUTHOR))
    
    app = QApplication(sys.argv)
    gui = MainViewController()
    gui.show()
    sys.exit(app.exec_())
    
    
