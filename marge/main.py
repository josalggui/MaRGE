"""
Main file to run MaRGE
"""
import os
import sys
from PyQt5.QtWidgets import QApplication

def MaRGE():
    # Run the gui
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

    from marge.controller.controller_session import SessionController

    app = QApplication(sys.argv)
    gui = SessionController()
    gui.show()
    sys.exit(app.exec_())
    console_controller.close_log()

if __name__ == "__main__":
    MaRGE()
