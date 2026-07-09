"""
Main file to run MaRGE
"""
import os
import sys
from PyQt5.QtWidgets import QApplication


def MaRGE():
    """
    Entry point for the MaRGE graphical user interface.

    Creates the required directory structure (experiments, calibration, protocols,
    reports, configs) if they do not exist, then launches the Qt application
    with the session controller as the main window.
    """
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
    if not os.path.exists('reports'):
        os.makedirs('reports')
    if not os.path.exists('configs'):
        os.makedirs('configs')

    from marge.controller.controller_session import SessionController

    app = QApplication(sys.argv)
    gui = SessionController()
    gui.show()
    sys.exit(app.exec_())


def main():
    """Console-script entry point."""
    MaRGE()


if __name__ == "__main__":
    main()
