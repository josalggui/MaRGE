"""
Main file to run MaRGE
"""
import os
import signal
import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication


def _install_signal_handlers(app, gui):
    """
    Handle SIGINT (Ctrl+C) and SIGTERM by scheduling a clean shutdown
    in the Qt event loop (signal handlers must not call Qt from the handler).
    """
    def request_close():
        gui.close()  # Runs cleanup (server, power modules, console) and sys.exit()

    def handle_term(signum, frame):
        QTimer.singleShot(0, request_close)

    signal.signal(signal.SIGINT, handle_term)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, handle_term)


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
    if not os.path.exists('reports'):
        os.makedirs('reports')
    if not os.path.exists('configs'):
        os.makedirs('configs')

    from marge.controller.controller_session import SessionController

    app = QApplication(sys.argv)
    gui = SessionController()
    _install_signal_handlers(app, gui)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    MaRGE()
