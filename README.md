# PhisioMRI_GUI

Python code associated to the Graphical User Interface of PhysioMRI scanner from MRILab.

- FirstMRI.py --> Executes the application.

- Folders and files:

        --> ui: contains the files created with Designer Qt5
                - MainWindow.ui: architecture of the main window
                - ConnDialog.ui: architecture of the connection dialog

        --> controller: contains the files that control the different windows of the GUI
                - mainviewcontroller.py: controls the main window
                - connectionDialog.py: controls the window to connect with the RP

        --> sec: contains the sequences implemented in the GUI
                - radial (radial.py)
                - turbo spin echo (turboSpinEcho.py)
                - gradient echo (gradEcho.py)

        --> resources: contains the different resources used in the GUI
                - icons: contains a library of icons. Some of them are used in the GUI.

        --> stylesheets: contains different styles that can be applied to the GUI
                - breeze-dark.qss
                - breeze-light.qss

