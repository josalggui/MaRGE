# PhysioMRI_GUI

Python code associated to the Graphical User Interface of PhysioMRI scanner from MRILab.

## How to install the GUI


**FirstMRI.py** --> Executes the application.

## How to operate with the GUI

The first window to appear is the sessionWindow (Figure 1), where you can first select the object that is going to be imaged and introduce parameters like the ID, demographic variables, etc.

![Figure 1.- Session Window](images/SessionWindow_completo.png)

Then, we launch the GUI main window by clicking to the corresponding icon (Figure 2). We distinguish 4 different zones in the GUI main window: 1) Main menu icons, 2) Sequence selection, 3) Parameters area, 4) Plotting area and 5) Console area.

![Figure 2.- Main Window](images/MainWindow_completo.png)

### GUI main menu

![GUI main menu](images/GUI_main_icons.png)

1) Initialization of the GPA: makes the initilization of the GPA gradients card.
2) Calibration of the system: launches a window with all the calibration functions.
3) Activate upload to XNAT: if this button is enabled, the GUI uploads the MRI to the XNAT system if XNAT is installed.
4) Start acquisition: start the acquisition with the selected sequence and the parameters introduced.
5) Load parameters of a sequence from a file: load sequence and associated parameters previously saved in a file to the GUI. 
6) Save the parameters of a sequence to a file: it saves the name of the sequence and the parameters introduced in the GUI into a file. It is useful to reuse configurations or to acquire multiple images with different configurations sequentially.
7) View the defined sequence: plot the sequence (RF pulses and X, Y and Z gradients).
8) Change session: it closes the `GUI main window` and it opens the `Session Window` again. 
9) Batch acquisition: it allows acquiring multiple images with multiple sequences sequentially, without human intervention.
10) Change Window appearance
11) Close window

#### Calibration of the system

![Figure 3.- Calibration window](images/CalibrationWindow_completo.png)

#### Batch acquisition

It allows acquiring multiple images with different sequences sequentially, without human intervention (Figure 4). 

![Figure 4.- Batch acquisition window](images/BatchWindow_completo.png)

With the `plus` button we can add files previously saved in the GUI main window. We can remove these acquisitions with the `minus` button.


## Structure of the folders and files in the GUI     

#### ui folder: it contains the files created with Designer Qt5, which are a basic structure of the different windows. 
		
There are different kinds of ui files in the GUI. On the one hand, we have the general structure of the four windows: session window, main window, calibration window and batch window.
Main and calibration uis are very similar with each other (Figure 5), and session and batch uis are also similar with each other (Figure 6).

	![Figure 5.- Main and calibration uis](images/main_calib.png)
	![Figure 6.- Session and batch uis](images/session_batch.png)

On the other hand, there are ui files that represent single elements that will be introduced programally to the general structure of the windows (to one of the four previous windows).
These elements are:

		- inputparameter.ui	
		- gradients.ui
		- inputshimming.ui
		 
![Figure 7.- Input parameter ui](images/ui_elements.png)

#### controller folder: 



#### seq folder:



#### resources folder:



#### images folder:



#### stylesheets folder



        --> sec: contains the sequences implemented in the GUI
                - radial (radial.py)
                - turbo spin echo (turboSpinEcho.py)
                - gradient echo (gradEcho.py)

        --> resources: contains the different resources used in the GUI
                - icons: contains a library of icons. Some of them are used in the GUI.

        --> stylesheets: contains different styles that can be applied to the GUI
                - breeze-dark.qss
                - breeze-light.qss

