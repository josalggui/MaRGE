"""
session_controller.py
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.uic import loadUiType, loadUi
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon
import pyqtgraph.exporters
import pyqtgraph as pg
from functools import partial
import os
import platform
import sys
import experiment as ex
from scipy.io import savemat
import csv
from sessionmodes import defaultsessions
from manager.datamanager import DataManager
from datetime import date,  datetime 
from globalvars import StyleSheets as style
from stream import EmittingStream
from local_config import ip_address
import cgitb
cgitb.enable(format = 'text')
import pdb
import numpy as np
import imageio
from plotview.spectrumplot import Spectrum3DPlot
from plotview.spectrumplot import SpectrumPlotSeq
from plotview.spectrumplot import SpectrumPlot
import time
from worker import Worker
st = pdb.set_trace
import copy
import configs.hw_config as hw

# Import sequences
from seq.sequences import defaultsequences
# from seq.localizer import Localizer

# Import controllers
from controller.batchcontroller import BatchController                                              # Batches
from controller.sequencecontroller import SequenceList                                              # Sequence list

# Stylesheets
import qdarkstyle

MainWindow_Form, MainWindow_Base = loadUiType('ui/mainview.ui')

class MainViewController(MainWindow_Form, MainWindow_Base):
    """
    MainViewController Class
    """
    onSequenceUpdate = pyqtSignal(str)
    onSequenceChanged = pyqtSignal(str)
    iterativeRun = False
    newRun = True
    marcosServer = False

    def __init__(self, session, parent=None):
        super(MainViewController, self).__init__(parent)

        # Load the mainview.up
        self.ui = loadUi('ui/mainview.ui')
        self.setupUi(self)

        # Set the style
        # self.styleSheet = style.breezeLight
        # self.setupStylesheet(self.styleSheet)
        self.styleSheet = qdarkstyle.load_stylesheet_pyqt5()
        self.setStyleSheet(self.styleSheet)
        
        # Initialisation of sequence list
        self.session = session
        self.sequencelist = SequenceList(self)
        self.sequencelist.setCurrentIndex(0)
        self.sequencelist.currentIndexChanged.connect(self.selectionChanged)
        self.layout_sequenceList.addWidget(self.sequencelist)
        self.sequence = self.sequencelist.currentText()
        self.session_label.setText(session["directory"])
                
        # Console
        self.cons = self.generateConsole('')
        self.layout_console.addWidget(self.cons)
        sys.stdout = EmittingStream(textWritten=self.onUpdateText)
        sys.stderr = EmittingStream(textWritten=self.onUpdateText)

        # List of results
        self.history_list = QListWidget()
        self.history_list.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.history_list.setMinimumWidth(400)
        self.history_list.itemDoubleClicked.connect(self.update_history_figure)
        self.history_list.itemClicked.connect(self.update_history_table)
        self.layout_history.addWidget(self.history_list)

        # Table with input parameters from historic images
        self.input_table = QTableWidget()
        self.layout_history.addWidget(self.input_table)

        # Create dictionaries to save historic widgets and inputs.
        self.history_list_outputs = {}
        self.history_list_inputs = {}
        self.history_list_rotations = {}    # Sequence fov rotations
        self.history_list_fovs = {}         # Sequence fov sizes
        self.history_list_shifts = {}       # Sequence fov displacements

        # Initialize multithreading
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads \n" % self.threadpool.maxThreadCount())

        # First plot
        self.firstPlot()

        # Connection to the server
        self.ip = ip_address
        
        # XNAT upload
        self.xnat_active = 'FALSE'

        # Toolbar Actions
        self.action_iterate.setCheckable(True)
        self.action_server.setCheckable(True)
        self.action_gpaInit.triggered.connect(self.initgpa)
        self.action_autocalibration.triggered.connect(self.autocalibration)
        self.action_changeappearance.triggered.connect(self.changeAppearanceSlot)
        self.action_acquire.triggered.connect(self.startAcquisition)
        self.action_add_to_list.triggered.connect(self.runToList)
        self.action_close.triggered.connect(self.close)
        self.action_exportfigure.triggered.connect(self.export_figure)
        self.action_viewsequence.triggered.connect(self.startSequencePlot)
        self.action_batch.triggered.connect(self.batch_system)
        self.action_XNATupload.triggered.connect(self.xnat)
        self.action_run_localizer.triggered.connect(self.startLocalizer)
        self.action_iterate.triggered.connect(self.iterate)
        self.action_server.triggered.connect(self.controlMarcosServer)
        self.action_copybitstream.triggered.connect(self.copyBitStream)

        # Menu Actions
        self.actionLoad_parameters.triggered.connect(self.load_parameters)
        self.actionSave_parameters.triggered.connect(self.save_parameters)
        self.actionSave_as_quick_calibration.triggered.connect(self.save_parameters_calibration)
        self.actionRun_sequence.triggered.connect(self.startAcquisition)
        self.actionPlot_sequence.triggered.connect(self.startSequencePlot)
        self.actionInit_GPA.triggered.connect(self.initgpa)
        self.actionNew_sesion.triggered.connect(self.change_session)
        self.actionInit_Red_Pitaya.triggered.connect(self.initRedPitaya)
        self.actionCopybitstream.triggered.connect(self.copyBitStream)
        self.actionInit_marcos_server.triggered.connect(self.controlMarcosServer)
        self.actionClose_marcos_server.triggered.connect(self.controlMarcosServer)
        # Calibration
        self.actionLocalizer.triggered.connect(self.startLocalizer)
        self.actionLarmor.triggered.connect(self.runLarmor)
        self.actionNoise.triggered.connect(self.runNoise)
        self.actionRabi_Flop.triggered.connect(self.runRabiFlop)
        self.actionCPMG.triggered.connect(self.runCPMG)
        self.actionInversion_Recovery.triggered.connect(self.runInversionRecovery)
        self.actionAutocalibration.triggered.connect(self.autocalibration)
        # Protocoles
        self.actionRARE_3D_T1.triggered.connect(self.protocoleRARE3DT1)
        self.actionRARE_3D_KNEE_T1.triggered.connect(self.protocoleRARE3DKNEET1)
        self.actionRARE_3D_HAND_RHO.triggered.connect(self.protocoleRARE3DHANDRHO)
        # Update the sequence parameters shown in the gui
        self.seqName = self.sequencelist.getCurrentSequence()
        defaultsequences[self.seqName].sequenceInfo()

        # Add the session to all sequences
        for sequence in defaultsequences.values():
            sequence.session = self.session

        # Show the gui maximized
        # self.showMaximized()

        # Start sequence listening
        # Pass the function to execute into the thread
        worker = Worker(self.waitingForRun)
        # Execute in a parallel thread
        self.threadpool.start(worker)


    def update_history_table(self, item):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: update the table when new element is clicked in the history list
        """
        # Get file name
        # fileName = item.text()[15::]
        name = item.text()[0:12]

        # Get the input data from history
        input_data = self.history_list_inputs[name]

        # Clear the table

        # Extract items from the input_data
        input_info = list(input_data[0])
        input_vals = list(input_data[1])

        # Set number of rows
        self.input_table.setColumnCount(1)
        self.input_table.setRowCount(len(input_info))

        # Input items into the table
        self.input_table.setVerticalHeaderLabels(input_info)
        self.input_table.setHorizontalHeaderLabels(['Values'])
        for m, item in enumerate(input_vals):
            newitem = QTableWidgetItem(str(item))
            self.input_table.setItem(m, 0, newitem)

    def update_history_figure(self, item):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: update the shown figure when new element is double clicked in the history list
        """
        # Get file self.currentFigure
        fileName = item.text()[15::]
        self.currentFigure = item.text()[0:12]

        # Get the widget from history
        output = self.history_list_outputs[self.currentFigure]

        # Get rotations and shifts from history
        rotations = self.history_list_rotations[self.currentFigure]
        shifts = self.history_list_shifts[self.currentFigure]
        fovs = self.history_list_fovs[self.currentFigure]
        for sequence in defaultsequences.values():
            sequence.rotations = rotations.copy()
            sequence.dfovs = shifts.copy()
            sequence.fovs = fovs.copy()

        # Clear the plotview
        self.clearPlotviewLayout()

        # Add plots to the plotview_layout
        win = pg.LayoutWidget()

        # Add label to show rawData self.currentFigure
        label = QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("background-color: black;color: white")
        win.addWidget(label, row = 0, col = 0, colspan = 2)
        label.setText(fileName)

        for item in output:
            if item['widget'] == 'image':
                image = Spectrum3DPlot(data = item['data'],
                                       xLabel = item['xLabel'],
                                       yLabel = item['yLabel'],
                                       title = item['title'])
                image.parent = self
                win.addWidget(image.getImageWidget(), row = item['row']+1, col = item['col'])
            elif item['widget'] == 'curve':
                plot = SpectrumPlot(xData = item['xData'],
                                    yData = item['yData'],
                                    legend = item['legend'],
                                    xLabel = item['xLabel'],
                                    yLabel = item['yLabel'],
                                    title = item['title'])
                win.addWidget(plot, row = item['row']+1, col = item['col'])
        self.plotview_layout.addWidget(win)
        self.newRun = True

    def controlMarcosServer(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: connect to marcos_server
        """
        if self.marcosServer:
            self.marcosServer = False
            # self.action_server.setIcon(
            #     QIcon('/home/physioMRI/git_repos/PhysioMRI_GUI/resources/icons/server-light.png'))
            self.action_server.setToolTip('Close marcos server')
            # self.action_server.setText('Close marcos server')
            os.system('ssh root@192.168.1.101 "killall marcos_server"')
            print("\n Server disconnected.")
        else:
            self.marcosServer = True
            # self.action_server.setIcon(
            #     QIcon('/home/physioMRI/git_repos/PhysioMRI_GUI/resources/icons/server-dark.png'))
            self.action_server.setToolTip('Connect to marcos server')
            # self.action_server.setText('Connect to marcos server')
            if platform.system() == 'Windows':
                os.system('ssh root@192.168.1.101 "killall marcos_server"')
                os.system('start ssh root@192.168.1.101 "~/marcos_server"')
            elif platform.system() == 'Linux':
                os.system('ssh root@192.168.1.101 "killall marcos_server"')
                os.system('ssh root@192.168.1.101 "~/marcos_server" &')
            print("\n Server connected.")

    def copyBitStream(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: execute copy_bitstream.sh
        """
        os.system('ssh root@192.168.1.101 "killall marcos_server"')
        if platform.system() == 'Windows':
            os.system('..\marcos_extras\copy_bitstream.sh 192.168.1.101 rp-122')
        elif platform.system() == 'Linux':
            os.system('../marcos_extras/copy_bitstream.sh 192.168.1.101 rp-122')
        print("\n MaRCoS updated")

    def initRedPitaya(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: execute startRP.sh: copy_bitstream.sh & marcos_server
        """
        if platform.system() == 'Windows':
            os.system('ssh root@192.168.1.101 "killall marcos_server"')
            os.system('start startRP.sh')
        elif platform.system() == 'Linux':
            os.system('ssh root@192.168.1.101 "killall marcos_server"')
            os.system('./startRP.sh &')
        print("\n MaRCoS updated and server connected.")
        print("Check the terminal for errors")
        print("If there are: 1) do 'Copybitstream', 2) do 'Init marcos server'")

    def runToList(self, seqName=None):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: add new run to the waiting list
        """
        # Load sequence name
        if seqName==None or seqName==False:
            seqName = self.sequencelist.getCurrentSequence()

        # Add item to the history list
        name = str(datetime.now())[11:23] + " | " + seqName
        self.history_list.addItem(name)

        # Save results into the history
        self.history_list_inputs[name[0:12]] = [list(defaultsequences[seqName].mapNmspc.values()),
                                          list(defaultsequences[seqName].mapVals.values()),
                                          True]

        # Save the rotation and shift to the history list
        self.history_list_rotations[name[0:12]] = defaultsequences[seqName].rotations.copy()
        self.history_list_shifts[name[0:12]] = defaultsequences[seqName].dfovs.copy()
        self.history_list_fovs[name[0:12]] = defaultsequences[seqName].fovs.copy()

    def waitingForRun(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: this method is continuously waiting for running new sequences in the history_list
        """
        while True:
            keys = list(self.history_list_inputs.keys()) # List of elements in the sequence history list
            element = 0
            for key in keys:
                if self.history_list_inputs[key][2]:
                    # Disable acquire button
                    self.action_acquire.setEnabled(False)

                    # Get the sequence to run
                    seqName = self.history_list_inputs[key][1][0]
                    sequence = copy.copy(defaultsequences[seqName])
                    # Modify input parameters of the sequence
                    n = 0
                    inputList = list(sequence.mapVals.keys())
                    for keyParam in inputList:
                        sequence.mapVals[keyParam] = self.history_list_inputs[key][1][n]
                        n += 1
                    # Run the sequence
                    output = self.runSequenceInlist(sequence=sequence)
                    time.sleep(1)
                    # Add item to the history list
                    fileName = sequence.mapVals['fileName']
                    self.history_list.item(element).setText(key + " | " + fileName)
                    # self.history_list.addItem()
                    # Save results into the history
                    self.history_list_outputs[key] = output
                    self.history_list_inputs[key] = [list(defaultsequences[seqName].mapNmspc.values()),
                                                     list(defaultsequences[seqName].mapVals.values()),
                                                     False]
                    # Delete outputs from the sequence
                    sequence.resetMapVals()
                    print(key+" Done!")
                else:
                    # Enable acquire button
                    self.action_acquire.setEnabled(True)
                element += 1
            time.sleep(0.1)

    def runSequenceInlist(self, sequence=None):
        # Save sequence list into the current sequence, just in case you need to do sweep
        sequence.sequenceList = defaultsequences

        # Save input parameters
        sequence.saveParams()

        # Create and execute selected sequence
        sequence.sequenceRun(0)

        # Update parameters, just in case something changed
        self.onSequenceUpdate.emit(self.sequence)

        time.sleep(1)

        # Do sequence analysis and get results
        return(sequence.sequenceAnalysis())

    def startAcquisition(self, seqName=None):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: run selected sequence
        """
        # Load sequence name
        if seqName == None or seqName == False:
            self.seqName = self.sequencelist.getCurrentSequence()
        else:
            self.seqName = seqName

        # Delete ouput if sequence is different from previous one
        if hasattr(self, "oldSeqName"):
            if self.seqName!=self.oldSeqName:
                self.newRun = True
                defaultsequences[self.seqName].deleteOutput()
        self.oldSeqName = copy.copy(self.seqName)

        if not hasattr(defaultsequences[self.seqName], 'out'):
            self.newRun = True

        # Save sequence list into the current sequence, just in case you need to do sweep
        defaultsequences[self.seqName].sequenceList = defaultsequences

        # Save input parameters
        defaultsequences[self.seqName].saveParams()

        # if not hasattr(defaultsequences[self.seqName], 'out'):  # If it is the first execution
        if self.newRun:
            self.newRun = False

            # Delete previous plots
            self.clearPlotviewLayout()

            # Add plots to the plotview_layout
            self.win = pg.LayoutWidget()

            # Create label with rawdata name
            self.label = QLabel()
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            self.label.setStyleSheet("background-color: black;color: white")
            self.win.addWidget(self.label, row = 0, col = 0, colspan = 2)

            # Create and execute selected sequence
            defaultsequences[self.seqName].sequenceRun(0)

            # Do sequence analysis and acquire de plots
            self.oldOut = defaultsequences[self.seqName].sequenceAnalysis()

            # Update parameters, just in case something changed
            self.onSequenceUpdate.emit(self.sequence)

            # Set name to the label
            fileName = defaultsequences[self.seqName].mapVals['fileName']
            self.label.setText(fileName)

            # Add item to the history list
            self.currentFigure = str(datetime.now())[11:23]
            name = self.currentFigure + " | " + fileName
            self.history_list.addItem(name)

            # Clear inputs
            defaultsequences[self.seqName].resetMapVals()

            # Save results into the history
            self.history_list_outputs[self.currentFigure] = self.oldOut
            self.history_list_inputs[self.currentFigure] = [list(defaultsequences[self.seqName].mapNmspc.values()),
                                                    list(defaultsequences[self.seqName].mapVals.values()),
                                                    False]

            # Save the rotation and shifts to the history list
            self.history_list_rotations[self.currentFigure] = defaultsequences[self.seqName].rotations.copy()
            self.history_list_shifts[self.currentFigure] = defaultsequences[self.seqName].dfovs.copy()
            self.history_list_fovs[self.currentFigure] = defaultsequences[self.seqName].fovs.copy()

            # Add plots to the plotview_layout
            self.plots = []
            for item in self.oldOut:
                if item['widget'] == 'image':
                    image = Spectrum3DPlot(data=item['data'],
                                           xLabel=item['xLabel'],
                                           yLabel=item['yLabel'],
                                           title=item['title'])
                    image.parent = self
                    self.win.addWidget(image.getImageWidget(), row=item['row'] + 1, col=item['col'])
                    defaultsequences[self.seqName].deleteOutput()
                elif item['widget'] == 'curve':
                    self.plots.append(SpectrumPlot(xData=item['xData'],
                                                yData=item['yData'],
                                                legend=item['legend'],
                                                xLabel=item['xLabel'],
                                                yLabel=item['yLabel'],
                                                title=item['title']))
                    self.win.addWidget(self.plots[-1], row=item['row'] + 1, col=item['col'])

            self.plotview_layout.addWidget(self.win)

            # Iterate in parallel thread (only for 1d plots)
            x = defaultsequences[self.seqName]
            if self.iterativeRun and hasattr(defaultsequences[self.seqName], 'out'):
                # Pass the function to execute into the thread
                worker = Worker(self.repeatAcquisition)  # Any other args, kwargs are passed to the run function
                # Execute in a parallel thread
                self.threadpool.start(worker)

            # Deactivate the iterative buttom if sequence is not iterable (2d and 3d plots)
            if not hasattr(defaultsequences[self.seqName], 'out') and self.action_iterate.isChecked():
                self.action_iterate.toggle()

        else:
            # Pass the function to execute into the thread
            worker = Worker(self.repeatAcquisition)  # Any other args, kwargs are passed to the run function
            # Execute in a parallel thread
            self.threadpool.start(worker)

    def repeatAcquisition(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: executed when you repeat some of the calibration sequences
        """
        # If single repetition, set iterativeRun True for one step
        singleRepetition = not self.action_iterate.isChecked()
        if singleRepetition: self.iterativeRun = True

        # Acquire while iterativeRun is True
        while self.iterativeRun:
            # Create and execute selected sequence
            defaultsequences[self.seqName].sequenceRun(0)

            # Do sequence analysis and acquire de plots
            self.newOut = defaultsequences[self.seqName].sequenceAnalysis()

            # Set name to the label
            fileName = defaultsequences[self.seqName].mapVals['fileName']
            self.label.setText(fileName)

            # Add item to the history list
            self.currentFigure = str(datetime.now())[11:23]
            name = self.currentFigure + " | " + fileName
            self.history_list.addItem(name)

            for plotIndex in range(len(self.newOut)):
                oldCurves = self.plots[plotIndex].plotitem.listDataItems()
                for curveIndex in range(len(self.newOut[plotIndex]['yData'])):
                    x = self.newOut[plotIndex]['xData']
                    y = self.newOut[plotIndex]['yData'][curveIndex]
                    oldCurves[curveIndex].setData(x, y)

            # Clear inputs
            defaultsequences[self.seqName].resetMapVals()

            # Save results into the history
            self.history_list_outputs[self.currentFigure] = self.newOut
            self.history_list_inputs[self.currentFigure] = [list(defaultsequences[self.seqName].mapNmspc.values()),
                                                            list(defaultsequences[self.seqName].mapVals.values()),
                                                            False]

            # Save the rotation and shifts to the history list
            self.history_list_rotations[self.currentFigure] = defaultsequences[self.seqName].rotations.copy()
            self.history_list_shifts[self.currentFigure] = defaultsequences[self.seqName].dfovs.copy()
            self.history_list_fovs[self.currentFigure] = defaultsequences[self.seqName].fovs.copy()

            # Stop repetitions if single acquision
            if singleRepetition:
                self.iterativeRun = False
                # self.action_iterate.toggle()

    def iterate(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: swtich the iterative mode
        """
        if self.action_iterate.isChecked():
            self.action_iterate.setToolTip('Switch to single run')
            self.iterativeRun = True
        else:
            self.action_iterate.setToolTip('Switch to iterative run')
            self.iterativeRun = False

    def startSequencePlot(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: plot sequence instructions
        """
        # Delete previous plots
        self.clearPlotviewLayout()

        # Load sequence name
        self.seqName = self.sequencelist.getCurrentSequence()

        # Create sequence to plot
        print('Plot sequence')
        defaultsequences[self.seqName].sequenceRun(1)  # Run sequence only for plot

        # Get sequence to plot
        out = defaultsequences[self.seqName].sequencePlot()  # Plot results

        # Create plots
        n = 0
        plot = []
        for item in out:
            plot.append(SpectrumPlotSeq(item[0], item[1], item[2], 'Time (ms)', 'Amplitude (a.u.)', item[3]))
            if n > 0: plot[n].plotitem.setXLink(plot[0].plotitem)
            n += 1
        for n in range(4):
            self.plotview_layout.addWidget(plot[n])

    def startLocalizer(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: run localizer
        """

        print('Start localizer')

        # Load sequence name
        seqName = 'Localizer'

        defaultsequences[seqName].loadParams(directory="calibration")

        # Sagittal localizer
        if defaultsequences[seqName].mapVals['planes'][0]:
            defaultsequences[seqName].mapVals['axesOrientation'] = [0, 1, 2]
            self.runToList(seqName=seqName)
            time.sleep(1)

        # Transversal localizer
        if defaultsequences[seqName].mapVals['planes'][1]:
            defaultsequences[seqName].mapVals['axesOrientation'] = [1, 2, 0]
            self.runToList(seqName=seqName)
            time.sleep(1)

        # Coronal localizer
        if defaultsequences[seqName].mapVals['planes'][2]:
            defaultsequences[seqName].mapVals['axesOrientation'] = [2, 0, 1]
            self.runToList(seqName=seqName)

    def autocalibration(self):
        self.clearPlotviewLayout()

        # Include here the sequences to run on autocalibration
        seqNames = [
                    # 'Larmor',
                    'Noise',
                    # 'RabiFlops',
                    # 'Shimming'
                    ]

        # Add plots to the plotview_layout
        self.win = pg.LayoutWidget()
        self.plots = []

        for seqName in seqNames:
            # Execute the sequence
            sequence = defaultsequences[seqName]
            sequence.sequenceRun()
            output = sequence.sequenceAnalysis()
            delattr(sequence, 'out')

            # Add item to the history list
            fileName = sequence.mapVals['fileName']
            name = str(datetime.now())[11:23] + " | " + fileName
            self.history_list.addItem(name)

            # Save results into the history
            self.history_list_outputs[name[0:12]] = output
            self.history_list_inputs[name[0:12]] = [list(sequence.mapNmspc.values()),
                                                    list(sequence.mapVals.values()),
                                                    False]

            # Specific for larmor
            if seqName == 'Larmor':
                for seq in defaultsequences:
                    seq.mapVals['larmorFreq'] = hw.larmorFreq

            # Specific for noise
            if seqName == 'Noise':
                # Create label with rawdata name
                self.label = QLabel()
                self.label.setAlignment(QtCore.Qt.AlignCenter)
                self.label.setStyleSheet("background-color: black;color: white")
                self.win.addWidget(self.label, row=0, col=0, colspan=2)
                fileName = sequence.mapVals['fileName']
                self.label.setText(fileName)

                # Noise spectrum
                item = output[1]
                self.plots.append(SpectrumPlot(xData=item['xData'],
                                               yData=item['yData'],
                                               legend=item['legend'],
                                               xLabel=item['xLabel'],
                                               yLabel=item['yLabel'],
                                               title=item['title']))
                self.win.addWidget(self.plots[-1], row=1, col=0)

            # Specifi for rabi
            if seqName == 'Rabi':
                item = output[0]
                self.plots.append(SpectrumPlot(xData=item['xData'],
                                               yData=item['yData'],
                                               legend=item['legend'],
                                               xLabel=item['xLabel'],
                                               yLabel=item['yLabel'],
                                               title=item['title']))
                self.win.addWidget(self.plots[-1], row=2, col=0)

            # Specific for shimming
            if seqName == 'Shimming':
                for seq in defaultsequences:
                    seq.mapVals['shimming'] = outShim[1]
                item = outShim[0]
                self.plots.append(SpectrumPlot(xData=item['xData'],
                                               yData=item['yData'],
                                               legend=item['legend'],
                                               xLabel=item['xLabel'],
                                               yLabel=item['yLabel'],
                                               title=item['title']))
                self.win.addWidget(self.plots[-1], row=3, col=0)

        # Add windows to the layout
        self.plotview_layout.addWidget(self.win)

        # Update the inputs of the sequences
        self.onSequenceUpdate.emit(self.sequence)

    def protocoleRARE3DT1(self):
        # Load parameters
        defaultsequences['RARE'].loadParams(directory='protocoles', file='RARE_3D_T1.csv')

        # Load fov and dfov from hw_config
        defaultsequences['RARE'].mapVals['fov'] = hw.fov
        defaultsequences['RARE'].mapVals['dfov'] = hw.dfov

        # Set larmor frequency to the value into the hw_config file
        defaultsequences['RARE'].mapVals['larmorFreq'] = hw.larmorFreq

        # Run the sequence
        self.startAcquisition(seqName='RARE')

    def protocoleRARE3DKNEET1(self):
        # Load parameters
        defaultsequences['RARE'].loadParams(directory='protocoles', file='RARE_3D_KNEE_T1.csv')

        # Load fov and dfov from hw_config
        defaultsequences['RARE'].mapVals['fov'] = hw.fov
        defaultsequences['RARE'].mapVals['dfov'] = hw.dfov

        # Set larmor frequency to the value into the hw_config file
        defaultsequences['RARE'].mapVals['larmorFreq'] = hw.larmorFreq

        # Run the sequence
        self.startAcquisition(seqName='RARE')

    def protocoleRARE3DHANDRHO(self):
        # Load parameters
        defaultsequences['RARE'].loadParams(directory='protocoles', file='RARE_3D_HAND_RHO.csv')

        # Load fov and dfov from hw_config
        defaultsequences['RARE'].mapVals['fov'] = hw.fov
        defaultsequences['RARE'].mapVals['dfov'] = hw.dfov

        # Set larmor frequency to the value into the hw_config file
        defaultsequences['RARE'].mapVals['larmorFreq'] = hw.larmorFreq

        # Run the sequence
        self.startAcquisition(seqName='RARE')

    def runLarmor(self):
        defaultsequences['Larmor'].loadParams(directory="calibration")
        self.startAcquisition(seqName="Larmor")

    def runNoise(self):
        defaultsequences['Noise'].loadParams(directory="calibration")
        self.startAcquisition(seqName="Noise")

    def runRabiFlop(self):
        defaultsequences['RabiFlops'].loadParams(directory="calibration")
        self.startAcquisition(seqName="RabiFlops")

    def runCPMG(self):
        defaultsequences['CPMG'].loadParams(directory="calibration")
        self.startAcquisition(seqName="CPMG")

    def runInversionRecovery(self):
        defaultsequences['InversionRecovery'].loadParams(directory="calibration")
        self.startAcquisition(seqName="InversionRecovery")

    def lines_that_start_with(self, str, f):
        return [line for line in f if line.startswith(str)]
    
    # @staticmethod
    def generateConsole(self, text):
        con = QTextEdit()
        con.setText(text)
        return con

    def onUpdateText(self, text):
        cursor = self.cons.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.cons.setTextCursor(cursor)
        self.cons.ensureCursorVisible()
    
    def __del__(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        os.system('ssh root@192.168.1.101 "killall marcos_server"') # Kill marcos server
        print('GUI closed successfully!')
        super().closeEvent(event)
    
    @pyqtSlot(bool)
    def changeAppearanceSlot(self) -> None:
        """
        Slot function to switch application appearance
        @return:
        """
        if self.styleSheet is style.breezeDark:
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        else:
            self.setupStylesheet(style.breezeDark)

    def close(self):
        os.system('ssh root@192.168.1.101 "killall marcos_server"')
        print('GUI closed successfully!')
        sys.exit()

    def setupStylesheet(self, style) -> None:
        """
        Setup application stylesheet
        @param style:   Stylesheet to be set
        @return:        None
        """
        self.styleSheet = style
        file = QFile(style)
        file.open(QFile.ReadOnly | QFile.Text)
        stream = QTextStream(file)
        stylesheet = stream.readAll()
        self.setStyleSheet(stylesheet)  

    def selectionChanged(self,item):
        # Load sequence name
        seqName = self.sequencelist.getCurrentSequence()

        # Delete output of the sequence
        defaultsequences[self.seqName].deleteOutput()

        self.sequence = self.sequencelist.currentText()
        self.onSequenceUpdate.emit(self.sequence)
        self.action_acquire.setEnabled(True)
        # self.clearPlotviewLayout()
    
    def clearPlotviewLayout(self) -> None:
        """
        Clear the plot layout
        @return:    None
        """
        for i in reversed(range(self.plotview_layout.count())):
            self.plotview_layout.itemAt(i).widget().setParent(None)

    def clearLocalizerLayout(self) -> None:
        """
        Clear the localizer layout
        @return:    None
        """
        for i in reversed(range(self.localizer_layout.count())):
            self.localizer_layout.itemAt(i).widget().setParent(None)
    
    def save_data(self):
        
        dataobject: DataManager = DataManager(self.data_avg, self.sequence.lo_freq, len(self.data_avg), [self.sequence.n_rd, self.sequence.n_ph, self.sequence.n_sl], self.sequence.BW)
        dict1=vars(defaultsessions[self.session])
        dict2 = vars(self.sequence)
        dict = self.merge_two_dicts(dict1, dict2)
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M")
        dt2 = date.today()
        dt2_string = dt2.strftime("%d-%m-%Y")
        dict["rawdata"] = self.rxd
        dict["fft"] = dataobject.f_fftData
        if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('experiments/acquisitions/%s' % (dt2_string))
            
        if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
            
        if not os.path.exists('/media/physiomri/TOSHIBA\ EXT/experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('/media/physiomri/TOSHIBA\ EXT/%s'% (dt2_string) )
            
        if not os.path.exists('/media/physiomri/TOSHIBA\ EXT/experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('/media/physiomri/TOSHIBA\ EXT/%s/%s'% (dt2_string, dt_string) )   
            
        savemat("experiments/acquisitions/%s/%s/%s.mat" % (dt2_string, dt_string, self.sequence), dict)
        savemat("/media/physiomri/TOSHIBA\ EXT/%s/%s/%s.mat" % (dt2_string, dt_string, self.sequence), dict)
        
        self.messages("Data saved")

        if hasattr(self.parent, 'f_plotview'):
            exporter1 = pyqtgraph.exporters.ImageExporter(self.f_plotview.scene())
            exporter1.export("experiments/acquisitions/%s/%s/Freq%s.png" % (dt2_string, dt_string, self.sequence))
        if hasattr(self.parent, 't_plotview'):
            exporter2 = pyqtgraph.exporters.ImageExporter(self.t_plotview.scene())
            exporter2.export("experiments/acquisitions/%s/%s/Temp%s.png" % (dt2_string, dt_string, self.sequence))

        from controller.WorkerXNAT import Worker
        
        if self.xnat_active == 'TRUE':
            # Step 2: Create a QThread object
            self.thread = QThread()
            # Step 3: Create a worker object
            self.worker = Worker()
            # Step 4: Move worker to the thread
            self.worker.moveToThread(self.thread)
            # Step 5: Connect signals and slots
            self.thread.started.connect(partial(self.worker.run, 'experiments/acquisitions/%s/%s' % (dt2_string, dt_string)))
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            
            # Step 6: Start the thread
            self.thread.start()

    def export_figure(self):
        
        dt = datetime.now()
        dt_string = dt.strftime("%d-%m-%Y_%H:%M")
        dt2 = date.today()
        dt2_string = dt2.strftime("%d-%m-%Y")

        if not os.path.exists('experiments/acquisitions/%s' % (dt2_string)):
            os.makedirs('experiments/acquisitions/%s' % (dt2_string))    
        if not os.path.exists('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)):
            os.makedirs('experiments/acquisitions/%s/%s' % (dt2_string, dt_string)) 
                    
        exporter1 = pyqtgraph.exporters.ImageExporter(self.f_plotview.scene())
        exporter1.export("experiments/acquisitions/%s/%s/Freq%s.png" % (dt2_string, dt_string, self.sequence))
        exporter2 = pyqtgraph.exporters.ImageExporter(self.t_plotview.scene())
        exporter2.export("experiments/acquisitions/%s/%s/Temp%s.png" % (dt2_string, dt_string, self.sequence))
        
        self.messages("Figures saved")
   
    def merge_two_dicts(self, x, y):
        z = x.copy()   # start with keys and values of x
        z.update(y)    # modifies z with keys and values of y
        return z
   
    def load_parameters(self):
    
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Parameters File', "experiments/parameterisations/")

        seq = defaultsequences[self.sequence]
        mapValsOld = seq.mapVals
        with open(file_name, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for l in reader:
                mapValsNew = l

        seq.mapVals = {}

        # Get key for corresponding modified parameter
        for key in seq.mapKeys:
            dataLen = seq.mapLen[key]
            valOld = mapValsOld[key]
            valNew = mapValsNew[key]
            valNew = valNew.replace('[', '')
            valNew = valNew.replace(']', '')
            valNew = valNew.split(',')
            if type(valOld) == str:
                valOld = [valOld]
            elif dataLen == 1:
                valOld = [valOld]
            dataType = type(valOld[0])

            inputNum = []
            for ii in range(dataLen):
                if dataType == float or dataType == np.float64:
                    try:
                        inputNum.append(float(valNew[ii]))
                    except:
                        inputNum.append(float(valOld[ii]))
                elif dataType == int:
                    try:
                        inputNum.append(int(valNew[ii]))
                    except:
                        inputNum.append(int(valOld[ii]))
                else:
                    try:
                        inputNum.append(str(valNew[0]))
                        break
                    except:
                        inputNum.append(str(valOld[0]))
                        break
            if dataType == str:
                seq.mapVals[key] = inputNum[0]
            else:
                if dataLen == 1:  # Save value into mapVals
                    seq.mapVals[key] = inputNum[0]
                else:
                    seq.mapVals[key] = inputNum

        self.onSequenceUpdate.emit(self.sequence)
        print("\nParameters of %s sequence loaded" %(self.sequence))

    def save_parameters_calibration(self):
        seq = defaultsequences[self.sequence]

        # Save csv with input parameters
        with open('calibration/%s_last_parameters.csv' % seq.mapVals['seqName'], 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=seq.mapKeys)
            writer.writeheader()
            mapVals = {}
            for key in seq.mapKeys:  # take only the inputs from mapVals
                mapVals[key] = seq.mapVals[key]
            writer.writerows([seq.mapNmspc, mapVals])

        print("\nParameters of %s sequence saved" %(self.sequence))

    def save_parameters(self):
        dt = datetime.now()
        dt_string = dt.strftime("%Y.%m.%d.%H.%M.%S.%f")[:-3]
        seq = defaultsequences[self.sequence]

        # Save csv with input parameters
        if not os.path.exists('experiments/parameterisations'):
            os.makedirs('experiments/parameterisations')
        with open('experiments/parameterisations/%s.%s.csv' % (seq.mapNmspc['seqName'], dt_string), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=seq.mapKeys)
            writer.writeheader()
            mapVals = {}
            for key in seq.mapKeys:  # take only the inputs from mapVals
                mapVals[key] = seq.mapVals[key]
            writer.writerows([seq.mapNmspc, mapVals])

        # self.messages("Parameters of %s sequence saved" %(self.sequence))
        print("\n Parameters of %s sequence saved" %(self.sequence))
        
    def plot_sequence(self):
        
        plotSeq=1
        self.sequence = defaultsequences[self.sequencelist.getCurrentSequence()]
        self.seqName = self.sequence.mapVals['seqName']
        defaultsequences[self.seqName].sequenceRun(plotSeq=plotSeq)
        
    def messages(self, text):
        msg = QtWarningMsg()
        # msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.exec();
        
    def calibrate(self):
        calibrationApp = CalibrationController(parent=self)
        calibrationApp.show()
        
    def initgpa(self):
        expt = ex.Experiment(init_gpa=True)
        expt.add_flodict({
            'grad_vx': (np.array([100]), np.array([0])),
        })
        expt.run()
        expt.__del__()
        print("GPA init done!")

    def batch_system(self):
        batchW = BatchController(self, self.sequencelist)
        batchW.show()

    def xnat(self):
        
        if self.xnat_active == 'TRUE':
            self.xnat_active = 'FALSE'
            self.action_XNATupload.setIcon(QIcon('/home/physioMRI/git_repos/PhysioMRI_GUI/resources/icons/upload-outline.svg') )
            self.action_XNATupload.setToolTip('Activate XNAT upload')
        else:
            self.xnat_active = 'TRUE'
            self.action_XNATupload.setIcon(QIcon('/home/physioMRI/git_repos/PhysioMRI_GUI/resources/icons/upload.svg') )
            self.action_XNATupload.setToolTip('Deactivate XNAT upload')
            
    def change_session(self):
        from controller.sessionviewer_controller import SessionViewerController
        sessionW = SessionViewerController(self.session)
        sessionW.show()
        self.hide()

    def firstPlot(self):
        """
        @author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
        @email: josalggui@i3m.upv.es
        @Summary: show the initial figure
        """
        logo = imageio.imread("resources/images/logo.png")
        # logo = np.flipud(logo)
        self.clearPlotviewLayout()
        welcome = Spectrum3DPlot(logo.transpose([1, 0, 2]),
                                 title='Institute for Instrumentation in Molecular Imaging (i3M)')
        welcome.hideAxis('bottom')
        welcome.hideAxis('left')
        welcome.showHistogram(False)
        welcome.imv.ui.menuBtn.hide()
        welcome.imv.ui.roiBtn.hide()
        welcomeWidget = welcome.getImageWidget()
        self.plotview_layout.addWidget(welcomeWidget)
