# MaRCoS Graphical User Interface

Python code associated to the Graphical User Interface of MaRCoS.

## Installation

The Graphical User Interface (GUI) for MaRCoS has been developed in python. Then, to run the GUI the user need to
install python3 into the computer.

Then, clone in the same folder the four repositories indicated here:

- marcos_client from *https://github.com/vnegnev*
- marcos_server from *https://github.com/vnegnev*
- marcos_extras from *https://github.com/vnegnev*
- PhysioMRI_GUI from *https://github.com/yvives/PhysioMRI_GUI*

Once the four repositories are cloned in your computer, the folders should look similar to 

<img src="resources/images/folder_example.png" alt="alt text">

**Figure 1.- Example of folder structure**

Once the four repositories are already cloned in the desired folder, we need to modify some files to make the GUI work.
The files to modify are:
- `local_config.py` from marcos_client folder. This file contains information about the version of the red pitaya, its
ip, and the GPA board used together with the redpitaya.
- `hw_config.py` from PhysioMRI_GUI/configs folder. This file contains information about the scanner hardware.
- `sys_config.py` from PhysioMRI_GUI/configs folder This file contains information to show the session windows of the
GUI.

Note that, once the repositories are downloaded, these files do not exist in the folder, but copies of the files do.

Finally, after proper modification of the files is done, the user can run the ***FirstMRI.py*** file to run the GUI.

## Description of the GUI

### Session window

When the user executes FirstMRI.py, the session window is displayed in the screen. 

<img src="resources/images/session.png" alt="alt text">

**Figure 2.- Session window**

The session window allows the user to input useful information like patient name, or weight. It also automatically
generates an ID number. This ID number is currently given by the date and time, but the user can modify the id number as
desired. Some information can only be selected between the provided options. The options available are given
according to the information contained in the file `sys_config.py`. Once the information has been completed, just click
on the `Launch GUI` button to execute the main window. At this point, a new folder is created in
`experiments/acquisitions` according to the Project and Subject ID.

### Main window

The main window opens after pushing the `Launch GUI` button on the session window. Here is where most of the work is
done, like calibrate the system, setup parameters of desired sequences, visualize results and others.

<img src="resources/images/main.png" alt="alt text">

**Figure 3.- Main window**

The main window is composed of different widgets
1) Menubar. It contains different options related to set up the redpitaya, sequences and others.
2) MaRCoS toolbar
3) Sequence toolbar
4) Image toolbar
5) Protocol toolbar
6) Sequence area. It includes custom modification or protocols to run pre-defined parameters
7) Console
8) History list table
9) Info table
10) Image area

Next, it is explained how to use the GUI through the most relevant controls.

## Toolbars
### MaRCoS toolbar (2)

The first that the user must do before to execute any sequence is to start the connection between the
GUI and the red pitaya. The connection to the red pitaya is done through the MaRCoS toolbar (2) or through the 
scanner menubar. 
This toolbar contains four different action buttons. From left to right we find:
- `Setup MaRCoS`. Execute MaRCoS init, MaRCoS server and init GPA, in that order.
- `MaRCoS init`. Update the last version of MaRCoS into the red pitaya.
- `MaRCoS server`. Connect or disconnect to the server.
- `Init GPA`. Execute a code to initialize the GPA board. The GUI must be previously connected to the server.

After executing `Setup MaRCoS` or `MaRCoS server`, the `MaRCoS server` button stays pressed and sequence buttons become 
enable. However, note that this happens even if the connection to the server fails.

***Poner figura de marcos server aquí***

### Sequence toolbar (3)

Here is the core of the GUI: run sequences. Sequence execution is performed through the sequence toolbar (3) or sequence
menubar. From left to right, the sequence toolbar contains:
- `Autocalibration`. It runs different sequences automatically and use the results to calibrate the system. This
task includes Larmor calibration, noise measurement, rabi flops and shimming.
- `Localizer`. It runs a fast RARE sequence with low resolution to help select the fov for next sequences.
- `Sequence to list`. It adds the current sequence configuration to the waiting list. It allows to continue working in
the GUI while the sequences are running.
- `Acquire`. It directly runs the current sequence configuration. When a new sequence is selected and the `Acquire`
button is clicked, the GUI may look frozen. This button will be deprecated in future versions.
- `Iterative run`. It activates the iterative mode. This is only available for some sequences such as larmor or noise.
Sequences to acquire images do not support iterative mode and the button will be toggled after the end of the sequence.
- `Bender button`.  This button place in the history list all the sequences contained in the selected protocol.
- `Plot sequence`. It plots the instructions that will be sent to the red pitaya.
- `Save parameters`. Save the parameters to ***experiments/parameterisations/SequenceNameInfo.year.month.day.hour.minutes.seconds.milliseconds.csv"
- `Load parameters`. Load input parameters files and update the sequence parameters to current sequence.
- `Save for calibration`. This button saves the parameters of the sequence into the ***calibration*** folder. In case the sequence
is used for autocalibration, it automatically loads parameters contained in ***calibration*** folder

### Figures toolbar (4)
It contains two buttons:
- `Fullscreen`. Expand image area to full screen mode.
- `Screenshot`. Take a snapshot of the GUI and save the image in ***screenshots*** folder 

### Protocols toolbar (5)
This toolbar allows the user to manage the protocols.
- `Add protocol`. Create a new protocol.
- `Remove protocol`. Delete a protocol.
- `Add sequence`. Add custom sequence to the selected protocol.
- `Remove sequence`. Remove a sequence from the selected protocol.

### GUI menubar (1)

1) `Scanner`
   1) `Setup MaRCoS`
   2) `MaRCoS init`
   3) `MaRCoS server`
   4) `Init GPA board`
2) `Protocoles`
   1) `New protocol`
   2) `Remove protocol`
   3) `New sequence`
   4) `Remove sequence`
3) `Sequences`
   1) `Load parameters`
   2) `Save parameters`
   3) `Save for calibration`
   4) `Sequence to list`
   5) `Acquire`
   6) `Bender`
   7) `Plot sequence`
4) `Session`
   1) `New session`

## Configure the system
### Autocalibration
The `Autocalibration` button runs a set of sequence to provide some required information to the GUI. `Autocalibration` runs four sequences:
- `Noise`
- `Larmor`
- `Rabi flops`
- `Shimming`

However, `Autocalibration` requires some files to run properly. The steps to create the file corresponding to each
sequence are described here and must be done for each sequence used by `autocalibration`:
1) Select one of the four mentioned sequences from the sequence list in the `Custom` tab.
2) Configure the desired input parameters.
3) Save the configuration using the `Save for calibration` button in the sequence toolbar (3).

`Save for calibration` button creates a .csv file in the ***calibration*** folder. Once the process has been complete for
the four sequences, four .csv files should be available in the folder and the user should be able to run autocalibration
properly.

### Localizer
Localizer typically is a fast sequence with low resolution to get information about the field of view. As for
autocalibration, it also needs a file with parameters to run when the `Localizer` button is clicked by the user.
To create the file:
1) Select `Localizer` sequence from the sequence list in `Custom` tab.
2) Configure the localizer with the desired parameters.
3) In the tab `Others` set the `Planes` inputs. This parameter is a list of three elements that can be 0 or 1. Use 1 (0)
to (not) acquire the corresponding plane. For example, [1, 1, 0] means that only sagittal and coronal planes will be
acquired after clicking the localizer button.
4) Save the configuration using the `Save for calibration` button in the sequence toolbar (3).

After saving configuration, clicking the `Localizer` button will place one Localizer sequence into the history list for
each selected plane in the `Others` tab.


## Run a custom sequence
1) To run a custom sequence, it is first required to run the `Autocalibration` and `Localizer`. After left-clicking the 
`Localizer` button, the sequence is placed in the waiting list (8) and it is executed in a parallel thread to prevent
the GUI from getting frozen. 
2) When the localizer finish, a message is shown in the console (7) and the item in the waiting list get its full name. 
The user can see the image by left double-clicking in the corresponding item of the waiting list, or can show more than
one image with right click menu. On the image widget,
click the FOV button to see a square representing the FOV that will be used to the next image. The user can change 
the size, position and angle (angle modification is not recommended in multiplot view) 
of the fov with the mouse, and the corresponding values in the sequence list will be
automatically updated. It is always preferred to use this method to modify the fov, given that this method update the
fov of all sequences, while typing the fov values directly in the corresponding text box of the sequence list will
affect only to the selected sequence.
3) Once the fov has been selected, select an image sequence and modify the parameters as desired. Right now, only three
image sequences are available: RARE, GRE and PETRA.
4) Run the sequence with the `Acquire button` (this will freeze the GUI until the end of the sequence) or place the
sequence in the waiting list with the `Sequence to list` button. It is preferred to use the `Sequence to list` button to
place the sequence in the waiting list. In this way, the user can continue programing the next image while others are
acquired. As in the localizer, once the result is ready, a message will appear in the console indicating that a sequence
is finished. Then you can see the results in the image area.

The parameters of the sequences are classified in four different fields:
1) ***RF***: it includes parameters related to RF signal like pulse amplitude, frequency or duration.
2) ***Image***: it includes parameters related to the image like field of view or matrix size.
3) ***Sequence***: it includes parameters related to contrast like echo time or repetition time.
4) ***Others***: a field to include other interesting parameters like gradient rise time or dummy pulses.

Each time the user runs a sequence:
1) It automatically save a file in ***experiments/parameterisation/SequenceName_last_parameters.csv***.
When the user initialize the GUI, it automatically loads the parameters from the files with ***last_parameters*** in the name to continue the session from the last point.
2) It creates three files in ***experiments/acquisitions*** inside the session folder
   1) a .mat raw data with name ***SequenceName.year.month.day.hour.minutes.secons.milliseconds.mat***.
   This file contains the inputs and outputs as well as any other useful variable.
   The values saved in the raw data are those contained inside the `mapVals` dictionary.
   2) a .csv file with the input parameters with name ***SequenceNameInfo.year.month.day.hour.minutes.secons.milliseconds.csv***
The .csv file is useful if you want to repeat a specific experiment by loading the parameters into the corresponding sequence with the ***Sequence/Load parameteres*** menu.
   3) a .dcm file in Dicom Standard 3.0

## Create a protocol
The GUI offers the possibility to run previously defined protocols. A protocol consists in a number of different
sequences with predefined parameters. `Protocols` are shown in the sequence area (6). The first time that the user runs
the GUI, the `Protocols` tab will be empty.

The user can create protocols by clicking the `New protocol` button. After clicking the `New protocol` button, a 
dialog box opens where the user can type the name of the protocol. The protocols must be created in the 
***protocols*** folder. Once the protocol is created, it will appear in the protocol list.

The user can add new sequences to the protocol. To add a sequence:
1) Select a sequence from the sequence list in the `Custom` tab.
2) Customize the sequence parameters as desired for the protocol.
3) Go to the `Protocol` tab and select the desired protocol.
4) Click the `Add sequence` button.
5) In the dialog box, write the name of the sequence to be shown in the `Protocol tab` tab.

After saving the sequence, it will appear in the `Protocols` in the corresponding protocol. To run the sequences from
the protocol, just double-click on the desired sequence to add the sequence to the waiting list.

Sequences can be deleted from protocols by right-clicking the sequence and selecting `Delete` or clicking the 
`Delete sequence` button. Also, protocols can be deleted by clicking the `Delete protocol` button and selecting the
protocol to delete from the dialog box.

## Structure of the folders and files in the GUI     

Here the GUI internal architecture is described. The GUI is composed of widgets and windows (even if a window is strictly
speaking a widget). Windows and widgets views are defined in the scripts contained in ***ui*** and ***widgets*** folders.
How the windows and widgets react to the user interaction is defined in scripts contained in the ***controller*** folder.

### `ui` folder
It conatins ***window_main.py***, ***window_session.py*** and ***window_postprocessing.py*** (uder development). The three
scripts contains a class that inherits from QMainWindow.

### `controller` folder

In this folder you will find all the python functions that control:
1) controller_main.py
2) controller_session.py
3) controller_postprocessing.py (under development)
4) controller_console.py
5) controller_figures.py
6) controller_history_list.py
7) controller_menu.py
8) controller_plot1d.py
9) controller_plot3d.py
10) controller_protocol_inputs.py
11) controller_protocol_list.py
12) controller_sequence_inputs.py
13) controller_sequence.list.py
14) controller_toolbar_figures.py
15) controller_toolbar_marcos.py
16) controller_toolbar_protocols.py
17) controller_toolbar_sequences.py

### `seq` folder

Folder that contains the different sequences that can be applied in the scanner.
In addition to the main sequences, it also contains the parent sequence `mriBlankSeq.py`.
Also in this folder we can find the `sequences.py` file, where we will import all the sequences that will be read by the GUI.

### `seq_standalone` folder

Folder that contains sequences that can be executed in the scanner independently of the GUI and are firstly stored here to be tested.
Last version of the GUI does not require of this folder.
Sequences in *`seq`* folder can be executed in the scanner with or without the GUI.
Then, in a future release of the GUI this folder will not be available.

### `configs` folder

- *hw_config.py*: a file with hardware information. 
These variables depend on the scanner hardware (i.e. gradients) or other general values.
When downloading the GUI by the first time, the file name is *hw_config.py.copy*.
The filename needs to be properly modified according to your hardware and renamed before to run the GUI.
- *sys_config.py*: file that contains usefully information for the session window.

### `protocols` folder

This folder contains the protocols created by the user.

### `experiments` folder

The results of the experiments are stored in this file. There are three folders inside:

- *acquisitions*: the scanner acquisitions are stored here.
A folder per day is created, with the date as name (YYYY.MM.DD).
The output of the scanner is stored inside, which is:
  - .mat file with raw data.
  - .dcm file with image.
  - .csv file with input parameters.
- *parameterization*: folder that contains:
  - the sequence last parameters in csv format
  - csv file generated when you click the *Save the parameters of a sequence to a file* icon in the GUI main window.

### `resources` folder
In this folder, the icons used in the main menu (and others) are stored as well as the images used in this readme.

### Other files 

- **Console function**: *stream.py* is a class used to write error messages in the console of the GUI.  

## How to add a new sequence to the GUI

In this section I will show how to create a new sequence.
To do this, I will go step by step until we can get a noise measurement.
At the end of this section, the user should be able to run simple sequences.

### Body of the sequence

We first start by creating a new *noise.py* file inside the *`seq`* folder (check the file to see the full code)
We need to import some different modules.
Importantly, to run the sequence wihtout the GUI we need to be able to import the experiment class that it is contained in the *`marcos_client`* repository.
Then we need to provide access to the folders:
````python
import os
import sys
#*****************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************
import experiment as ex
````
You may note that I also include the *`PhysioMRI_GUI`* folder.
This is due to some issues that I found under Windows OS.
Under linux OS you can just include the *`marcos_client`* directory.

Next we need to import the *mriBlankSequence* class.
*mriBlankSequence* contains many useful methods that are common to many sequences such as create parameters, rf pulses, gradient pulses, readout or save/load data between others.
New sequences should inherit from the mriBlankSequence.

````Python
import seq.mriBlankSeq as mriBlankSeq  # Import the mriBlankSequence for any new sequence.
````

Then we can create our class *Noise* that will inherit from *mriBlankSeq*.
To be properly used by the GUI, the class must contain at least four methods:
1) *sequenceInfo* that contains any useful information about the sequence.
2) *sequenceTime* that returns the time required by the sequence in minutes.
3) *sequenceRun* that inputs the instructions into the Red Pitaya.
It includes a keyword argument *seqPlot* that should be 1 if user do not want to run the instructions contained in the red pitaya.
This is used in the GUI to plot the sequence instead of run the experiment.
4) *sequenceAnalysis* to analyse the data acquired in the experiment.
````Python
class Noise(mriBlankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(Noise, self).__init__()

    def sequenceInfo(self):

    def sequenceTime(self):

    def sequenceRun(self, plotSeq=0):

    def sequenceAnalysis(self, obj=''):
````

### How to add input parameters to the sequence
To add new input parameters, the *mriBlankSequence* contains the method *addParameter* that needs to be run in the constructor of the class.
This method requires four keywords arguments: 
- *key*: a string to be used as key on the dictionaries
- *string*: a string to be shown in the GUI
- *value*: number or list of numbers to be shown in the GUI
- *field*: a string to classify the parameter into the boxes.
It can be 'RF', 'IMG', 'SEQ' or 'OTH'
```Python
def addParameter(self, key='', string='', val=0, field=''):
```
For the example of *Noise* sequence I will include the input parameters shown here:
````Python
    def __init__(self):
        super(Noise, self).__init__()
        # Input the parameters
        self.addParameter(key='seqName', string='NoiseInfo', val='Noise')
        self.addParameter(key='larmorFreq', string='Central frequency (MHz)', val=3.00, field='RF')
        self.addParameter(key='nPoints', string='Number of points', val=2500, field='RF')
        self.addParameter(key='bw', string='Acquision bandwidth (kHz)', val=50.0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
````
Note that the first parameter does not include the keyword *field*.
In this way, seqName is not included into the input parameters field of the GUI, but it is still saved in the raw data as information.

### sequenceInfo
The *sequenceInfo* method contains useful information about the sequence.
This method is not critical, and you may leave it empty, but make sure that the method exist because the GUI will ask for it.
For this example I will just do:
```Python
    def sequenceInfo(self):
        print("If you want a better world \n do open source")
```

### sequenceTime
The *sequenceTime* method returns the time of the sequence in minutes.
As the *sequenceInfo* method, it is no critical, but the GUI will ask for it.
```Python
    def sequenceTime(self):
        return(0)
```

### sequenceRun
In this method we will input the instructions into the Red Pitaya and we will run the sequence depending on the *plotSeq* value.
According to the GUI, *plotSeq = 0* is used to run the sequence and *plotSeq = 1* is used to plot the sequence.
To create the instructions, we will make use of the methods already available into the parent *mriBlankSequence*.

First we will create a local copy of the input parameters into the *sequenceRun*
````Python
        # Create inputs parameters
        larmorFreq = self.mapVals['larmorFreq'] # MHz
        nPoints = self.mapVals['nPoints']
        bw = self.mapVals['bw']*1e-3 # MHz
        rxChannel = self.mapVals['rxChannel']
````
Note that time parameters are defined in microseconds, as demanded by the Red Pitaya.

Now we create the experiment object.
The experiment object needs to be defined into the *self* object to be used by the methods of the parent class *mriBlankSequence*.
````Python
         bw = bw * hw.oversamplingFactor
         samplingPeriod = 1 / bw
         self.expt = ex.Experiment(lo_freq=larmorFreq,
                                   rx_t=samplingPeriod,
                                   init_gpa=False,
                                   gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                   print_infos=False)
         samplingPeriod = self.expt.get_rx_ts()[0]
         bw = 1/samplingPeriod/hw.oversamplingFactor
         acqTime = nPoints/bw
````
Note that:
- To fit an issue related to the CIC filter in the Red Pitaya, we apply an oversampling factor of six to the acquired data.
This factor is contained into the hardware module
````Python
import configs.hw_config as hw
````
Once the acquired data is obtained, we apply a decimation with a *fir* filter to recover the required acquisition bandwidth.
- One of the keyword arguments of the experiment is the sampling period *rx_t*.
The true sampling rate is not the same as the value that we input to the experiment class.
Once the experiment is defined, the user needs to get the true sampling rate with the method *get_rx_ts().
Then, the user must calculate the true bandwidth and acquisition time to avoid data miss registration.
- Future release will include a method that will do this automatically.

Now that we have the true values of the sampling period and sampling time, we input the instructions into the Red Pitaya.
````Python
         # SEQUENCE
         self.iniSequence(20, np.array((0, 0, 0)))
         self.rxGate(20, acqTime, rxChannel=rxChannel)
         self.endSequence(acqTime+40)
````
here we do:
- initialize the arrays to zero
- open the Rx gate to measure data
- finish the experiment by setting all the arrays to zero

At this point sequence instructions are already in the Red Pitaya.
Here is where we have to choose if the sequence is going to be run or not.
To do this, place the experiment *run* method in a conditional *if not plotSeq*.
In this way, we will run the sequence only if the keyword argument is *plotSeq = 0*:
````Python
         if not plotSeq:
             rxd, msgs = self.expt.run()
             data = rxd['rx%i'%rxChannel]*13.788
             data = sig.decimate(data, hw.oversamplingFactor, ftype='fir', zero_phase=True)
             self.mapVals['data'] = data
         self.expt.__del__()
````
If keyword argument *plotSeq = 0*, then the first is to run the experiment.
This is done by 
````Python
rxd, msgs = self.expt.run()
````
that provides two outputs:
- *rxd* that contains the data from the two different Rx channels in rxd['rx0'] and rxd['rx1']
- *msgs* that contains information provided by the server.

A correction that I use to employ is to multiply the signal by 13.788.
````Python
data = rxd['rx%i'%rxChannel]*13.788
````
This is to get the values in millivolts.

Finally, data must be decimated and filtered to obtain the results in the required acquisition bandwidth.
````Python
data = sig.decimate(data, hw.oversamplingFactor, ftype='fir', zero_phase=True)
````

Then, I save the data into the *mapVals* variable to save data into the rawData as .mat file:
````Python
self.mapVals['data'] = data
````

At this point the last remaining task is to delete the experiment object:
````Python
self.expt.__del__()
````

### The `mapVals` variable
`mapVals` is a key variable of the sequences.
It is a dictionary inherited from the `mriBlankSeq`.
The `addParameter` method creates the keys and values that are added to the dictionary.
The `saverRawData` method creates the .mat file with all information contained inside the `mapVals` dictionary.
Then, you have to save any information that you want into your rawdata in this variable.
Note that each time tha you run a sequence, the GUI clear all information in the mapVals except for the inputs.

### sequenceAnalysis
The *sequenceAnalysis* method is where we manipulate the data to be properly shown into the GUI or into our defined plot if we do not use the GUI.
Basically what this method need to do is
- Recover the acquired data previously saved in *self.mapVals['data']*
- Calculate the signal spectrum through the inverse fast fourier transform
- Save the rawData with *self.saveRawData()*
- Create the widget to be located into the GUI layout.

In our example, to recover the acquired data and other useful data:
````Python
acqTime = self.mapVals['acqTime'] # ms
nPoints = self.mapVals['nPoints']
bw = self.mapVals['bw'] # kHz
data = self.mapVals['data']
````

Then, we need to create the time and frequency arrays, and calculate the spectrum:
````Python
tVector = np.linspace(0, acqTime, num=nPoints) # ms
spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
fVector = np.linspace(-bw / 2, bw / 2, num=nPoints) * 1e3  # kHz
dataTime = [tVector, data]
dataSpec = [fVector, spectrum]
noise = np.abs(data)
noiserms = np.mean(noise)
self.mapVals['RMS noise'] = noiserms
self.mapVals['spectrum'] = dataSpec
````
Note that after processing the data, the results are saved in the *self.mapVals* dictionary.
Anything you want to save into the raw data file needs to be in the *self.mapVals* dictionary.
Then we can save the raw data with
````Python
self.saveRawData()
````

Once everything is ready, we have to create the pyqtgraph widget that is expected by the GUI.
For 1D plots, we have to import the class *SpectrumPlot* from the module *spectrumplot*.
````Python
from plotview.spectrumplot import SpectrumPlot
````
SpectrumPlot is a class that inherits from *GraphicsLayoutWidget* and requires different input arguments:
- *xData*: a numpy array with x data.
- *yData*: a list of numpy arrays with different y data.
- *legend*: a list of strings with legend for different curves in *yData*.
- *xLabel*: a string with the label of x axis.
- *yLabel*: a string with the label of y axis.
- *title*: a string with the label for the title.

for 3D images, we have to import the class *Spectrum3DPlot* from the module *spectrumplot*
````Python
from plotview.spectrumplot import Spectrum3DPlot # To show nice 2d or 3d images
````
Spectrum3DPlot is a class that make use of a modified version of *ImageView* module from *pyqtgraph*
to create nice 2D or 3D images.
This class requires different input arguments:
- *data*: a 3D numpy array with dimensions given by number of slices, phases and readouts.
- *xLabel*: a string with the label for the x axis.
- *yLabel*: a string with the label for the y axis.

To do this we type this:
````Python
# Plot signal versus time
timePlotWidget = SpectrumPlot(xData=self.dataTime[0],
                       yData=[np.abs(self.dataTime[1]), np.real(self.dataTime[1]), np.imag(self.dataTime[1])],
                       legend=['abs', 'real', 'imag'],
                       xLabel='Time (ms)',
                       yLabel='Signal amplitude (mV)',
                       title='Noise vs time, rms noise: %1.3f mV' %noiserms)

# Plot spectrum
freqPlotWidget = SpectrumPlot(xData=self.dataSpec[0],
                       yData=[np.abs(self.dataSpec[1])],
                       legend=[''],
                       xLabel='Frequency (kHz)',
                       yLabel='Mag FFT (a.u.)',
                       title='Noise spectrum')
````

Finally, we create the output variable that contains a list with the widgets to be shown into the GUI:
````Python
# create self.out to run in iterative mode
self.out = [timePlotWidget, freqPlotWidget]
return (self.out)
````
It is convenient to save the output list of widgets into *self.out*.
This varible is used in the GUI to update the widgets if the user repeats the experiment.
If *self.out* does not exist, the GUI will delete the previous widgets and new ones will be created each time the user repeat the same experient.

Note that the user can create its own widgets from *pyqtgraph* module and the GUI should be able to place it into the layout.
