[![](https://img.shields.io/badge/marcos__client-blue)](https://github.com/vnegnev/marcos_client)
[![](https://img.shields.io/badge/marcos__server-blue)](https://github.com/vnegnev/marcos_server)
[![](https://img.shields.io/badge/marcos__extras-blue)](https://github.com/vnegnev/marcos_extras)

# MaRGE (MaRCoS Graphical Environment)

This repository contains the Python code for the MaRCoS Graphical Environment (MaRGE), a system for magnetic resonance imaging research. The GUI provides a user-friendly interface to interact with the MaRCoS system.

Take a look at the MaRGE [Wiki](https://github.com/josalggui/MaRGE/wiki)! (under development)

Take a look at the MaRGE [Documentation](https://josalggui.github.io/MaRGE/)! (under development)

## [Setting up a Red Pitaya](https://github.com/josalggui/MaRGE/wiki/Setting-up-Red-Pitaya)

## [Setting up MaRGE](https://github.com/josalggui/MaRGE/wiki/Setting-up-MaRGE)

## Description of the GUI

## Toolbars

#### MaRCoS Toolbar (2)

Before executing any sequence, the user must establish a connection between the GUI and the Red Pitaya. This connection can be initiated through the MaRCoS toolbar (2) or via the scanner menubar. The MaRCoS toolbar consists of four distinct action buttons, from left to right:

- `Setup MaRCoS`: This button executes a sequence of actions in the following order: MaRCoS initialization, starting the MaRCoS server, and initializing the GPA board.

- `MaRCoS init`: This action updates the Red Pitaya with the latest version of the MaRCoS software.

- `MaRCoS server`: Use this button to connect to or disconnect from the MaRCoS server.

- `Init GPA`: Clicking this button triggers a code execution to initialize the GPA board. 
It's important to note that the GUI must be connected to the server before initializing the GPA board.
In case an interlock (under development) is connected to GPA and RFPA from Barthel, it also enables the power modules
remotely.

Upon executing `MaRCoS server` button remains pressed, and the sequence buttons become enabled. It's worth mentioning that this state remains even if the connection to the server fails (under development).
However, if connection is done, a terminal will show the information shown in Figure 4.

![MaRCoS Server](resources/images/server.png)

**Figure 4: MaRCoS server connection information**

#### Sequence Toolbar (3)

The Sequence toolbar is at the heart of the GUI, allowing users to run sequences efficiently. Sequence execution can be initiated through the Sequence toolbar (3) or the Sequence menubar. From left to right, the Sequence toolbar offers the following options:

- `Autocalibration`: This button automatically runs a series of sequences and uses the results to calibrate the system. The calibration process includes Larmor calibration, RF coil impedance matching, noise measurement, Rabi flops, and shimming.

- `Localizer`: Use this button to execute a quick RARE sequence with low resolution, helping users select the field of view (FOV) for subsequent sequences.

- `Sequence to List`: Clicking this button adds the current sequence configuration to the waiting list. This feature allows you to continue working in the GUI while sequences are running.

- `Acquire`: This button directly runs the current sequence configuration. Note that when a new sequence is selected and the `Acquire` button is clicked, the GUI may appear frozen. Please be aware that this button is slated for deprecation in future versions.

- `Iterative Run`: Activate this mode for sequences that support it, such as Larmor or noise measurements. Sequences that acquire images do not support iterative mode, and this button will be automatically toggled after the sequence ends.

- `Bender Button`: This button places all the sequences contained in the selected protocol into the history list.

- `Plot Sequence`: It generates a visualization of the instructions that will be sent to the Red Pitaya.

- `Save Parameters`: Use this button to save the sequence parameters to a CSV file located at `experiments/parameterisations/SequenceNameInfo.year.month.day.hour.minutes.seconds.milliseconds.csv`.

- `Load Parameters`: Load input parameter files and update the sequence parameters to the current sequence.

- `Save for Calibration`: This button saves the sequence parameters in the `calibration` folder. If the sequence is intended for autocalibration, it will automatically load parameters from the `calibration` folder.

These options within the Sequence toolbar provide users with the flexibility and control needed to execute various sequences and manage their parameters effectively.

#### Figures Toolbar (4)

The Figures toolbar provides quick access to two essential functions:

- `Fullscreen`: Clicking this button expands the image area to full-screen mode for a more detailed view.

- `Screenshot`: Use this button to capture a snapshot of the GUI. The captured image is saved in the `screenshots` folder.

These functions enhance the user experience by allowing for better visualization and documentation of the GUI's interface.

#### Protocols Toolbar (5)

The Protocols toolbar facilitates the management of protocols within the GUI. It includes the following options:

- `Add Protocol`: Clicking this button creates a new protocol, enabling users to organize and categorize their sequences.

- `Remove Protocol`: Use this button to delete a protocol that is no longer needed.

- `Add Sequence`: Adding a custom sequence to the selected protocol is made easy with this button.

- `Remove Sequence`: Clicking this button removes a sequence from the currently selected protocol.

The Protocols toolbar streamlines the organization and customization of your workflow, making it easier to work with sequences and protocols.

#### GUI Menubar (1)

The GUI menubar is a centralized hub for accessing various functions and features of MaRGE. It is divided into several categories, each containing specific options:

1. **Scanner**
   - `Setup MaRCoS`: Executes MaRCoS initialization, starts the MaRCoS server, and initializes the GPA board in sequence.
   - `MaRCoS init`: Updates the Red Pitaya with the latest MaRCoS software version.
   - `MaRCoS server`: Allows you to connect or disconnect from the MaRCoS server.
   - `Init GPA board`: Initializes the GPA board. Note that this action requires prior connection to the server.

2. **Protocols**
   - `New protocol`: Creates a new protocol for organizing and categorizing sequences.
   - `Remove protocol`: Deletes an existing protocol.
   - `New sequence`: Adds a custom sequence to the selected protocol.
   - `Remove sequence`: Removes a sequence from the currently selected protocol.

3. **Sequences**
   - `Load parameters`: Loads input parameter files and updates the sequence parameters to match the current sequence.
   - `Save parameters`: Saves the sequence parameters to a designated CSV file.
   - `Save for calibration`: Saves the sequence parameters to the `calibration` folder, useful for autocalibration purposes.
   - `Sequence to list`: Adds the current sequence configuration to the waiting list, allowing you to continue working while sequences run.
   - `Acquire`: Directly executes the current sequence configuration. (Scheduled for deprecation in future versions)
   - `Bender`: Places all sequences contained in the selected protocol into the history list.
   - `Plot sequence`: Generates a visualization of the instructions that will be sent to the Red Pitaya.

4. **Session**
   - `New session`: Initiates a new session, providing a fresh start for inputting essential information (comming soon).

The GUI menubar serves as a user-friendly interface for navigating and accessing the various functionalities of MaRGE.

### [Setting up the autocalibration](https://github.com/josalggui/MaRGE/wiki/Setting-up-autocalibration)

### [Setting up the localizer](https://github.com/josalggui/MaRGE/wiki/Setting-up-localizer)

### [Run custom sequences](https://github.com/josalggui/MaRGE/wiki/Run-custom-sequences)

### [Protocols](https://github.com/josalggui/MaRGE/wiki/Protocols)

## Structure of Folders and Files in the GUI

The internal architecture of MaRGE is organized into distinct folders and files that define its functionality and user interface. Understanding this structure can be helpful for those interested in further customization or development.

### `ui` Folder

The `ui` folder contains scripts that define the main windows of the GUI. Currently, it includes the following scripts:

- **window_main.py**: This script defines a class that inherits from QMainWindow. It forms the foundation of the main GUI window, where most user interactions take place.

- **window_session.py**: Similar to `window_main.py`, this script also defines a class that inherits from QMainWindow. It is responsible for managing the session window, which allows users to input essential information before conducting experiments.

- **window_postprocessing.py** (under development): This script is intended to define a class for a post-processing window, which will likely offer tools for analyzing and visualizing data after experiments.

### `widgets` Folder

The `widgets` folder contains scripts that define individual widgets or components used within the GUI. These widgets are responsible for various specific functionalities and user interactions.

### `controller` Folder

Scripts in the `controller` folder play a crucial role in determining how the windows and widgets react to user interactions. They define the logic behind the GUI's behavior, ensuring that it responds appropriately to user input.

As the GUI evolves and additional features are developed, more scripts and files may be added to these folders, enhancing the functionality and usability of MaRCoS.

Understanding this folder and file structure can provide a foundation for those interested in extending or customizing MaRGE to suit their specific research needs.

### `seq` Folder

The `seq` folder is where you can access the different sequences that can be applied in the scanner. It contains not only the primary sequences but also a parent sequence named `mriBlankSeq.py`. Additionally, you'll find the `sequences.py` file in this folder, which serves as an import point for all the sequences that the GUI can utilize.

### `configs` Folder

Within the `configs` folder, you'll encounter two essential configuration files:

- **hw_config.py**: This file stores hardware-related information crucial for the GUI. Variables in this file depend on the specific scanner hardware, such as gradients, or other essential values. Upon downloading the GUI for the first time, the filename is typically named `hw_config.py.copy`. Be sure to modify the filename appropriately to match your hardware and rename it before running the GUI.

- **sys_config.py**: This file contains useful information utilized by the session window of the GUI.

- **autotuning.py**: This file contains the serial number of the arduino used to control the autotuning.

### `protocols` Folder

The `protocols` folder is where user-created protocols are stored. Protocols are collections of predefined sequences with preset parameters, allowing for streamlined experimental workflows.

### `experiments` Folder

The `experiments` folder serves as the repository for storing the results of experiments conducted within the GUI. Within this folder, you'll find two subfolders:

- **acquisitions**: Scanner acquisitions are stored here, with each day's data stored in a separate folder labeled with the date (YYYY.MM.DD). The outputs of the scanner include:
  - .mat files containing raw data.
  - .dcm files with images.
  - .csv files containing input parameters.

- **parameterization**: This folder contains important data, including:
  - Sequence last parameters in CSV format.
  - CSV files generated when you click the "Save the parameters of a sequence to a file" icon in the GUI main window.

### `resources` Folder

In the `resources` folder, you'll find various icons used in the main menu and other parts of the GUI, as well as the images used in this README.

This structured organization of folders and files ensures that MaRGE remains efficient and organized, allowing for effective experimentation and customization.

# Adding a New Sequence to the GUI

In this section, we'll guide you through the process of creating and adding a new sequence to MaRGE. By following these steps, you'll be able to run simple sequences within the GUI. We'll use the example of creating a noise measurement sequence as a starting point.

## Body of the Sequence

1. **Create a New Sequence File**: Start by creating a new Python file for your sequence. In this example, we'll create a `noise.py` file inside the `seq` folder.

2. **Import Required Modules**: In your sequence file, import the necessary modules. It's crucial to have access to the `experiment` class from the `marcos_client` repository. Additionally, ensure access to the `marcos_client` and `MaRGE` folders for execution in standalone mode. Also, import the `mriBlankSequence` class, which contains various methods commonly used in many sequences. New sequences should inherit from the mriBlankSequence class. We also include the `configs.hw_config` module to get access to hardware properties and `configs.hw.units` that will be used to set the units of input parameters.

```python
import os
import sys
#*****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory = os.path.dirname(parent_directory)

# Define the subdirectories you want to add to sys.path
subdirs = ['MaRGE', 'marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
#******************************************************************************
import controller.experiment_gui as ex
import numpy as np
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.
import configs.hw_config as hw
import configs.units as units

```

3. **Create the `Noise` Sequence Class**: In your sequence file (`noise.py` in this example), create the `Noise` class that will inherit from the `mriBlankSeq.MRIBLANKSEQ` class. To be properly used by the GUI, the `Noise` class should contain at least four methods:

    a) **`sequenceInfo`**: This method should provide any useful information about the sequence, such as a brief description or relevant details (it can be empty).

    b) **`sequenceTime`**: Implement the `sequenceTime` method, which should return the time required by the sequence in minutes (it can returns 0).

    c) **`sequenceRun`**: The `sequenceRun` method is responsible for inputting the instructions into the Red Pitaya. It includes a `plotSeq` keyword argument that should be set to `1` if you want to plot the sequence or `0` for running the experiment. Another keyword is `demo` that can be established to `True` or `False` in case user can run simulated signals.

    d) **`sequenceAnalysis`**: Lastly, the `sequenceAnalysis` method is used to analyze the data acquired during the experiment. It includes a `mode` keyword that I use to plot call `plotResults` method from `mriBlankSeq` in case this parameter is set to `'standalone'`, but it can be used at convenience.

Here's an example of what the `Noise` class structure might look like:

```python
class Noise(mriBlankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(Noise, self).__init__()

    def sequenceInfo(self):
        # Provide sequence information here

    def sequenceTime(self):
        # Return the time required by the sequence in minutes

    def sequenceRun(self, plotSeq=0, demo=False):
        self.demo = demo
        
        # Create sequence instructions
        
        # Input instructions into the Red Pitaya
        
        # Use the plotSeq argument to control plotting versus running.
        
        # Use the demo argument to control if you want to simulate signals.

    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        ...
        result1 = {}
        result2 = {}
        ...
        self.output = [result1, result2]
        
        return self.output
        
        
        # Implement data analysis logic here
```

## Adding Input Parameters to the Sequence

```python
def __init__(self):
    super(Noise, self).__init__()

    # Input the parameters
    self.addParameter(key='seqName', string='NoiseInfo', val='Noise')
    self.addParameter(key='larmorFreq', string='Central frequency (MHz)', val=3.00, field='RF', units=units.MHz)
    self.addParameter(key='nPoints', string='Number of points', val=2500, field='RF')
    self.addParameter(key='bw', string='Acquisition bandwidth (kHz)', val=50.0, field='RF', units=units.kHz)
    self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
```

In this section, we'll walk you through the process of adding input parameters to your sequence. This step is essential for configuring and customizing your sequence within MaRGE.

To add new input parameters, we'll utilize the `addParameter` method available in the `mriBlankSequence` class. This method should be executed in the constructor of your sequence class and has keyword arguments:

- **key**: A string to be used as a key in dictionaries.
- **string**: A string that will be displayed in the GUI, serving as a user-friendly label.
- **value**: A numerical value or a list of numbers that will be presented in the GUI.
- **field**: A string to classify the parameter into one of four categories: 'RF', 'IMG', 'SEQ', or 'OTH'. The parameter will be shown in a tab according to this field.
- **units**: kHz, ms or similar called from `units` module
- **tip**: A string with tips regarding the parameter that will be shown in the tooltip bar of the GUI.

`mriBlankSeq` has the method `sequenceAtributes` that creates attributes with names according to the `key` field and associates values according to the `value` field taking into account the `units`. This method is executed by the GUI when a new run starts.

In this example, the `Noise` sequence is configured with several input parameters, each associated with a key, a user-friendly label, a default value, and categorized under the 'RF' field. The 'seqName' parameter, however, doesn't include the field keyword, making it informational but not displayed as a user-configurable input.

By following this approach, you can seamlessly add and customize input parameters for your sequence, allowing users to tailor the sequence parameters to their specific needs within MaRGE.

## Defining the `sequenceInfo` Method

```python
def sequenceInfo(self):
    print("Be open-mind,\nDo open-source")
```

In your sequence, you have the option to include a `sequenceInfo` method that provides useful information about the sequence. While this method is not critical for the functionality of your sequence, it is recommended to have it in place because the GUI may request this information.

## Implementing the `sequenceTime` Method

```python
def sequenceTime(self):
    return 0  # Duration in minutes (modify as needed)
```

The `sequenceTime` method is responsible for returning the duration of the sequence in minutes. While this method is not critical for the sequence's functionality, it is recommended to have it in place because the GUI may request this information.

In this example, the `sequenceTime` method returns a default duration of 0 minutes. You should adjust the return value to reflect the actual duration of your sequence.

## Implementing the `sequenceRun` Method

```python
def sequenceRun(self, plotSeq=0, demo=False):
    self.demo = demo
    
    # 1) Update bandwidth by oversampling factor
    self.bw = self.bw * hw.oversamplingFactor # Hz
    samplingPeriod = 1 / self.bw # s
    
    # 2) Create the experiment object
    self.expt = ex.Experiment(
        lo_freq=self.larmorFreq * 1e-6,
        rx_t=samplingPeriod * 1e6,
        init_gpa=False,
        gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
        print_infos=False
    )
    
    # 3) Get true sampling period from experiment object
    samplingPeriod = self.expt.get_rx_ts()[0] # us
    
    # 4) Update bandwidth and acquision time
    self.bw = 1 / samplingPeriod / hw.oversamplingFactor # MHz
    acqTime = self.nPoints / self.bw # us
    self.mapVals['acqTime'] = acqTime

    # 5) Create sequence instructions
    self.iniSequence(20, np.array((0, 0, 0)))
    self.rxGate(20, acqTime, rxChannel=rxChannel)
    self.endSequence(acqTime+40)
    
    # 6) Execute the sequence
    if not plotSeq:
        rxd, msgs = self.expt.run()
        data = rxd['rx%i'%rxChannel]*13.788
        data = sig.decimate(data, hw.oversamplingFactor, ftype='fir', zero_phase=True)
        self.mapVals['data'] = data
    self.expt.__del__()
```

The `sequenceRun` method plays a pivotal role in your sequence as it manages the input of instructions into the Red Pitaya and controls whether to run or plot the sequence. The value of the `plotSeq` parameter determines the behavior: `plotSeq = 0` is used to run the sequence, while `plotSeq = 1` is used to plot the sequence.

You may assume that values associated to keys created in the constructor are available as attributes.

Here's a step-by-step breakdown of how to implement the `sequenceRun` method:

1. **Get the True Bandwidth and Sampling Period**: To address an issue related to the CIC filter in the Red Pitaya, an oversampling factor is applied to the acquired data. This factor can be encapsulated in the hardware module.

2. **Initialize the Experiment Object**: Next, initialize the experiment object (`self.expt`) using the parameters you've defined. The experiment object must be defined within the self object so that it can be accessed by methods of the parent class `mriBlankSequence`.

3. **Obtain True Sampling Rate**: After defining the experiment, obtain the true sampling rate used by the experiment object using the `get_rx_ts()` method.

4. **Update Acquision Parameters**: Calculate the true bandwidth and acquisition time based on the true sampling rate to prevent data misregistration and ensure precise measurements.

5. **Create sequence instructions**: Now that we have the true values of the sampling period and sampling time, we create the instructions of the pulse sequence.
   1. **Initialization**:
      - To begin the sequence, we initialize the necessary arrays and parameters.
      - In this step, we ensure that all relevant variables are set to zero and that the Red Pitaya is ready for data acquisition.

   2. **Rx Gate Configuration**:
      - The next step involves configuring the Rx gate for data measurement.
      - We specify the duration of the Rx gate, which is determined by the acquisition time and the selected Rx channel.
   
   3. **Completing the Sequence**:
      - To finish the experiment, we perform cleanup tasks.
      - All arrays and variables are reset to their initial values, ensuring a clean slate for the next sequence or experiment.
      - The total duration of the sequence is adjusted to account for the Rx gate duration and additional time for safety.

6. **Execute the sequence**:
   1. **Conditional Execution**:
      - Before running the sequence, we determine whether it should be executed or just plotted.
      - The decision is made based on the value of the `plotSeq` keyword argument, where:
        - `plotSeq = 0`: The sequence will be executed to collect data.
        - `plotSeq = 1`: The sequence will be plotted without data acquisition.
      - This flexibility allows users to visualize the sequence before running it, which can be useful for verification and debugging.

   2. **Data Acquisition (if applicable)**:
      - If the sequence is set to run (i.e., `plotSeq = 0`), the Red Pitaya performs data acquisition as instructed.
      - Data collected during the Rx gate period is processed to yield meaningful experimental results.
      - The acquired data is decimated, filtered, and stored for subsequent analysis.

   3. **Cleanup**:
      - Once data acquisition is complete, the sequence is finalized.
      - Cleanup tasks ensure that the Red Pitaya and related resources are reset to their initial state.
      - This step is crucial for maintaining the integrity of subsequent experiments.

## Implementing `sequenceAnalysis` method

```Python
    def sequenceAnalysis(self, mode=None):
        # Set the mode attribute
        self.mode = mode

        # 1) Data retrieval
        acqTime = self.mapVals['acqTime']  # ms
        data = self.mapVals['data']

        # 2) Data processing
        tVector = np.linspace(0, acqTime, num=self.nPoints)  # Time vector in milliseconds (ms)
        spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))  # Signal spectrum
        fVector = np.linspace(-self.bw / 2, self.bw / 2, num=self.nPoints) * 1e3  # Frequency vector in kilohertz (kHz)
        dataTime = [tVector, data]  # Time-domain data as a list [time vector, data]
        dataSpec = [fVector, spectrum]  # Frequency-domain data as a list [frequency vector, spectrum]
        
        # 3) Create result dictionaries
        # Plot signal versus time
        result1 = {'widget': 'curve',
                   'xData': dataTime[0],
                   'yData': [np.abs(dataTime[1]), np.real(dataTime[1]), np.imag(dataTime[1])],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Noise vs time',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 0,
                   'col': 0}

        # Plot spectrum
        result2 = {'widget': 'curve',
                   'xData': dataSpec[0],
                   'yData': [np.abs(dataSpec[1])],
                   'xLabel': 'Frequency (kHz)',
                   'yLabel': 'Mag FFT (a.u.)',
                   'title': 'Noise spectrum',
                   'legend': [''],
                   'row': 1,
                   'col': 0}
        
        # 4) Define output
        self.output = [result1, result2]
        
        # 5) Save the rawData
        self.saveRawData()
        
        # In case of 'Standalone' execution, use the plotResults method from mriBlankSeq
        if self.mode == 'Standalone':
            self.plotResults()
        
        # 6) Return the self.output
        return self.output
```


The `sequenceAnalysis` method plays a crucial role in preparing data for display in the GUI or generating custom plots when the GUI is not utilized. This method performs a series of tasks to achieve this goal:

1. **Data Retrieval**: It retrieves previously acquired data stored in `self.mapVals`.

2. **Data processing**: Using the Inverse Fast Fourier Transform, it calculates the signal spectrum from the acquired data.

3. **Results**: Creates dictionaries that will be used in the GUI layout for user interaction and data display.

4. **Define Outputs**: `self.outputs` is a list of results that will be used by the `plotResults` method and by the GUI to generate the figure widget. It is convenient to save the list in `self.output` attribute. This attribute is used in the GUI to update the widgets if the user repeats the experiment.

5. **Save rawData**: Saves the raw data using the `self.saveRawData()` method.

6. **return**: the `sequenceAnalysis` method should return the `self.output` variable containing the list of dictionaries that provides the results.

## Execute the sequence in Standalone
```Python
if __name__=='__main__':
    seq = Noise()   # Creates the sequence object
    seq.sequenceAtributes()     # Creates attributes according to input parameters keys and values
    seq.sequenceRun(demo=False) # Execute the sequence
    seq.sequenceAnalysis(mode='Standalone') # Show results
```

# Additional notes

## CIC filter issues

It's crucial to be aware of a systematic delay that occurs as a result of the CIC filter applied to the acquired data in the Red Pitaya. This delay consists of 3 data points and should be taken into account when processing and analyzing acquired data.

The CIC filter's delay impacts the alignment of acquired data and can influence the timing of various sequence operations. This means that the timestamp associated with a data point may not reflect its true acquisition time accurately. Understanding and accommodating this delay is essential for accurate data processing and interpretation within MaRGE.

To mitigate potential issues related to the CIC filter's delay, it is also recommended to discard the first five to ten data points during data processing. This practice helps in stabilizing the data and removing any transient effects caused by the filter's delay. Additionally, consider adjusting timestamps or applying correction factors to accurately account for the delay when conducting precise time-sensitive analyses.

The `mriBlankSeq` module already includes methods for rx gating that account for these considerations, simplifying the implementation of sequences and ensuring reliable data acquisition and processing (not shown in this example).

## The `mapVals` Variable

The `mapVals` variable is a crucial element within the sequences of MaRGE. It serves as a dictionary inherited from the `mriBlankSeq` class, playing a vital role in managing and preserving information throughout the sequence execution. Below, we explore the significance and usage of the `mapVals` variable:

- **Initialization and Structure**:
  - The `mapVals` dictionary is initialized with predefined key-value pairs.
  - These keys act as unique identifiers for specific information, and their associated values can encompass numbers or lists of numbers.

- **Storage of Information**:
  - During the sequence's execution, you have the flexibility to store pertinent information within the `mapVals` dictionary. This information can encompass various aspects, including parameters, interim results, or any other data deemed essential.

- **Saving Data in Raw Data**:
  - Upon the sequence's completion, the `saveRawData` method is utilized to generate .mat and .dcm files containing all the data stored within the `mapVals` dictionary. This .mat file plays a pivotal role in preserving the experimental data and results.
  - **TODO**: save data in ISMRMD-format and NIFTI-format. Add XNAT.
- **Persistent Inputs**:
  - It is worth noting that, although the `mapVals` dictionary is cleared of most information after each sequence run, the inputs defined through the `addParameter` method remain intact. This ensures the retention of critical input parameters for reference and potential use in future experiments.

In summary, the `mapVals` variable functions as a dynamic storage space for various types of data within a sequence. It facilitates the management and organization of vital information throughout the sequence execution process. Additionally, it guarantees that essential input parameters are accessible for reference and subsequent experiments.

