[![](https://img.shields.io/badge/marcos__client-blue)](https://github.com/vnegnev/marcos_client)
[![](https://img.shields.io/badge/marcos__server-blue)](https://github.com/vnegnev/marcos_server)
[![](https://img.shields.io/badge/marcos__extras-blue)](https://github.com/vnegnev/marcos_extras)

# MaRGE (MaRCoS Graphical Environment)

This repository contains the Python code for the MaRCoS Graphical Environment (MaRGE), a system for magnetic resonance imaging research. The GUI provides a user-friendly interface to interact with the MaRCoS system.

Take a look at the MaRGE [Wiki](https://github.com/josalggui/MaRGE/wiki)! (under development)

Take a look at the MaRGE [Documentation](https://josalggui.github.io/MaRGE/)! (under development)

### [Setting up a Red Pitaya](https://github.com/josalggui/MaRGE/wiki/Setting-up-Red-Pitaya)

### [Setting up MaRGE](https://github.com/josalggui/MaRGE/wiki/Setting-up-MaRGE)

### [Description of the GUI](https://github.com/josalggui/MaRGE/wiki/Interface-description)

### [Toolbars](https://github.com/josalggui/MaRGE/wiki/Toolbars)

### [Setting up the autocalibration](https://github.com/josalggui/MaRGE/wiki/Setting-up-autocalibration)

### [Setting up the localizer](https://github.com/josalggui/MaRGE/wiki/Setting-up-localizer)

### [Run custom sequences](https://github.com/josalggui/MaRGE/wiki/Run-custom-sequences)

### [Protocols](https://github.com/josalggui/MaRGE/wiki/Protocols)

### [Adding a New Sequence to the GUI](https://github.com/josalggui/MaRGE/wiki/Create-your-own-sequence)


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

