"""
run_recon.py

This script dynamically loads and runs reconstruction functions for
different sequences stored in `.mat` files.

The reconstruction functions must be located in the `marge/recon`
directory. The script dynamically imports all Python modules in this
directory (excluding `__init__.py`) and looks for functions whose
names match the `seqName` field in the `.mat` file.

Each sequence-specific function must:
- Be named exactly as the `seqName` value in the `.mat` file.
- Be located in a Python module inside `marge/recon`.

The script can operate in two modes:
- Standard: Returns the output dictionary and plotting data.
- Standalone: Automatically plots the results in a window or
  sends them to a printer object.

Dependencies:
- numpy
- scipy
- matplotlib
- marge.recon
"""

import importlib
import os
from io import BytesIO

import scipy as sp
import inspect
from matplotlib import pyplot as plt

import marge.recon

module_path = marge.recon.__path__[0]

def run_recon(raw_data_path=None, mode=None, printer=None):
    """
    Run the reconstruction for a given sequence.

    This function dynamically loads the reconstruction function that
    corresponds to the `seqName` in the provided `.mat` file. It then
    executes the function and optionally plots the results.

    Parameters
    ----------
    raw_data_path : str, optional
        Path to the `.mat` file containing the raw sequence data.
        Must contain the `seqName` field.
    mode : str, optional
        If set to `"Standalone"`, the function will generate plots
        for the results. Otherwise, it simply returns the outputs.
    printer : object, optional
        If provided and `mode="Standalone"`, the plots are sent to
        this printer object instead of being displayed.

    Returns
    -------
    output_dict : dict
        Dictionary of derived numerical values, metrics, or metadata
        returned by the sequence-specific reconstruction function.
    output : list
        List of plotting dictionaries describing images or curves to
        visualize.

    Notes
    -----
    - The function requires that reconstruction scripts reside in
      `marge/recon`.
    - The reconstruction function name must exactly match the
      `seqName` from the `.mat` file.
    - If the sequence is unrecognized or not implemented, the
      function prints a warning and returns `False`.
    """
    # Load .mat file and get sequence name
    mat_data = sp.io.loadmat(raw_data_path)
    seq = mat_data['seqName'].item()  # Use `.item()` if it's a MATLAB cell

    # List all .py files in recon (excluding __init__.py)
    files = [
        f for f in os.listdir(module_path)
        if f.endswith(".py") and f != "__init__.py"
    ]

    # Dictionary to hold all discovered functions
    functions = {}

    for filename in files:
        module_name = filename[:-3]  # strip .py
        full_module = f"marge.recon.{module_name}"

        # Dynamically import the module
        mod = importlib.import_module(full_module)

        # Get all functions defined in this module
        funcs = inspect.getmembers(mod, inspect.isfunction)

        for name, func in funcs:
            # Optional: qualify function name with filename
            functions[name] = func

    # Check if the sequence is recognized and implemented
    if seq in functions.keys():
        recon = functions[seq]
        if recon is not None:
            output_dict, output = recon(raw_data_path=raw_data_path)
        else:
            print(f"Recon for '{seq}' not found.")
            return False
    else:
        print(f"Unknown or unimplemented sequence: {seq}")
        return False

    if mode == 'Standalone' and len(output) > 0:
        file_name = mat_data['fileName'][0]
        plot_results(output=output, title=file_name, printer=printer)

    return output_dict, output

def plot_results(output, title=None, printer=None):
    """
    Generate plots from the reconstruction output.

    This function creates a figure and inserts subplots based on the
    output dictionaries. Each item can be a curve plot or an image.

    Parameters
    ----------
    output : list
        List of dictionaries, each describing a plot to generate.
        Each dictionary must contain keys such as 'widget', 'xData',
        'yData', 'title', 'xLabel', 'yLabel', 'row', and 'col'.
    title : str, optional
        Overall title for the figure.
    printer : object, optional
        If provided, the plot is saved into a BytesIO buffer and
        passed to the printer's `add_image_to_story` method. If not
        provided, the figure is displayed using `plt.show()`.

    Returns
    -------
    None

    Notes
    -----
    - Automatically arranges subplots according to 'row' and 'col' keys.
    - Supports both 'curve' and 'image' widgets.
    - Uses tight_layout to prevent overlapping titles and labels.
    """
    # Determine the number of columns and rows for subplots
    cols = 0
    rows = 0
    for item in output:
        if item['row'] > rows:
            rows = item['row']
        if item['col'] > cols:
            cols =item['col']
    cols = cols + 1
    rows = rows + 1

    # Create the plot window
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))

    # Insert plots
    plot = 0
    for item in output:
        if item['widget'] == 'image':
            nz, ny, nx = item['data'].shape
            image_to_show = item['data'][int(nz / 2), :, :]
            plt.subplot(rows, cols, plot + 1)
            plt.imshow(image_to_show.T, cmap='gray')
            plt.title(item['title'])
            plt.xlabel(item['xLabel'])
            plt.ylabel(item['yLabel'])
        elif item['widget'] == 'curve':
            plt.subplot(rows, cols, plot + 1)
            n = 0
            for y_data in item['yData']:
                if isinstance(item['xData'], list):
                    plt.plot(item['xData'][n], y_data, label=item['legend'][n])
                else:
                    plt.plot(item['xData'], y_data, label=item['legend'][n])
                n += 1
            plt.title(item['title'])
            plt.xlabel(item['xLabel'])
            plt.ylabel(item['yLabel'])
        plot += 1

    # Set the figure title
    plt.suptitle(title)

    # Adjust the layout to prevent overlapping titles
    plt.tight_layout()

    if printer is not None:
        # Save the figure to a BytesIO buffer
        buf = BytesIO()
        fig.savefig(buf, format='PNG')  # or 'JPEG'
        plt.close(fig)
        buf.seek(0)  # Rewind the buffer
        printer.add_image_to_story(buf)
    else:
        plt.show()

if __name__ == '__main__':
    run_recon("../experiments/acquisitions/Example/2025.05.29.16.42/Example/None/mat/Noise.2025.05.29.16.42.11.897.mat",
              mode="Standalone")