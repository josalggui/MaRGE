import importlib
import os

import scipy as sp
import inspect
from matplotlib import pyplot as plt
import recon

module_path = recon.__path__[0]

def run_recon(raw_data_path=None, mode=None):
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
        full_module = f"recon.{module_name}"

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
            output = recon(raw_data_path=raw_data_path)
        else:
            print(f"Recon for '{seq}' not found.")
            return None
    else:
        raise ValueError(f"Unknown or unimplemented sequence: {seq}")

    if mode == 'Standalone':
        file_name = mat_data['fileName'][0]
        plot_results(output=output, title=file_name)

    return output

def plot_results(output, title=None):
    """
    Plot results in a standalone window.

    This method generates plots based on the output data provided. It creates a plot window, inserts each plot
    according to its type (image or curve), sets titles and labels, and displays the plot.

    Returns:
        None

    """
    # Determine the number of columns and rows for subplots
    cols = 1
    rows = 1
    for item in output:
        if item['row'] + 1 > rows:
            rows += 1
        if item['col'] + 1 > cols:
            cols += 1

    # Create the plot window
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))

    # Insert plots
    plot = 0
    for item in output:
        if item['widget'] == 'image':
            nz, ny, nx = item['data'].shape
            plt.subplot(rows, cols, plot + 1)
            plt.imshow(item['data'][int(nz / 2), :, :], cmap='gray')
            plt.title(item['title'])
            plt.xlabel(item['xLabel'])
            plt.ylabel(item['yLabel'])
        elif item['widget'] == 'curve':
            plt.subplot(rows, cols, plot + 1)
            n = 0
            for y_data in item['yData']:
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

    # Show the plot
    plt.show()

if __name__ == '__main__':
    run_recon("../experiments/acquisitions/Example/2025.05.29.16.42/Example/None/mat/Noise.2025.05.29.16.42.11.897.mat",
              mode="Standalone")