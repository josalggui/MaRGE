import numpy as np
import scipy as sp
import inspect
from matplotlib import pyplot as plt
import configs.hw_config as hw


def run_recon(raw_data_path=None, mode=None):
    # Load .mat file and get sequence name
    mat_data = sp.io.loadmat(raw_data_path)
    seq = mat_data['seqName'].item()  # Use `.item()` if it's a MATLAB cell

    functions = {
        name: obj
        for name, obj in globals().items()
        if inspect.isfunction(obj) and obj.__module__ == __name__
    }

    # output = functions['Noise'](raw_data_path=raw_data_path)

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

def Noise(raw_data_path=None):
    if raw_data_path is None:
        return None

    # load .mat
    mat_data = sp.io.loadmat(raw_data_path)

    # Get data and time vector
    data = mat_data['data']
    data = np.squeeze(data)
    bw = mat_data['bw'][0][0]  # kHz
    acq_time = mat_data['nPoints'][0][0] / bw
    t_vector = np.linspace(0, acq_time, np.size(data))

    # Get spectrum a frequency vector
    spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(data)))
    spectrum = np.squeeze(spectrum)
    f_vector = np.linspace(-bw / 2, bw / 2, num=np.size(data), endpoint=False)

    # Get rms noise
    noiserms = np.std(data)
    noiserms = noiserms * 1e3  # uV
    print('rms noise: %0.1f uV @ %0.1f kHz' % (noiserms, bw))
    johnson = np.sqrt(2 * 50 * 300 * bw * 1e3 * 1.38e-23) * 10 ** (hw.lnaGain / 20) * 1e6  # uV
    print('Expected by Johnson: %0.1f uV @ %0.1f kHz' % (johnson, bw * 1e-3))
    print('Noise factor: %0.1f johnson' % (noiserms / johnson))
    if noiserms / johnson > 2:
        print("WARNING: Noise is too high")

    # Plot signal versus time
    result1 = {'widget': 'curve',
               'xData': t_vector,
               'yData': [np.abs(data), np.real(data), np.imag(data)],
               'xLabel': 'Time (ms)',
               'yLabel': 'Signal amplitude (mV)',
               'title': 'Noise vs time',
               'legend': ['abs', 'real', 'imag'],
               'row': 0,
               'col': 0}

    # Plot spectrum
    result2 = {'widget': 'curve',
               'xData': f_vector,
               'yData': [np.abs(spectrum)],
               'xLabel': 'Frequency (kHz)',
               'yLabel': 'Mag FFT (a.u.)',
               'title': 'Noise spectrum',
               'legend': [''],
               'row': 1,
               'col': 0}

    output = [result1, result2]

    return output

if __name__ == '__main__':
    run_recon("../experiments/acquisitions/Example/2025.05.29.16.42/Example/None/mat/Noise.2025.05.29.16.42.11.897.mat",
              mode="Standalone")