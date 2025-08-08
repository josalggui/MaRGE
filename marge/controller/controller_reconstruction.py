import os
import time
import threading
import numpy as np
from marge.widgets.widget_reconstruction import ReconstructionTabWidget
from marge.marge_utils import utils

try:
    import cupy as cp
except ImportError:
    pass


def getPath():
    """
    Get the absolute path to the MATLAB script file.

    Returns:
        str: Absolute path to the MATLAB script file.
    """
    # Get the absolute path of the current Python script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Relative path to the "scripts" directory in your project
    scripts_dir = os.path.join(current_dir, '..', 'scripts')

    # Absolute path to the "art.m" file
    matlab_script_path = os.path.join(scripts_dir, 'art.m')

    return matlab_script_path


def ramp(kSpace, n, m, nb_point):
    """
    Apply a ramp filter to the k-space data.

    Args:
        kSpace (ndarray): K-space data.
        n (int): Number of zero-filled points in k-space.
        m (int): Number of acquired points in k-space.
        nb_point (int): Number of points before m+n where reconstruction begins to go to zero.

    Returns:
        ndarray: K-space data with the ramp filter applied.
    """
    kSpace_ramp = np.copy(kSpace)
    kSpace_ramp[:, :, n + m::] = 0.0

    # Index of the gradient
    start_point = n + m - nb_point
    end_point = n + m
    gradient_factor = kSpace_ramp[:, :, n + m - nb_point] / nb_point

    # Go progressively to 0
    for i in range(start_point, end_point):
        kSpace_ramp[:, :, i + 1] *= kSpace_ramp[:, :, i] - gradient_factor

    return kSpace_ramp


def hanningFilter(kSpace, mm, nb_point):
    """
    Apply a Hanning filter to the k-space data.

    Args:
        kSpace (ndarray): K-space data.
        n (int): Number of zero-filled points in k-space.
        m (int): Number of acquired points in k-space.
        nb_point (int): Number of points before m+n where reconstruction begins to go to zero.

    Returns:
        ndarray: K-space data with the Hanning filter applied.
    """
    kSpace_hanning = np.copy(kSpace)
    kSpace_hanning[mm[0]::, :, :] = 0.0
    kSpace_hanning[:, mm[1]::, :] = 0.0
    kSpace_hanning[:, :, mm[2]::] = 0.0

    # Calculate the Hanning window
    hanning_window = np.hanning(nb_point * 2)
    hanning_window = hanning_window[int(len(hanning_window)/2)::]

    if not mm[0] == np.size(kSpace, 0):
        for ii in range(nb_point):
            kSpace_hanning[mm[0]-nb_point+ii+1, :, :] *= hanning_window[ii]
    if not mm[1] == np.size(kSpace, 1):
        for ii in range(nb_point):
            kSpace_hanning[:, mm[1]-nb_point+ii+1, :] *= hanning_window[ii]
    if not mm[2] == np.size(kSpace, 2):
        for ii in range(nb_point):
            kSpace_hanning[:, :, mm[2]-nb_point+ii+1] *= hanning_window[ii]

    return kSpace_hanning


class ReconstructionTabController(ReconstructionTabWidget):
    """
    Controller class for the ReconstructionTabWidget.

    Inherits from ReconstructionTabWidget to provide additional functionality for image reconstruction.

    Attributes:
        pocs_button: QPushButton for performing POCS.
        image_fft_button: QPushButton for performing FFT reconstruction.
        image_art_button: QPushButton for performing ART reconstruction.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the ReconstructionTabController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(ReconstructionTabController, self).__init__(*args, **kwargs)

        # Connect the image_fft_button clicked signal to the fftReconstruction method
        self.pocs_button.clicked.connect(self.pocsReconstruction)
        self.zero_button.clicked.connect(self.zeroReconstruction)
        self.ifft_button.clicked.connect(self.ifft)
        self.dfft_button.clicked.connect(self.dfft)
        self.image_art_button.clicked.connect(self.artReconstruction)

    def dfft(self):

        thread = threading.Thread(target=self.runDFFT)
        thread.start()

    def runDFFT(self):
        """
        """
        # Get the k-space data from the main matrix
        image = self.main.image_view_widget.main_matrix

        # Perform direct FFT shift, inverse FFT, and inverse FFT shift to reconstruct the image in the spatial domain
        k_space = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(image)))

        # Update the main matrix of the image view widget with the image fft data
        self.main.image_view_widget.main_matrix = k_space
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="dFFT",
                                          image=self.main.image_view_widget.main_matrix,
                                          orientation=orientation,
                                          operation="dFFT",
                                          space="k",
                                          image_key=self.main.image_view_widget.image_key)

    def ifft(self):
        """
        Perform FFT reconstruction in a separate thread.

        Creates a new thread and runs the runFftReconstruction method in that thread.
        """

        thread = threading.Thread(target=self.runIFFT)
        thread.start()

    def runIFFT(self):
        """
        Perform FFT reconstruction.

        Retrieves the k-space data from the main matrix of the image view widget.
        Performs inverse FFT shift, inverse FFT, and inverse FFT shift to reconstruct the image in the spatial domain.
        Updates the main matrix of the image view widget with the reconstructed image.
        Adds the "FFT" operation to the history widget and updates the history dictionary and operations history.
        """
        # Get the k-space data from the main matrix
        k_space = self.main.image_view_widget.main_matrix

        # Perform inverse FFT shift, inverse FFT, and inverse FFT shift to reconstruct the image in the spatial domain
        image = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(k_space)))

        # Update the main matrix of the image view widget with the image fft data
        self.main.image_view_widget.main_matrix = image

        figure = image / np.max(np.abs(image)) * 100
        
        orientation = None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="iFFT",
                                          image=figure,
                                          orientation=orientation,
                                          operation="iFFT",
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)

    def artReconstruction(self):
        """
        Perform ART reconstruction in a separate thread.

        Creates a new thread and runs the runArtReconstruction method in that thread.
        """

        thread = threading.Thread(target=self.runArtReconstruction)
        thread.start()

    def runArtReconstruction(self):
        """
        Perform ART reconstruction.

        Retrieves the mat data from the loaded .mat file in the main toolbar controller.
        Extracts the necessary data from the mat file.
        Performs the ART reconstruction algorithm.
        Updates the main matrix of the image view widget with the reconstructed image.
        Adds the "ART" operation to the history widget and updates the history dictionary and operations history.
        """
        # Get the mat data from the loaded .mat file in the main toolbar controller
        mat_data = self.main.toolbar_image.mat_data

        # Extract datas data from the loaded .mat file
        sampled = self.main.toolbar_image.k_space_raw
        fov = np.reshape(mat_data['fov'], -1) * 1e-2
        nPoints = np.reshape(mat_data['nPoints'], -1)
        k = sampled[:, 0:3]
        s = sampled[:, 3]

        # Points where rho will be estimated
        x = np.linspace(-fov[0] / 2, fov[0] / 2, nPoints[0])
        y = np.linspace(-fov[1] / 2, fov[1] / 2, nPoints[1])
        z = np.linspace(-fov[2] / 2, fov[2] / 2, nPoints[2])
        y, z, x = np.meshgrid(y, z, x)
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        z = np.reshape(z, (-1, 1))

        # k-points
        kx = np.reshape(k[:, 0], (-1, 1))
        ky = np.reshape(k[:, 1], (-1, 1))
        kz = np.reshape(k[:, 2], (-1, 1))

        # Iterative process
        lbda = float(self.lambda_text_field.text())
        n_iter = int(self.niter_text_field.text())
        index = np.arange(len(s))

        def iterative_process_gpu(kx, ky, kz, x, y, z, s, rho, lbda, n_iter, index):
            n = 0
            n_samples = len(s)
            m = 0
            for iteration in range(n_iter):
                cp.random.shuffle(index)
                for jj in range(n_samples):
                    ii = index[jj]
                    x0 = cp.exp(-1j * 2 * cp.pi * (kx[ii] * x + ky[ii] * y + kz[ii] * z))
                    x1 = (x0.T @ rho) - s[ii]
                    x2 = x1 * cp.conj(x0) / (cp.conj(x0.T) @ x0)
                    d_rho = lbda * x2
                    rho -= d_rho
                    n += 1
                    if n / n_samples > 0.01:
                        m += 1
                        n = 0
                        print("ART iteration %i: %i %%" % (iteration + 1, m))

            return rho

        def iterative_process_cpu(kx, ky, kz, x, y, z, s, rho, lbda, n_iter, index):
            n = 0
            n_samples = len(s)
            m = 0
            for iteration in range(n_iter):
                np.random.shuffle(index)
                for jj in range(n_samples):
                    ii = index[jj]
                    x0 = np.exp(-1j * 2 * np.pi * (kx[ii] * x + ky[ii] * y + kz[ii] * z))
                    x1 = (x0.T @ rho) - s[ii]
                    x2 = x1 * np.conj(x0) / (np.conj(x0.T) @ x0)
                    d_rho = lbda * x2
                    rho -= d_rho
                    n += 1
                    if n / n_samples > 0.01:
                        m += 1
                        n = 0
                        print("ART iteration %i: %i %%" % (iteration + 1, m))

            return rho

        # Launch the GPU function
        rho = np.reshape(np.zeros((nPoints[0] * nPoints[1] * nPoints[2]), dtype=complex), (-1, 1))
        start = time.time()

        gpu_ready = False
        if 'cp' in globals():
            try:
                print('Executing ART in GPU...')

                # Transfer numpy arrays to cupy arrays
                kx_gpu = cp.asarray(kx)
                ky_gpu = cp.asarray(ky)
                kz_gpu = cp.asarray(kz)
                x_gpu = cp.asarray(x)
                y_gpu = cp.asarray(y)
                z_gpu = cp.asarray(z)
                s_gpu = cp.asarray(s)
                index = cp.asarray(index)
                rho_gpu = cp.asarray(rho)

                # Execute ART
                rho_gpu = iterative_process_gpu(kx_gpu, ky_gpu, kz_gpu, x_gpu, y_gpu, z_gpu, s_gpu, rho_gpu, lbda, n_iter,
                                                index)
                rho = cp.asnumpy(rho_gpu)
                gpu_ready = True
            except:
                print("GPU not available...")
        if not gpu_ready:
            print('Executing ART in CPU...')

            rho = iterative_process_cpu(kx, ky, kz, x, y, z, s, rho, lbda, n_iter,
                                            index)
        end = time.time()
        print("Reconstruction time = %0.1f s" % (end - start))

        rho = np.reshape(rho, nPoints[-1::-1])

        # Update the main matrix of the image view widget with the image fft data
        self.main.image_view_widget.main_matrix = rho

        figure = rho/np.max(np.abs(rho))*100
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="ART",
                                          image=figure,
                                          orientation=orientation,
                                          operation="ART n = %i, lambda = %0.3f" % (n_iter, lbda),
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)

        return

    def zeroReconstruction(self):

        thread = threading.Thread(target=self.runZeroReconstruction)
        thread.start()

    def runZeroReconstruction(self):
        """
        Run the partial reconstruction operation.

        Retrieves the necessary parameters and performs the partial reconstruction on the loaded image.
        Updates the main matrix of the image view widget with the partially reconstructed image, adds the operation to
        the history widget, and updates the operations history.
        """
        # Get the k_space data and its shape
        k_space = self.main.image_view_widget.main_matrix.copy()
        img_ref = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(k_space))))
        nPoints = self.main.toolbar_image.nPoints[-1::-1]

        # Percentage for partial reconstruction from the text field
        factors = self.partial_reconstruction_factor.text().split(',')
        factors = [float(num) for num in factors][-1::-1]
        mm = np.array([round(num) for num in (nPoints * factors)])

        # Set to zero the corresponding values
        k_space[mm[0]::, :, :] = 0.0
        k_space[:, mm[1]::, :] = 0.0
        k_space[:, :, mm[2]::] = 0.0

        # Calculate logarithmic scale
        image = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(k_space))))

        # Get correlation with reference image
        correlation = np.corrcoef(img_ref.flatten(), image.flatten())[0, 1]
        print("Respect the reference image:")
        print("Convergence: %0.2e" % (1 - correlation))

        # Update the main matrix of the image view widget with the k-space data
        self.main.image_view_widget.main_matrix = image
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="Partial Zero Reconstruction",
                                          image=image,
                                          orientation=orientation,
                                          operation="Partial Reconstruction - " + str(factors[-1::-1]),
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)

    def pocsReconstruction(self):
        """
        Perform POCS reconstruction in a separate thread.

        Creates a new thread and runs the runPocsReconstruction method in that thread.
        """

        thread = threading.Thread(target=self.runPocsReconstruction)
        thread.start()

    def runPocsReconstruction(self):
        # Get n_points
        mat_data = self.main.toolbar_image.mat_data
        n_points = mat_data['nPoints'][0][-1::-1]

        # Number of extra lines which has been taken past the center of k-space
        factors = self.partial_reconstruction_factor.text().split(',')
        factors = [float(num) for num in factors][-1::-1]

        # Get the k_space data
        k_space_ref = self.main.image_view_widget.main_matrix.copy()

        # Run pocs
        img_reconstructed = utils.run_pocs_reconstruction(n_points, factors, k_space_ref)

        # Update the main matrix of the image view widget with the interpolated image
        self.main.image_view_widget.main_matrix = img_reconstructed

        figure = img_reconstructed / np.max(np.abs(img_reconstructed)) * 100

        # Add new item to the history list
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
        self.main.history_list.addNewItem(stamp="POCS",
                                          image=figure,
                                          orientation=orientation,
                                          operation="POCS - " + str(factors[-1::-1]),
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)
