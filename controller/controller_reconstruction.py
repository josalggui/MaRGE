import os
import sys
import time
import threading
import numpy as np
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget
import configs.hw_config as hw
from widgets.widget_reconstruction import ReconstructionTabWidget
import scipy.io as sio
try:
    import cupy as cp
    print("GPU will be used for ART reconstruction")
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
        self.tyger_ifft_button.clicked.connect(self.tyger_ifft_clicked)
        self.tyger_art_button.clicked.connect(self.tyger_art_clicked)
        self.tyger_artpk_button.clicked.connect(self.tyger_artpk_clicked)
        self.tyger_ifft_bart_button.clicked.connect(self.tyger_ifft_bart_clicked)
        self.tyger_ifftus_bart_button.clicked.connect(self.tyger_ifftus_bart_clicked)
        self.tyger_cs_bart_button.clicked.connect(self.tyger_cs_bart_clicked)
        self.tyger_pix2pix_button.clicked.connect(self.tyger_pix2pix_clicked)

    def tyger_pix2pix_clicked(self):
        print("Running Pix2Pix in tyger...")
        process = subprocess.run([hw.bash_path, "--", "./Tyger/control_buttons/pix2pix_python.sh", self.main.file_path])

        rawData = sio.loadmat(self.main.file_path)
        imgTyger = rawData['imgReconTyger2D_Red'][0]
        self.main.image_view_widget.main_matrix = imgTyger
        figure = imgTyger[0] / np.max(np.abs(imgTyger)) * 100
        figure = np.reshape(figure, (1,figure.shape[0],figure.shape[1]))
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
        figure = np.transpose(figure, (0,2,1))
        figure = np.flip(figure, 2)
        figure = np.flip(figure, 1)
        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="Pix2Pix net Tyger",
                                          image=figure,
                                          orientation=orientation,
                                          operation="Pix2Pix net Tyger",
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)
        
        rawData = sio.loadmat(self.main.file_path)
        img3T = rawData['highfield_img'][18]
        self.main.image_view_widget.main_matrix = imgTyger
        figure = img3T/ np.max(np.abs(imgTyger)) * 100
        figure = np.reshape(figure, (1,figure.shape[0],figure.shape[1]))
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
        figure = np.transpose(figure, (0,2,1))
        figure = np.flip(figure, 2)
        figure = np.flip(figure, 1)
        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="High field ref Img",
                                          image=figure,
                                          orientation=orientation,
                                          operation="High field ref Img",
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)
        print("Done!")
    def tyger_art_clicked(self):
        print("Running python ART in tyger...")
        process = subprocess.run([hw.bash_path, "--", "./Tyger/control_buttons/art_python.sh", self.main.file_path])

        rawData = sio.loadmat(self.main.file_path)
        imgTyger = rawData['imgReconTyger2D'][0]
        self.main.image_view_widget.main_matrix = imgTyger
        figure = imgTyger[0] / np.max(np.abs(imgTyger)) * 100
        figure = np.reshape(figure, (1,figure.shape[0],figure.shape[1]))
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="Python ART Tyger",
                                          image=figure,
                                          orientation=orientation,
                                          operation="Python ART Tyger",
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)
    
    def tyger_artpk_clicked(self):
        print("Running python ART PK in tyger...")
        process = subprocess.run([hw.bash_path, "--", "./Tyger/control_buttons/artPK_python.sh", self.main.file_path])

        rawData = sio.loadmat(self.main.file_path)
        imgTyger = rawData['imgReconTyger2D'][0]
        self.main.image_view_widget.main_matrix = imgTyger
        figure = imgTyger[0] / np.max(np.abs(imgTyger)) * 100
        figure = np.reshape(figure, (1,figure.shape[0],figure.shape[1]))
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="Python ART PK Tyger",
                                          image=figure,
                                          orientation=orientation,
                                          operation="Python ART PK Tyger",
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)

    def tyger_ifft_clicked(self):
        print("Running python iFFT in tyger...")
        subprocess.call([hw.bash_path, "--", "./Tyger/control_buttons/fft_python.sh", self.main.file_path])

        rawData = sio.loadmat(self.main.file_path)
        imgTyger = rawData['imgReconTyger2D'][0]
        self.main.image_view_widget.main_matrix = imgTyger
        figure = imgTyger[0] / np.max(np.abs(imgTyger)) * 100
        figure = np.reshape(figure, (1,figure.shape[0],figure.shape[1]))
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="Python iFFT Tyger",
                                          image=figure,
                                          orientation=orientation,
                                          operation="Python iFFT Tyger",
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)

    def tyger_ifft_bart_clicked(self):
        print("Running BART iFFT in tyger...")
        process = subprocess.run([hw.bash_path, "--", "./Tyger/control_buttons/bart_fft.sh", self.main.file_path])

        rawData = sio.loadmat(self.main.file_path)
        imgTyger = rawData['imgReconTyger2D'][0]
        self.main.image_view_widget.main_matrix = imgTyger
        figure = imgTyger[0] / np.max(np.abs(imgTyger)) * 100
        figure = np.reshape(figure, (1,figure.shape[0],figure.shape[1]))
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
        figure = np.transpose(figure, (0,2,1))
        figure = np.flip(figure, 2)
        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="BART iFFT Tyger",
                                          image=figure,
                                          orientation=orientation,
                                          operation="BART iFFT Tyger",
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)
        
    def tyger_ifftus_bart_clicked(self):
        print("Running BART iFFT us in tyger...")
        subprocess.run([hw.bash_path, "--", "./Tyger/control_buttons/bart_fftus.sh", self.main.file_path])

        rawData = sio.loadmat(self.main.file_path)
        imgTyger = rawData['imgReconTyger2D'][0]
        self.main.image_view_widget.main_matrix = imgTyger
        figure = imgTyger[0] / np.max(np.abs(imgTyger)) * 100
        figure = np.reshape(figure, (1,figure.shape[0],figure.shape[1]))
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
        figure = np.transpose(figure, (0,2,1))
        figure = np.flip(figure, 2)
        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="BART iFFT us  Tyger",
                                          image=figure,
                                          orientation=orientation,
                                          operation="BART iFFT us Tyger",
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)
    def tyger_cs_bart_clicked(self):
        print("Running BART CS in tyger...")
        subprocess.run([hw.bash_path, "--", "./Tyger/control_buttons/bart_cs.sh", self.main.file_path])

        rawData = sio.loadmat(self.main.file_path)
        imgTyger = rawData['imgReconTyger2D'][0]
        self.main.image_view_widget.main_matrix = imgTyger
        figure = imgTyger[0] / np.max(np.abs(imgTyger)) * 100
        figure = np.reshape(figure, (1,figure.shape[0],figure.shape[1]))
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
        figure = np.transpose(figure, (0,2,1))
        figure = np.flip(figure, 2)
        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="BART CS Tyger",
                                          image=figure,
                                          orientation=orientation,
                                          operation="BART CS Tyger",
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)
    
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
        
        orientation=None
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
        # if 'cp' in globals():
        #     print('Executing ART in GPU...')

        #     # Transfer numpy arrays to cupy arrays
        #     kx_gpu = cp.asarray(kx)
        #     ky_gpu = cp.asarray(ky)
        #     kz_gpu = cp.asarray(kz)
        #     x_gpu = cp.asarray(x)
        #     y_gpu = cp.asarray(y)
        #     z_gpu = cp.asarray(z)
        #     s_gpu = cp.asarray(s)
        #     index = cp.asarray(index)
        #     rho_gpu = cp.asarray(rho)

        #     # Execute ART
        #     rho_gpu = iterative_process_gpu(kx_gpu, ky_gpu, kz_gpu, x_gpu, y_gpu, z_gpu, s_gpu, rho_gpu, lbda, n_iter,
        #                                     index)
        #     rho = cp.asnumpy(rho_gpu)
        # else:
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
        """
        Perform POCS reconstruction.

        Retrieves the number of points before m+n where reconstruction begins to go to zero.
        Retrieves the correlation threshold for stopping the iterations.
        Computes the partial image and full image for POCS reconstruction.
        Applies the iterative reconstruction with phase correction.
        Updates the main matrix of the image view widget with the interpolated image.
        Adds the "POCS" operation to the history widget and updates the history dictionary and operations history.
        """

        def getCenterKSpace(k_space, n, m_vec):
            # fix n_vec
            output = np.zeros(np.shape(k_space), dtype=complex)
            n_vec = np.array(np.shape(k_space))

            # fill with zeros
            idx0 = n_vec // 2 - m_vec
            idx1 = n_vec // 2 + m_vec
            output[idx0[0]:idx1[0], idx0[1]:idx1[1], idx0[2]:idx1[2]] = \
                k_space[idx0[0]:idx1[0], idx0[1]:idx1[1], idx0[2]:idx1[2]]

            return output
            # # fix n_vec
            # output = k_space.copy()
            # n_vec = np.array([0, 0, 0])
            # for ii in range(3):
            #     if m_vec[ii] == n[ii]:
            #         n_vec[ii] = np.size(k_space, ii)
            #     else:
            #         n_vec[ii] = n[ii]
            #
            # # fill with zeros
            # output[0:n_vec[0] - m_vec[0], 0:n_vec[1] - m_vec[1], 0:n_vec[2] - m_vec[2]] = 0.0
            # output[n_vec[0] + m_vec[0]::, n_vec[1] + m_vec[1]::, n_vec[2] + m_vec[2]::] = 0.0
            #
            # return output

        mat_data = self.main.toolbar_image.mat_data
        nPoints = mat_data['nPoints'][0][-1::-1]

        # Number of extra lines which has been taken past the center of k-space
        factors = self.partial_reconstruction_factor.text().split(',')
        factors = [float(num) for num in factors][-1::-1]
        mm = np.array([int(num) for num in (nPoints * factors)])

        n = np.array([int(num) for num in (nPoints / 2.0)])  # Divide the data per 2
        m = np.array([int(num) for num in (nPoints * factors - nPoints / 2)])


        # Get the k_space data
        kSpace_ref = self.main.image_view_widget.main_matrix.copy()
        img_ref = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace_ref))))

        # Create a copy with the center of k-space
        kSpace_center = getCenterKSpace(kSpace_ref, n, m)

        # Number of points before m+n where we begin to go to zero
        nb_point = int(self.nb_points_text_field.text())

        # Set the correlation threshold for stopping the iterations
        threshold = float(self.threshold_text_field.text())

        # Get image phase
        img_center = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace_center)))
        phase = img_center / abs(img_center)

        # Generate the corresponding image with the Hanning filter
        kSpace_hanning = hanningFilter(kSpace_ref, mm, nb_point)
        img_hanning = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace_hanning))))

        num_iterations = 0  # Initialize the iteration counter
        previous_img = img_hanning.copy()  # you have the choice between img_hanning or img_ramp

        while True:
            # Iterative reconstruction
            img_iterative = previous_img * phase
            kSpace_new = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(img_iterative)))

            # Apply constraint: Keep the region of k-space from n+m onwards and restore the rest
            kSpace_new[0:mm[0], 0:mm[1], 0:mm[2]] = kSpace_ref[0:mm[0], 0:mm[1], 0:mm[2]]

            # Reconstruct the image from the modified k-space
            img_reconstructed = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace_new))))

            # Compute correlation between consecutive reconstructed images
            correlation = np.corrcoef(previous_img.flatten(), img_reconstructed.flatten())[0, 1]

            # Display correlation and current iteration number
            print("Iteration: %i, Convergence: %0.2e" % (num_iterations, (1-correlation)))

            # Check if correlation reaches the desired threshold
            if (1-correlation) <= threshold or num_iterations >= 100:
                break

            # Update previous_img for the next iteration
            previous_img = img_reconstructed.copy()

            # Increment the iteration counter
            num_iterations += 1

        # Update the main matrix of the image view widget with the interpolated image
        self.main.image_view_widget.main_matrix = img_reconstructed

        figure = img_reconstructed / np.max(np.abs(img_reconstructed)) * 100

        # Get correlation with reference image
        correlation = np.corrcoef(img_ref.flatten(), img_reconstructed.flatten())[0, 1]
        print("Respect the reference image:")
        print("Convergence: %0.2e" % (1 - correlation))
        orientation=None
        if self.main.toolbar_image.mat_data and 'axesOrientation' in self.main.toolbar_image.mat_data:
            orientation = self.main.toolbar_image.mat_data['axesOrientation'][0]
        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="POCS",
                                          image=figure,
                                          orientation=orientation,
                                          operation="POCS - " + str(factors[-1::-1]),
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ReconstructionTabController(parent=QWidget)
    window.main.file_path = "D:/CSIC/REPOSITORIOS/MaRGE/RARE.2024.09.25.11.07.51.836.mat"
    window.show()
    sys.exit(app.exec_())
