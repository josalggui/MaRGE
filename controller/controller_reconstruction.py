import os
import time
import threading
import numpy as np
# import matlab.engine
from widgets.widget_reconstruction import ReconstructionTabWidget
import cupy as cp


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


def hanningFilter(kSpace, n, m, nb_point):
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
    kSpace_hanning[:, :, n + m::] = 0.0

    # Calculate the Hanning window
    hanning_window = np.hanning(nb_point * 2)

    # Apply the Hanning filter to the k-space
    start_point = n + m - nb_point
    end_point = n + m

    for i in range(start_point, end_point):
        window_index = i - (n + m - nb_point)
        kSpace_hanning[:, :, i] *= hanning_window[window_index]

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

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="dFFT",
                                          image=self.main.image_view_widget.main_matrix,
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

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="iFFT",
                                          image=self.main.image_view_widget.main_matrix,
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

        self.main.console.print('Executing ART in GPU...')

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
                        self.main.console.print("ART iteration %i: %i %%" % (iteration + 1, m))

            return rho

        # Transfer numpy arrays to cupy arrays
        kx_gpu = cp.asarray(kx)
        ky_gpu = cp.asarray(ky)
        kz_gpu = cp.asarray(kz)
        x_gpu = cp.asarray(x)
        y_gpu = cp.asarray(y)
        z_gpu = cp.asarray(z)
        s_gpu = cp.asarray(s)
        index = cp.asarray(index)

        # Launch the GPU function
        rho = np.reshape(np.zeros((nPoints[0] * nPoints[1] * nPoints[2]), dtype=complex), (-1, 1))
        rho_gpu = cp.asarray(rho)
        start = time.time()
        # for iteration in range(n_iter):
        rho_gpu = iterative_process_gpu(kx_gpu, ky_gpu, kz_gpu, x_gpu, y_gpu, z_gpu, s_gpu, rho_gpu, lbda, n_iter,
                                        index)
        end = time.time()
        rho = cp.asnumpy(rho_gpu)
        self.main.console.print("Reconstruction time = %0.1f s" % (end - start))

        rho = np.reshape(rho, nPoints[-1::-1])

        # Update the main matrix of the image view widget with the image fft data
        self.main.image_view_widget.main_matrix = rho

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="ART",
                                          image=self.main.image_view_widget.main_matrix,
                                          operation="ART n = %i, lambda = %0.3f" % (n_iter, lbda),
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)

        return

    # eng = matlab.engine.start_matlab()
    #
    # eng.workspace['fov'] = matlab.double(fov.tolist(), is_complex=True)
    # eng.workspace['nPoints'] = matlab.double(nPoints.tolist(), is_complex=True)
    # eng.workspace['sampled'] = matlab.double(self.sampled.tolist(), is_complex=True)
    # eng.workspace['s'] = matlab.double(s.tolist(), is_complex=True)
    # eng.workspace['niter'] = matlab.double(niter, is_complex=True)
    # eng.workspace['lbda'] = matlab.double(lbda, is_complex=True)
    # eng.workspace['rho'] = matlab.double(rho.tolist(), is_complex=True)
    #
    # start_time = time.time()
    #
    # matlab_script_path = getPath()
    #
    # # Run the MATLAB script
    # eng.run(matlab_script_path, nargout=0)
    #
    # rho = eng.workspace['rho']
    #
    # # Close MATLAB engine
    # eng.quit()
    #
    # rho = np.array(rho)
    #
    # print('ART has been applied')
    #
    # # Update the main matrix of the image view widget with the cosbell data
    # self.main.image_view_widget.main_matrix = rho
    #
    # # Add the "Cosbell" operation to the history widget
    # self.main.history_list.addItemWithTimestamp("ART")
    #
    # # Update the history dictionary with the new main matrix for the current matrix info
    # self.main.history_list.image_hist[self.main.history_list.image_key] = \
    #     self.main.image_view_widget.main_matrix
    #
    # # Update the operations history
    # self.main.history_list.operations_hist[self.main.history_list.image_key] = ["ART"]
    #
    # # Update the space dictionary
    # self.main.history_list.space[self.main.history_list.image_key] = 'i'
    #
    # # Get the end time
    # end_time = time.time()
    #
    # # Calculate the elapsed time in seconds
    # elapsed_time = end_time - start_time
    #
    # # Calculate the time components
    # hours = int(elapsed_time // 3600)
    # minutes = int((elapsed_time % 3600) // 60)
    # seconds = int(elapsed_time % 60)
    #
    # print(f"Time : {hours} hours, {minutes} minutes, {seconds} seconds")

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
        # Number of points before m+n where we begin to go to zero
        nb_point = int(self.nb_points_text_field.text())

        # Set the correlation threshold for stopping the iterations
        correlation_threshold = int(self.threshold_text_field.text())

        m = self.main.preprocessing_controller.m
        n = self.main.preprocessing_controller.n
        kSpace_center = self.main.preprocessing_controller.kSpace_center
        kSpace_ref = self.main.preprocessing_controller.kSpace_ref
        img_ref = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(10 ** self.main.image_view_widget.main_matrix)))

        kSpace_partial = np.copy(kSpace_ref)
        kSpace_partial[:, :, n + m::] = 0.0
        img_partial = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace_partial))))

        img_full = np.concatenate((np.abs(img_ref), img_partial), axis=2)

        img_center = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace_center)))
        phase = img_center / abs(img_center)

        # Generate the corresponding image with a ramp filter
        kSpace_ramp = ramp(kSpace_ref, n, m, nb_point)
        img_ramp = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace_ramp))))

        # Generate the corresponding image with the Hanning filter
        kSpace_hanning = hanningFilter(kSpace_ref, n, m, nb_point)
        img_hanning = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace_hanning))))

        num_iterations = 0  # Initialize the iteration counter
        previous_img = img_hanning  # you have the choice between img_hanning or img_ramp

        while True:
            # Iterative reconstruction
            img_iterative = previous_img * phase
            kSpace_new = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(img_iterative)))

            # Apply constraint: Keep the region of k-space from n+m onwards and restore the rest
            kSpace_new[:, :, 0:n + m] = kSpace_ref[:, :, 0:n + m]

            # Reconstruct the image from the modified k-space
            img_reconstructed = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kSpace_new))))

            # Compute correlation between consecutive reconstructed images
            correlation = np.corrcoef(previous_img.flatten(), img_reconstructed.flatten())[0, 1]

            # Display correlation and current iteration number
            print(f"Iteration: {num_iterations}, Correlation: {correlation}")

            # Check if correlation reaches the desired threshold
            if correlation >= correlation_threshold:
                break

            # Update previous_img for the next iteration
            previous_img = img_reconstructed.copy()

            # Increment the iteration counter
            num_iterations += 1

        # Display the final reconstructed image
        img_reconstructed = np.abs(img_reconstructed)

        img_full = np.concatenate((img_full, img_reconstructed), axis=2)

        # Update the main matrix of the image view widget with the interpolated image
        self.main.image_view_widget.main_matrix = img_full

        # Add new item to the history list
        self.main.history_list.addNewItem(stamp="POCS",
                                          image=self.main.image_view_widget.main_matrix,
                                          operation="POCS",
                                          space="i",
                                          image_key=self.main.image_view_widget.image_key)
