import copy

import bm4d
import numpy as np
import nibabel as nib
from skimage.util import view_as_blocks
import scipy as sp
from marge.manager.dicommanager import DICOMImage
import pydicom
from pydicom.dataset import Dataset, FileDataset
from datetime import date, datetime
from marge.configs import hw_config as hw


def fix_image_orientation(image, axes, orientation='FFS', rd_direction=1):
    """
    Adjusts the orientation of a 3D image array to match standard anatomical planes
    (sagittal, coronal, or transversal) and returns the oriented image along with labeling
    and metadata for visualization.

    Args:
        image (np.ndarray): A 3D numpy array representing the image data to be reoriented.
        axes (list[int]): A list of three integers representing the current order of the
                          axes in the image (e.g., [0, 1, 2] for x, y, z).

    Returns:
        output (dict): A dictionary containing the following keys:
            - 'widget': A fixed string "image" or "curve" indicating the type of data for visualization.
            - 'data': The reoriented 3D image array (np.ndarray).
            - 'xLabel': A string representing the label for the x-axis in the visualization.
            - 'yLabel': A string representing the label for the y-axis in the visualization.
            - 'title': A string representing the title of the visualization (e.g., "Sagittal").
        image (np.ndarray): Reoriented 3D image array
        dicom_orientation (list): orientation for dicom file
    """

    # Get axes in strings
    axes_dict = {'x': 0, 'y': 1, 'z': 2}
    axes_keys = list(axes_dict.keys())
    axes_vals = list(axes_dict.values())
    axes_str = ['', '', '']
    n = 0
    for val in axes:
        index = axes_vals.index(val)
        axes_str[n] = axes_keys[index]
        n += 1

    # Create output dictionaries to plot figures
    x_label = "%s axis" % axes_str[1]
    y_label = "%s axis" % axes_str[0]
    title = "Image"
    if axes[2] == 2:  # Sagittal
        title = "Sagittal"
        if axes[0] == 0 and axes[1] == 1:
            image = np.flip(image, axis=0)
            x_label = "(-Y) A | PHASE | P (+Y)"
            y_label = "(+X) I | READOUT | S (-X)"
            if rd_direction == -1:
                image = image[:, :, ::-1]
        else:
            image = np.transpose(image, (0, 2, 1))
            image = np.flip(image, axis=0)
            x_label = "(-Y) A | READOUT | P (+Y)"
            y_label = "(+X) I | PHASE | S (-X)"
            if rd_direction == -1:
                image = image[:, ::-1, :]
        image_orientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
    elif axes[2] == 1:  # Coronal
        title = "Coronal"
        if axes[0] == 0 and axes[1] == 2:
            x_label = "(-Z) R | PHASE | L (+Z)"
            y_label = "(+X) I | READOUT | S (-X)"
            if rd_direction == -1:
                image = image[:, :, ::-1]
        else:
            image = np.transpose(image, (0, 2, 1))
            x_label = "(-Z) R | READOUT | L (+Z)"
            y_label = "(+X) I | PHASE | S (-X)"
            if rd_direction == -1:
                image = image[:, ::-1, :]
        image_orientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    elif axes[2] == 0:  # Transversal
        title = "Transversal"
        if axes[0] == 1 and axes[1] == 2:
            image = np.flip(image, axis=0)
            x_label = "(-Z) R | PHASE | L (+Z)"
            y_label = "(+Y) P | READOUT | A (-Y)"
            if rd_direction == -1:
                image = image[:, :, ::-1]
        else:
            image = np.transpose(image, (0, 2, 1))
            image = np.flip(image, axis=0)
            x_label = "(-Z) R | READOUT | L (+Z)"
            y_label = "(+Y) P | PHASE | A (-Y)"
            if rd_direction == -1:
                image = image[:, ::-1, :]
        image_orientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    output = {
        'widget': 'image',
        'data': image,
        'xLabel': x_label,
        'yLabel': y_label,
        'title': title,
    }

    return output, image, image_orientation_dicom

def save_dicom(axes_orientation, n_points, fov, image, file_path, meta_data=None, session=None):
    if session is None:
        session = {}
    if meta_data is None:
        meta_data = {}
    image = np.abs(image)
    axes_orientation = np.array(axes_orientation)
    n_xyz = [0, 0, 0]
    reorder = [0, 0, 0]

    # Determine correct dimension ordering
    for i, axis in enumerate(axes_orientation):
        n_xyz[axis] = n_points[i]
        reorder[axis] = i

    fov = np.array(fov)  # cm
    fov = fov[axes_orientation]
    resolution = fov / n_points * 10  # mm

    """
    ## FUNCTIONS
    study_accession_map = {}
    scanner_prefix = 1111   # Los 4 primeros digitos son fijo
    def generate_accession_number(study_id):
        if study_id not in study_accession_map:
            random_suffix = str(random.randint(0,10**12-1)).zfill(12)
            accession = scanner_prefix + random_suffix
            study_accession_map[study_id] = accession
        return study_accession_map[study_id]
    """

    ## DICOM TAGS
    # Orientation tags. Look carefully
    # """Added comments"""
    if (axes_orientation == [0, 1, 2]).all():
        meta_data["ImageOrientationPatient"] = [0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        meta_data['PixelSpacing'] = [resolution[1], resolution[0]]
        meta_data['SliceThickness'] = resolution[2]
    elif (axes_orientation == [1, 0, 2]).all():
        meta_data["ImageOrientationPatient"] = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        meta_data['PixelSpacing'] = [resolution[1], resolution[0]]
        meta_data['SliceThickness'] = resolution[2]
    elif (axes_orientation == [1, 2, 0]).all():
        image = image[::-1, : ,:]
        meta_data["ImageOrientationPatient"] = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        meta_data['PixelSpacing'] = [resolution[1], resolution[0]]
        meta_data['SliceThickness'] = resolution[2]
    elif (axes_orientation == [2, 1, 0]).all():
        image = image[::-1, : ,:]
        meta_data["ImageOrientationPatient"] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        meta_data['PixelSpacing'] = [resolution[1], resolution[0]]
        meta_data['SliceThickness'] = resolution[2]
    elif (axes_orientation == [2, 0, 1]).all():
        meta_data["ImageOrientationPatient"] = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        meta_data['PixelSpacing'] = [resolution[1], resolution[0]]
        meta_data['SliceThickness'] = resolution[2]
    elif (axes_orientation == [0, 2, 1]).all():
        meta_data["ImageOrientationPatient"] = [0.0, 0.0, -1.0, 1.0, 0.0, 0.0]
        meta_data['PixelSpacing'] = [resolution[1], resolution[0]]
        meta_data['SliceThickness'] = resolution[2]

    # Normalize to 16-bit (0 - 65535)
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * (2 ** 15 - 1)
    image = image.astype(np.uint16)

    # Create metadata
    if len(image.shape) > 2:
        # Obtener dimensiones
        slices, rows, columns = image.shape
        meta_data["Columns"] = columns
        meta_data["Rows"] = rows
        meta_data["NumberOfSlices"] = slices
        meta_data["NumberOfFrames"] = slices
    # if it is a 2d image
    else:
        # Obtener dimensiones
        rows, columns = image.shape
        slices = 1
        meta_data["Columns"] = columns
        meta_data["Rows"] = rows
        meta_data["NumberOfSlices"] = 1
        meta_data["NumberOfFrames"] = 1

    meta_data["PixelData"] = image.tobytes()
    meta_data["WindowWidth"] = 26373
    meta_data["WindowCenter"] = 13194

    # Create DICOM object
    dicom_image = DICOMImage()

    # Save image into DICOM object
    dicom_image.meta_data["PixelData"] = meta_data["PixelData"]

    # General Info - STATIC TAGS  """Added all the fields below"""
    dicom_image.meta_data["Modality"] = "MR PORTABLE"
    dicom_image.meta_data["InstitutionName"] = session['institution_name']
    dicom_image.meta_data["Manufacturer"] = session['scanner_manufacturer']
    dicom_image.meta_data["ManufacturerModelName"] = session['scanner_name']
    dicom_image.meta_data["SoftwareVersions"] = f"MARGE {session['software_version']}"
    dicom_image.meta_data["ImagingFrequency"] = hw.larmorFreq
    if 'FFS' in session['orientation']:
        dicom_image.meta_data["PatientPosition"] = "FFS"

    # Sessiontags -- ALL NEW EC
    dicom_image.meta_data["PatientName"] = session["subject_id"]
    dicom_image.meta_data["StudyID"] = session["study_id"]
    dicom_image.meta_data["PatientID"] = session["subject_id"]
    dicom_image.meta_data["OperatorsName"] = session['user']

    if session['subject_birthday'] != 'YY/MM/DD':
        dicom_image.meta_data["PatientBirthDate"] =  session["subject_birthday"]
    if session['subject_weight'] != 'kg':
        dicom_image.meta_data["PatientWeight"] = session["subject_weight"]
    if session['subject_height']:
        dicom_image.meta_data["PatientSize"] = session["subject_height"]
    
    # Study tags: Static ones. -- All New EC
    """
    Study refers to all the measurement related to a particular patient. 
    - Accession number: This is an identifier number composed of 16 digits. 
    It is at study level. All the images that belong to the same study need 
    to have the same value for the accession number. In our case, it will be 
    random number generated for each study. We will make sure though that the first 
    3 digits are the same in general to indentify our specific scan, e.g. PHYSIO I.

    - Study ID: The studyID should be the same as the accession number. 
    """
    current_time = datetime.now()
    meta_data["StudyDate"] = current_time.strftime("%Y%m%d")
    meta_data["StudyTime"] = current_time.strftime("%H%M%S")

    # Series tags -- All New EC
    """
    Series refers to every single measurement perform for each patient. 
    """
    dicom_image.meta_data["SeriesDate"] = current_time.strftime("%Y%m%d")
    dicom_image.meta_data["SeriesNumber"] = session['seriesNumber']


    # Update the DICOM metadata
    dicom_image.meta_data.update(meta_data)

    # Save metadata dictionary into DICOM object metadata (Standard DICOM 3.0)
    dicom_image.image2Dicom()

    # Save DICOM file
    dicom_image.save(f"{file_path}")

def save_nifti(axes_orientation, n_points, fov, dfov, image, file_path):
    axes_orientation = np.array(axes_orientation)
    n_xyz = [0, 0, 0]
    reorder = [0, 0, 0]
    for i, axis in enumerate(axes_orientation):
        n_xyz[axis] = n_points[i]
        reorder[axis] = i
    fov = np.array(fov)  # cm
    dfov = np.array(dfov)  # mm
    resolution = fov / n_xyz * 10  # mm
    image = np.transpose(image, axes=(2, 1, 0))
    image = np.transpose(image, reorder)
    image = np.transpose(image, axes=(2, 1, 0))
    image = image[::-1, ::-1, ::-1]
    image = image / np.max(image)

    orientation = 'FFS'

    # Create afine matrix
    if orientation == 'FFS':
        affine = np.zeros((4, 4))
        affine[0, 0] = resolution[2]
        affine[1, 1] = resolution[1]
        affine[2, 2] = resolution[0]
        affine[0, 3] = dfov[2]
        affine[1, 3] = - dfov[1]
        affine[2, 3] = dfov[0]
        affine[3, 3] = 1
    elif orientation == 'HFS':
        print("Affine matrix not ready.")
    elif orientation == 'FFP':
        print("Affine matrix not ready.")
    elif orientation == 'HFP':
        print("Affine matrix not ready.")

    # Create the NIfTI image
    nifti_img = nib.Nifti1Image(np.abs(image), affine)

    # Save the NIfTI file
    nib.save(nifti_img, file_path)

def run_cosbell_filter(sampled, data, cosbell_order):
    """
    Apply the Cosbell filter operation to the k-space data along three directions.

    This method applies the Cosbell filter to the k-space data along the readout ('rd'), phase ('ph'), and slice
    ('sl') directions. It modifies the input k-space data in-place.

    Args:
        sampled (ndarray): The sampled k-space coordinates in a nx3 matrix, where n is the number of points in k-space. The three columns correspond to the readout, phase, and slice directions.
        data (ndarray): The 3D matrix representing the k-space data to be filtered (sl, ph, rd).
        cosbell_order (list): The order of the Cosbell filter for each direction (rd, ph, sl)

    Returns:
        ndarray: The filtered k-space data (sl, ph, rd).
    """
    n_points = data.shape

    # Along readout
    k = np.reshape(sampled[:, 0], n_points)
    kmax = np.max(np.abs(k[:]))
    theta = k / kmax
    data *= (np.cos(theta * (np.pi / 2)) ** cosbell_order[0])

    # Along phase
    k = np.reshape(sampled[:, 1], n_points)
    kmax = np.max(np.abs(k[:]))
    theta = k / kmax
    data *= (np.cos(theta * (np.pi / 2)) ** cosbell_order[1])

    # Along slice
    k = np.reshape(sampled[:, 2], n_points)
    kmax = np.max(np.abs(k[:]))
    theta = k / kmax
    data *= (np.cos(theta * (np.pi / 2)) ** cosbell_order[2])

    return data

def run_bm4d_filter(image_data, std=0):
    """
    Apply the BM4D filter to denoise the image.

    This method retrieves the image data, rescales it, calculates the standard deviation for the BM4D filter,
    applies the BM4D filter to denoise the rescaled image, and rescales the denoised image back to its original
    scale.

    Args:
        image_data (ndarray): The input image data.
        std (float): standard deviation for bm4d. If zero, the standard deviation will be calculated automatically. Default 0

    Returns:
        ndarray: The denoised image.

    """
    # Rescale the image data for processing
    reference = np.max(image_data)
    image_rescaled = image_data / reference * 100

    # Calculate the standard deviation for BM4D filter
    if std==0:
        # Quantize image
        num_bins = 1000
        image_quantized = np.digitize(image_rescaled, bins=np.linspace(0, 1, num_bins + 1)) - 1

        # Divide the image into blocks
        n_multi = (np.array(image_quantized.shape) / 5).astype(int) * 5
        blocks_r = view_as_blocks(image_rescaled[0:n_multi[0], 0:n_multi[1], 0:n_multi[2]], block_shape=(5, 5, 5))

        # Calculate the standard deviation for each block
        block_std_devs = np.std(blocks_r, axis=(3, 4, 5))

        # Calculate the average value for each block
        block_mean = np.mean(blocks_r, axis=(3, 4, 5))

        # Find the indices of the block with the minimum mean
        min_mean_index = np.unravel_index(np.argmin(block_mean), block_mean.shape)

        # Extract the block with the highest entropy from the block_std_devs array
        std = 4 * block_std_devs[min_mean_index]
        print("Standard deviation for BM4D: %0.2f" % std)

    # Create a BM4D profile and set options
    profile = bm4d.BM4DProfile()
    stage_arg = bm4d.BM4DStages.ALL_STAGES
    blockmatches = (False, False)

    # Apply the BM4D filter to the rescaled image
    denoised_rescaled = bm4d.bm4d(image_rescaled, sigma_psd=std, profile=profile, stage_arg=stage_arg,
                                  blockmatches=blockmatches)

    # Rescale the denoised image back to its original dimensions
    denoised_image = denoised_rescaled / 100 * reference

    return denoised_image

def run_dfft(image):
    """
    Perform direct FFT reconstruction.

    This method performs the direct Fourier transform to obtain the k-space data from an image in the spatial
    domain.

    Args:
        image (ndarray): The image in the spatial domain (sl, ph, rd).

    Returns:
        ndarray: The k-space data.

    """
    k_space = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))
    return k_space

def run_ifft(k_space):
    """
    Perform inverse FFT reconstruction.

    This method performs the inverse Fourier transform to reconstruct the image in the spatial domain from k-space
    data.

    Args:
        k_space (ndarray): The k-space data (sl, ph, rd).

    Returns:
        ndarray: The reconstructed image in the spatial domain.

    """
    image = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(k_space)))
    return image

def run_zero_padding(k_space, new_size):
    """
    Perform zero-padding on the given k-space data.

    This function retrieves the desired zero-padding dimensions from the user input,
    creates a new k-space matrix with the specified size filled with zeros, and places
    the original k-space data at the center of the new matrix. It ensures that the
    k-space is properly centered even if the new dimensions are larger or smaller
    than the original.

    Parameters:
    -----------
    k_space : np.ndarray
        A 3D complex-valued NumPy array representing the original k-space data.
    new_size: list
        A list with the new matrix size (rd, ph, sli)

    Returns:
    --------
    np.ndarray
        A 3D complex-valued NumPy array containing the new k-space data.

    Notes:
    ------
    - The function calculates the appropriate offsets to center the original k-space
      within the new zero-padded matrix.
    - If the new matrix size is smaller in any dimension, it crops the k-space accordingly.
    """
    # Get the k_space shape
    shape_0 = k_space.shape

    # Determine the new shape after zero-padding
    shape_1 = new_size

    # Create an image matrix filled with zeros
    image_matrix = np.zeros(shape_1, dtype=complex)

    # Calculate the centering offsets for each dimension
    offset_0 = (shape_1[0] - shape_0[0]) // 2
    offset_1 = (shape_1[1] - shape_0[1]) // 2
    offset_2 = (shape_1[2] - shape_0[2]) // 2

    # Calculate the start and end indices to center the k_space within the new image_matrix
    new_start_0 = offset_0 if offset_0 >= 0 else 0
    new_start_1 = offset_1 if offset_1 >= 0 else 0
    new_start_2 = offset_2 if offset_2 >= 0 else 0
    new_end_0 = new_start_0 + shape_0[0] if offset_0 > 0 else shape_1[0]
    new_end_1 = new_start_1 + shape_0[1] if offset_1 > 0 else shape_1[1]
    new_end_2 = new_start_2 + shape_0[2] if offset_2 > 0 else shape_1[2]

    # Calculate the start and end indices of old matrix
    old_start_0 = 0 if offset_0 >= 0 else -offset_0
    old_start_1 = 0 if offset_1 >= 0 else -offset_1
    old_start_2 = 0 if offset_2 >= 0 else -offset_2
    old_end_0 = shape_0[0] if offset_0 >= 0 else old_start_0 + shape_1[0]
    old_end_1 = shape_0[1] if offset_1 >= 0 else old_start_1 + shape_1[1]
    old_end_2 = shape_0[2] if offset_2 >= 0 else old_start_2 + shape_1[2]

    # Copy the k_space into the image_matrix at the center
    image_matrix[new_start_0:new_end_0, new_start_1:new_end_1, new_start_2:new_end_2] = k_space[
                                                                                        old_start_0:old_end_0,
                                                                                        old_start_1:old_end_1,
                                                                                        old_start_2:old_end_2
                                                                                        ]

    return image_matrix

def hanning_filter(kSpace, mm, nb_point):
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

def run_pocs_reconstruction(n_points, factors, k_space_ref, test=False):
    """
    Perform POCS reconstruction.

    Retrieves the number of points before m+n where reconstruction begins to go to zero.
    Retrieves the correlation threshold for stopping the iterations.
    Computes the partial image and full image for POCS reconstruction.
    Applies the iterative reconstruction with phase correction.
    Updates the main matrix of the image view widget with the interpolated image.
    Adds the "POCS" operation to the history widget and updates the history dictionary and operations history.
    """
    print("Running POCS...")
    def getCenterKSpace(k_space, m_vec):
        # fix n_vec
        output = np.zeros(np.shape(k_space), dtype=complex)
        n_vec = np.array(np.shape(k_space))

        # fill with zeros
        idx0 = n_vec // 2 - m_vec
        idx1 = n_vec // 2 + m_vec
        output[idx0[0]:idx1[0], idx0[1]:idx1[1], idx0[2]:idx1[2]] = \
            k_space[idx0[0]:idx1[0], idx0[1]:idx1[1], idx0[2]:idx1[2]]

        return output

    # Get n and m
    factors = [float(num) for num in factors]
    mm = np.array([int(num) for num in (n_points * factors)])
    m = np.array([int(num) for num in (n_points * factors - n_points / 2)])

    # Get the reference image
    img_ref = np.abs(run_ifft(k_space_ref))

    if test:
        n_sl, n_ph, n_rd = np.shape(img_ref)
        plt.figure()
        plt.imshow(img_ref[n_sl // 2, :, :], cmap="gray")
        plt.title("Reference image")

    # Create a copy with the center of k-space
    k_space_center = getCenterKSpace(k_space_ref, m)

    # Number of points before m+n where we begin to go to zero
    nb_point = 2

    # Set the correlation threshold for stopping the iterations
    threshold = 1e-6

    # Get image phase
    img_center = run_ifft(k_space_center)
    phase = img_center / abs(img_center)
    if test:
        plt.figure()
        plt.imshow(np.abs(img_center[n_sl // 2, :, :]), cmap='gray')
        plt.title("Center image")

    # Generate zero padding image for comparison
    k_space_zp = np.zeros_like(k_space_ref, dtype=complex)
    k_space_zp[0:mm[0], 0:mm[1], 0:mm[2]] = k_space_ref[0:mm[0], 0:mm[1], 0:mm[2]]
    img_zp = np.abs(run_ifft(k_space_zp))
    if test:
        plt.figure()
        plt.imshow(np.abs(img_zp[n_sl // 2, :, :]), cmap='gray')
        plt.title("ZP")

    # Generate the corresponding image with the Hanning filter
    k_space_hanning = hanning_filter(k_space_ref, mm, nb_point)
    img_hanning = run_ifft(k_space_hanning)

    num_iterations = 0  # Initialize the iteration counter
    img_reconstructed = [img_hanning]
    while True:
        # Iterative reconstruction
        img_iterative = np.abs(img_reconstructed[-1]) * phase
        k_space_new = run_dfft(img_iterative)

        # Apply constraint: Keep the region of k-space from n+m onwards and restore the rest
        k_space_new[0:mm[0], 0:mm[1], 0:mm[2]] = k_space_ref[0:mm[0], 0:mm[1], 0:mm[2]]

        # Reconstruct the image from the modified k-space
        img_reconstructed.append(run_ifft(k_space_new))

        if test:
            plt.figure()
            plt.imshow(np.abs(np.abs(img_reconstructed[-1][n_sl // 2, :, :]) - np.abs(img_reconstructed[-2][n_sl // 2, :, :])),
                       vmin=0, vmax=1e-5)
            plt.title(f"Iteration {num_iterations+1}")

        # Compute correlation between consecutive reconstructed images
        correlation = np.corrcoef(np.abs(img_reconstructed[-2].flatten()), np.abs(img_reconstructed[-1].flatten()))[0, 1]

        # Display correlation and current iteration number
        print("Iteration: %i, Convergence: %0.2e" % (num_iterations, (1 - correlation)))

        # Check if correlation reaches the desired threshold
        if (1-correlation) <= threshold or num_iterations >= 100:
            break

        # Increment the iteration counter
        num_iterations += 1

    if test:
        plt.figure()
        plt.imshow(np.abs(img_reconstructed[-1][n_sl // 2, :, :]), cmap='gray')
        plt.title(f"POCS")
        plt.show()


        # Get correlation with reference image
        correlation_1 = np.corrcoef(np.abs(img_ref.flatten()), np.abs(img_reconstructed[-1].flatten()))[0, 1]
        print("POCS compared to reference image:")
        print("Convergence: %0.2e" % (1 - correlation_1))

    return k_space_new

def run_zero_padding_reconstruction(n_points, factors, k_space_ref):
    """
    Run the partial reconstruction operation.

    Retrieves the necessary parameters and performs the partial reconstruction on the loaded image.
    Updates the main matrix of the image view widget with the partially reconstructed image, adds the operation to
    the history widget, and updates the operations history.
    """
    # Get the k_space data and its shape
    img_ref = run_ifft(k_space_ref)

    # Percentage for partial reconstruction from the text field
    factors = [float(num) for num in factors][-1::-1]
    mm = np.array([int(num) for num in (n_points * factors)])

    # Set to zero the corresponding values
    k_space = np.zeros_like(k_space_ref, dtype=complex)
    k_space[:mm[0], :mm[1], :mm[2]] = k_space_ref[:mm[0], :mm[1], :mm[2]]

    # Calculate logarithmic scale
    image = run_ifft(k_space)

    # Get correlation with reference image
    correlation = np.corrcoef(np.abs(img_ref.flatten()), np.abs(image.flatten()))[0, 1]
    print("ZP compared to reference image:")
    print("Convergence: %0.2e" % (1 - correlation))

    return np.abs(image)

def fix_echo_position(data_oversampled, dummy_pulses, etl, n_rd, n_batches, n_readouts, n_scans, add_rd_points, oversampling_factor):
    """
    Adjust the position of k=0 in the echo data to the center of the acquisition window.

    This method uses oversampled data obtained with a given echo train length and readout gradient to determine the
    true position of k=0. It then shifts the sampled data to place k=0 at the center of each acquisition window for
    each gradient-spin-echo.

    Args:
        data_oversampled (numpy.ndarray): The original data array to be adjusted with dimensions [channels, etl, n].

    Returns:
        numpy.ndarray: The adjusted data array with k=0 positioned at the center of each acquisition window.

    """
    # Get relevant data
    data_noise = []
    data_dummy = []
    data_signal = []
    points_per_rd = n_rd * oversampling_factor
    points_per_train = points_per_rd * etl
    idx_0 = 0
    idx_1 = 0
    for batch in range(n_batches):
        n_rds = n_readouts[batch] * oversampling_factor
        for scan in range(n_scans):
            idx_1 += n_rds
            data_prov = data_oversampled[idx_0:idx_1]
            data_noise = np.concatenate((data_noise, data_prov[0:points_per_rd]), axis=0)
            if dummy_pulses > 0:
                data_dummy = np.concatenate((data_dummy, data_prov[points_per_rd:points_per_rd + points_per_train]),
                                            axis=0)
            data_signal = np.concatenate((data_signal, data_prov[points_per_rd + points_per_train::]), axis=0)
            idx_0 = idx_1

    # Get echo position
    data_dummy_b = np.reshape(data_dummy, (-1, etl, points_per_rd))
    data_dummy_b = np.average(data_dummy_b, axis=0)
    idx = np.argmax(np.abs(data_dummy_b[:, 5:-5]), axis=1) + 5

    # Apply to full data
    data_signal = []
    idx_0 = 0
    idx_1 = 0
    for batch in range(n_batches):
        n_rds = n_readouts[batch] * oversampling_factor
        for scan in range(n_scans):
            idx_1 += n_rds
            data_prov = data_oversampled[idx_0:idx_1]
            if dummy_pulses > 0:
                data_dummy = data_prov[points_per_rd:points_per_rd + points_per_train]
                data_dummy = np.reshape(data_dummy, (etl, points_per_rd))
                data_signal = data_prov[points_per_rd + points_per_train::]
                data_signal = np.reshape(data_signal, (-1, etl, points_per_rd))
                for ii in range(etl):
                    dx = 0
                    if idx[ii] >= points_per_rd // 2:
                        dx = points_per_rd - idx[ii]
                    if idx[ii] < points_per_rd // 2:
                        dx = idx[ii]
                    data_dummy[ii, points_per_rd // 2 - dx:points_per_rd // 2 + dx] = data_dummy[ii, idx[ii] - dx: idx[ii] + dx]
                    data_signal[:, ii, points_per_rd // 2 - dx:points_per_rd // 2 + dx] = data_signal[:, ii, idx[ii] - dx: idx[ii] + dx]
                data_dummy = np.reshape(data_dummy, -1)
                data_signal = np.reshape(data_signal, -1)
                data_prov[points_per_rd:points_per_rd + points_per_train] = data_dummy
                data_prov[points_per_rd + points_per_train::] = data_signal
            data_oversampled[idx_0:idx_1] = data_prov
            idx_0 = idx_1

    # Decimate the signal
    data_decimated = decimate(data_over=data_oversampled,
                              n_adc=np.size(data_oversampled)//points_per_rd,
                              option='Normal',
                              remove=False,
                              add_rd_points=add_rd_points,
                              oversampling_factor=oversampling_factor)

    return data_decimated

def decimate(data_over, n_adc, option='PETRA', remove=True, add_rd_points=10, oversampling_factor=5):
    """
    Decimates oversampled MRI data, with optional preprocessing to manage oscillations and postprocessing
    to remove extra points.

    Parameters:
    -----------
    data_over : numpy.ndarray
        The oversampled data array to be decimated.
    n_adc : int
        The number of adc windows in the dataset, used to reshape and process the data appropriately.
    option : str, optional
        Preprocessing option to handle data before decimation:
        - 'PETRA': Adjusts initial points to avoid oscillations during decimation.
        - 'Normal': Applies no preprocessing (default is 'PETRA').
    remove : bool, optional
        If True, removes `addRdPoints` from the start and end of each readout line after decimation.
        Defaults to True.
    add_rd_points : int, optional
        Number of additional points at the begining and end of each readout line.
        Defaults to 10.
    oversampling_factor : int, optional
        Oversampling factor applied to data before decimation.
        Defaults to 5.

    Returns:
    --------
    numpy.ndarray
        The decimated data array, optionally adjusted to remove extra points.

    Workflow:
    ---------
    1. **Preprocess data (optional)**:
        - For 'PETRA' mode, reshapes the data into adc windows and adjusts the first few points of each line
          to avoid oscillations caused by decimation.
        - For 'Normal' mode, no preprocessing is applied.

    2. **Decimate the signal**:
        - Applies a finite impulse response (FIR) filter and decimates the signal by the oversampling factor
          (`hw.oversamplingFactor`).
        - Starts decimation after skipping `(oversamplingFactor - 1) / 2` points to minimize edge effects.

    3. **Postprocess data (if `remove=True`)**:
        - Reshapes the decimated data into adc windows.
        - Removes `hw.addRdPoints` from the start and end of each line.
        - Reshapes the cleaned data back into a 1D array.

    Notes:
    ------
    - This method uses the hardware-specific parameters:
      - `hw.oversamplingFactor`: The oversampling factor applied during data acquisition.
      - `hw.addRdPoints`: The number of additional readout points to include or remove.
    - The 'PETRA' preprocessing mode is tailored for specialized MRI acquisitions that require smoothing of
      initial points to prevent oscillations.
    """

    # Preprocess the signal to avoid oscillations due to decimation
    if option == 'PETRA':
        data_over = np.reshape(data_over, (n_adc, -1))
        for line in range(n_adc):
            data_over[line, 0:add_rd_points * oversampling_factor] = data_over[
                line, add_rd_points * oversampling_factor]
        data_over = np.reshape(data_over, -1)
    elif option == 'Normal':
        pass

    # Decimate the signal after 'fir' filter
    if oversampling_factor > 1:
        data_decimated = sp.signal.decimate(data_over[int((oversampling_factor - 1) / 2)::],
                                            oversampling_factor,
                                            ftype='fir',
                                            zero_phase=True)
    else:
        data_decimated = data_over

    # Remove addRdPoints
    if remove:
        nPoints = int(data_decimated.shape[0] / n_adc) - 2 * add_rd_points
        data_decimated = np.reshape(data_decimated, (n_adc, -1))
        data_decimated = data_decimated[:, add_rd_points:add_rd_points + nPoints]
        data_decimated = np.reshape(data_decimated, -1)

    return data_decimated

def get_snr_histogram(image, roi_size=4):
    """
    Compute a pixel-wise SNR map for a 3D image.
    1) signal: mean value in a cubic ROI.
    2) noise: mean noise from the histogram.
    SNR = local signal / noise from the histogram.

    Parameters
    ----------
    image : np.ndarray
        3D image (e.g., shape (x, y, z)).
    roi_size : int
        Size of the cubic ROI (odd number recommended, e.g., 3, 5, 7).

    Returns
    -------
    snr_map : np.ndarray
        3D array with same shape as `image`, containing SNR values.
    """

    image = np.abs(image)
    image = image.astype(np.float64)

    # Local mean
    mean = sp.ndimage.uniform_filter(image, roi_size)

    # Get histogram
    counts, bins = np.histogram(mean, bins=512)

    # Find the peak (maximum count)
    peak_bin_index = np.argmax(counts)
    peak_value = (bins[peak_bin_index] + bins[peak_bin_index + 1]) / 2

    # Get SNR
    snr = mean / peak_value
    return snr

def get_snr_3d(image, roi_size=4):
    """
    Compute a pixel-wise SNR map for a 3D image.
    SNR = local mean / local std, computed in a cubic ROI.

    Parameters
    ----------
    image : np.ndarray
        3D image (e.g., shape (x, y, z)).
    roi_size : int
        Size of the cubic ROI (odd number recommended, e.g., 3, 5, 7).

    Returns
    -------
    snr_map : np.ndarray
        3D array with same shape as `image`, containing SNR values.
    """

    image = np.abs(image)
    image = image.astype(np.float64)
    # Local mean
    mean = sp.ndimage.uniform_filter(image, roi_size)
    # Local variance and std
    var = sp.ndimage.uniform_filter((image - mean) ** 2, roi_size)
    std = np.sqrt(var)
    # Avoid division by zero
    snr = mean / std
    return snr

def get_snr_2d(image, roi_size=4):
    """
    Compute a pixel-wise SNR map for a 3D image slice by slice.
    SNR = local mean / local std, computed in a cubic ROI.

    Parameters
    ----------
    image : np.ndarray
        3D image (e.g., shape (x, y, z)).
    roi_size : int
        Size of the cubic ROI (odd number recommended, e.g., 3, 5, 7).

    Returns
    -------
    snr_map : np.ndarray
        3D array with same shape as `image`, containing SNR values.
    """

    n_sl, n_ph, n_rd = image.shape
    snr = np.zeros((n_sl, n_ph, n_rd))
    for sl in range(n_sl):
        image_sl = np.abs(image[sl])
        image_sl = image_sl.astype(np.float64)
        # Local mean
        mean = sp.ndimage.uniform_filter(image_sl, roi_size)
        # Local variance and std
        var = sp.ndimage.uniform_filter((image_sl - mean) ** 2, roi_size)
        std = np.sqrt(var)
        # Avoid division by zero
        snr[sl] = mean / std
    return snr

def get_snr_from_individual_acquisitions(data, roi_size):
    """
    Compute a voxel-wise SNR map from a 4D dataset composed of repeated acquisitions.

    The function treats the first dimension of `data` as the acquisition axis:
    - The **signal** is computed as the mean magnitude across acquisitions,
      followed by a uniform (mean) filter of size `roi_size`.
    - The **noise** is computed as the standard deviation of the magnitude across
      acquisitions, also followed by the same uniform filter.

    Parameters
    ----------
    data : np.ndarray
        4D array (n_acq, x, y, z) containing repeated measurements of the same volume.
    roi_size : int
        Size of the cubic window for the uniform filter. An odd value (e.g., 3, 5, 7)
        is recommended to maintain symmetry.

    Returns
    -------
    snr_map : np.ndarray
        3D array with shape (x, y, z), containing the computed SNR values.
    """

    noise = np.std(np.abs(data), axis=0)
    mean = np.mean(np.abs(data), axis=0)
    noise = sp.ndimage.uniform_filter(noise, roi_size)
    signal = sp.ndimage.uniform_filter(mean, roi_size)
    snr = signal / noise
    return snr

# TODO: include new filters and other methods from Miguel


if __name__ == "__main__":
    # from matplotlib import pyplot as plt
    # mat_data = sp.io.loadmat("C:/CSIC/RareDoubleImage.2025.09.24.13.32.29.066.mat")
    # # mat_data = sp.io.loadmat("C:/CSIC/RareDoubleImage.2025.09.24.13.03.47.729.mat")
    #
    # n_points = mat_data['nPoints'][0][-1::-1]
    #
    # # Number of extra lines which has been taken past the center of k-space
    # factors = [0.7, 1, 1]
    #
    # # Get the k_space data
    # k_space_ref = mat_data['kSpace3D']
    #
    # # Run pocs
    # k_space_ref_zp = run_zero_padding(k_space_ref, (20, 240, 240))
    # img_reconstructed = run_pocs_reconstruction(n_points, factors, k_space_ref, test=True)

    ####################################################################################################################
    # Fix echo position example
    ####################################################################################################################
    from matplotlib import pyplot as plt
    mat_data = sp.io.loadmat("RareDoubleImage.2025.09.22.14.51.12.793.mat")
    data = mat_data["data_over"][0]
    n_batches = mat_data['n_batches'].item()
    n_readouts = mat_data['n_readouts'][0]
    etl = mat_data['etl'].item()
    n_scans = mat_data['nScans'].item()
    dummy_pulses = mat_data['dummyPulses'].item()
    add_rd_points = mat_data['addRdPoints'].item()
    n_points = np.squeeze(mat_data['nPoints'])
    n_rd, n_ph, n_sl = n_points
    n_rd = n_rd + 2 * add_rd_points

    # Run method
    fix_echo_position(data, dummy_pulses, etl, n_rd, n_batches, n_readouts, n_scans, 10, 5)
