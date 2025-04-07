import bm4d
import numpy as np
import nibabel as nib
from skimage.util import view_as_blocks
from scipy.ndimage import map_coordinates


def run_distortion_correction(image, dx, dy, dz):
    """
    Apply a warp to a 3D image using displacement fields dx, dy, dz.

    Parameters:
        image (numpy.ndarray): 3D input image.
        dx (numpy.ndarray): Displacement in x-direction (same shape as image).
        dy (numpy.ndarray): Displacement in y-direction (same shape as image).
        dz (numpy.ndarray): Displacement in z-direction (same shape as image).

    Returns:
        numpy.ndarray: Warped 3D image.
    """
    depth, rows, cols = image.shape  # Get dimensions

    # Create coordinate grids
    z, y, x = np.meshgrid(
        np.arange(depth), np.arange(rows), np.arange(cols), indexing="ij"
    )

    # Apply displacement and clip to valid indices
    x_new = np.clip(x + dx, 0, cols - 1)
    y_new = np.clip(y + dy, 0, rows - 1)
    z_new = np.clip(z + dz, 0, depth - 1)

    # Interpolate new coordinates
    warped_image = map_coordinates(image, [z_new.ravel(), y_new.ravel(), x_new.ravel()], order=1)

    return warped_image.reshape(image.shape)

def fix_image_orientation(image, axes, orientation='FFS'):
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
            y_label = "(-X) I | READOUT | S (+X)"
        else:
            image = np.transpose(image, (0, 2, 1))
            image = np.flip(image, axis=0)
            x_label = "(-Y) A | READOUT | P (+Y)"
            y_label = "(-X) I | PHASE | S (+X)"
        image_orientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
    elif axes[2] == 1:  # Coronal
        title = "Coronal"
        if axes[0] == 0 and axes[1] == 2:
            x_label = "(+Z) R | PHASE | L (-Z)"
            y_label = "(-X) I | READOUT | S (+X)"
        else:
            image = np.transpose(image, (0, 2, 1))
            x_label = "(+Z) R | READOUT | L (-Z)"
            y_label = "(-X) I | PHASE | S (+X)"
        image_orientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    elif axes[2] == 0:  # Transversal
        title = "Transversal"
        if axes[0] == 1 and axes[1] == 2:
            image = np.flip(image, axis=0)
            x_label = "(+Z) R | PHASE | L (-Z)"
            y_label = "(+Y) P | READOUT | A (-Y)"
        else:
            image = np.transpose(image, (0, 2, 1))
            image = np.flip(image, axis=0)
            x_label = "(+Z) R | READOUT | L (-Z)"
            y_label = "(+Y) P | PHASE | A (-Y)"
        image_orientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    output = {
        'widget': 'image',
        'data': image,
        'xLabel': x_label,
        'yLabel': y_label,
        'title': title,
    }

    return output, image, image_orientation_dicom

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

        # Set the origin at the center of the image
        center_voxel = np.array(n_xyz[::-1]) / 2  # Reverse order to match data axes
        origin_mm = -center_voxel * resolution[::-1]  # Reverse resolution to match
        affine[:3, 3] = origin_mm

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
    denoised_rescaled = bm4d.bm4d(image_rescaled, sigma_psd=5, profile=profile, stage_arg=stage_arg,
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
    n_rd = int(new_size[0])
    n_ph = int(new_size[1])
    n_sl = int(new_size[2])
    shape_1 = n_sl, n_ph, n_rd

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
