import numpy as np
import nibabel as nib

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