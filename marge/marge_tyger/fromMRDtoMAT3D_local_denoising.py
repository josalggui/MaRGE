import numpy as np
import scipy.io as sio
import mrd


def export(mrd_input, mat_in_path: str, mat_out_path: str = None,
           out_field: str = "image3D_den",
           out_field_k: str = None):
    """
    Read a locally denoised image from an MRD stream and save it into a .mat file.

    Reads axesOrientation from the MRD header and reorders the image axes from
    physical space (ch, x, y, z) to MaRGE format (sl, ph, rd). Optionally computes
    the denoised k-space via FFT and saves it under a second field.

    Args:
        mrd_input (file-like | str): Binary MRD stream or path to the input file.
        mat_in_path (str): Path to the input .mat file (used to preserve existing fields).
        mat_out_path (str, optional): Path to the output .mat file. Defaults to mat_in_path.
        out_field (str, optional): Field name for the denoised image. Defaults to "image3D_den".
        out_field_k (str, optional): If provided, also saves the denoised k-space under this field name.

    Raises:
        RuntimeError: If no ImageFloat item is found in the MRD stream.
    """
    if mat_out_path is None:
        mat_out_path = mat_in_path

    mat = sio.loadmat(mat_in_path)

    with mrd.BinaryMrdReader(mrd_input) as r:
        header = r.read_header()

        # Read axesOrientation: maps acq dims (rd=0, ph=1, sl=2) to spatial axes (x=0, y=1, z=2)
        axesOrientation = [0, 1, 2]  # default: rd=x, ph=y, sl=z
        if header.user_parameters:
            for param in header.user_parameters.user_parameter_string:
                if param.name == 'axesOrientation':
                    axesOrientation = [int(v) for v in param.value.split(',')]
                    break

        img_data = None

        for item in r.read_data():
            if isinstance(item, mrd.StreamItem.ImageFloat):
                img = item.value
                # img.data is (ch, x, y, z) in physical space.
                # Reorder to MaRGE format (ch, sl, ph, rd):
                perm = (0, 1 + axesOrientation[2], 1 + axesOrientation[1], 1 + axesOrientation[0])
                img_data = np.transpose(np.asarray(img.data), perm)

        if img_data is None:
            raise RuntimeError("No se encontro ninguna ImageFloat en el MRD de salida.")

    # If batch=1, from 4D to 3D (sl, ph, rd)
    print("Img shape: ",img_data.shape)
    if img_data.ndim == 4 and img_data.shape[0] == 1:
        img_data = img_data[0]  # (sl, ph, rd)

    mat[out_field] = img_data.astype(np.float32, copy=False)

    # k-space denoised: fftshift(fftn(fftshift(img))) -> (sl, ph, rd), complex64
    # Allows distortion correction on the k-space denoised.
    if out_field_k is not None:
        kspace_den = np.fft.fftshift(
            np.fft.fftn(
                np.fft.fftshift(img_data.astype(np.complex64, copy=False))
            )
        ).astype(np.complex64, copy=False)
        mat[out_field_k] = kspace_den
        print(f"Export OK: '{out_field_k}' (k-space denoised) saved at {mat_out_path}")

    # Save (overwrites original if mat_out_path == mat_in_path)
    sio.savemat(mat_out_path, mat)
    print(f"Export OK: '{out_field}' saved at {mat_out_path}")
