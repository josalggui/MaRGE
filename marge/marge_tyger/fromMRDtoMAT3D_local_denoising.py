import numpy as np
import scipy.io as sio
import mrd


def export(mrd_input, mat_in_path: str, mat_out_path: str = None,
           out_field: str = "image3D_den",
           out_field_k: str = None):
    
    if mat_out_path is None:
        mat_out_path = mat_in_path

    mat = sio.loadmat(mat_in_path)

    with mrd.BinaryMrdReader(mrd_input) as r:
        r.read_header()

        img_data = None

        for item in r.read_data():
            if isinstance(item, mrd.StreamItem.ImageFloat):
                img = item.value 
                img_data = np.asarray(img.data)  # 4D (1, sl, ph, rd)

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
