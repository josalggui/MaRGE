import numpy as np
import scipy.io as sio
import mrd


def export(mrd_input, mat_in_path: str, mat_out_path: str = None,
           out_field: str = "image3D_den",
           out_field_k: str = None):
    """
    Lee un MRD (stream) producido por stream_recon_RARE.py y escribe los campos
    de salida en el MAT. Si mat_out_path == mat_in_path (o es None), sobreescribe
    el MAT original añadiendo los campos nuevos sin generar archivos adicionales.

    Parametros
    ----------
    mrd_input    : stream MRD de entrada (BytesIO o file-like).
    mat_in_path  : ruta al MAT original (se usa como base para copiar todos los campos).
    mat_out_path : ruta de salida. Si es None o igual a mat_in_path, sobreescribe
                   el original.
    out_field    : campo de imagen denoised (magnitud, float32).
    out_field_k  : campo de k-space denoised (fftshift(fftn(fftshift(img))),
                   complejo64). Necesario para correccion de distorsiones posterior.
    """

    # Si no se especifica salida, sobreescribir el original
    if mat_out_path is None:
        mat_out_path = mat_in_path

    # Cargamos MAT original para copiar todos sus campos y añadir los nuevos
    mat = sio.loadmat(mat_in_path)

    # Abrimos MRD reader (admite file-like, BytesIO, etc.)
    with mrd.BinaryMrdReader(mrd_input) as r:
        r.read_header()

        img_data = None

        # IMPORTANTE: consumir el iterador completo para no disparar ProtocolError
        for item in r.read_data():
            if isinstance(item, mrd.StreamItem.ImageFloat):
                img = item.value  # mrd.Image
                img_data = np.asarray(img.data)  # 4D (1, sl, ph, rd)
                # NO hacemos break: seguimos consumiendo hasta el final

        if img_data is None:
            raise RuntimeError("No se encontro ninguna ImageFloat en el MRD de salida.")

    # Convertimos de 4D a 3D si venia con batch=1
    # stream_recon_RARE escribe (1, sl, ph, rd)
    if img_data.ndim == 4 and img_data.shape[0] == 1:
        img_data = img_data[0]  # (sl, ph, rd)

    # Campo de imagen denoised (magnitud, float32)
    mat[out_field] = img_data.astype(np.float32, copy=False)

    # Campo de k-space denoised: fftshift(fftn(fftshift(img))) -> (sl, ph, rd), complejo64
    # Permite que la correccion de distorsiones opere sobre el k-space denoised.
    if out_field_k is not None:
        kspace_den = np.fft.fftshift(
            np.fft.fftn(
                np.fft.fftshift(img_data.astype(np.complex64, copy=False))
            )
        ).astype(np.complex64, copy=False)
        mat[out_field_k] = kspace_den
        print(f"Export OK: '{out_field_k}' (k-space denoised) guardado en {mat_out_path}")

    # Guardar (sobreescribe el original si mat_out_path == mat_in_path)
    sio.savemat(mat_out_path, mat)
    print(f"Export OK: '{out_field}' guardado en {mat_out_path}")
