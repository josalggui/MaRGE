"""
tyger_denoising.py
==================
Pipeline de denoising via Tyger integrado con MaRGE.

Firma publica (compatible con sequenceAnalysis):
    denoisingTyger(rawData_path, output_field, output_field_k)

Los campos output_field (imagen denoised) y output_field_k (k-space denoised)
se escriben directamente en el MAT original, sin generar archivos adicionales.
"""

import io
import os
import sys
import time
import subprocess
import scipy.io as sio

from marge.marge_tyger.fromMATtoMRD3D_RARE_local_denoising import matToMRD
from marge.marge_tyger.fromMRDtoMAT3D_local_denoising import export

# Ruta fija al YML de Tyger, en el mismo directorio que este archivo
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_YML = os.path.join(_THIS_DIR, "tyger_example_local_denoising.yml")


def denoisingTyger(rawData_path: str,
                   output_field: str,
                   output_field_k: str,
                   input_field: str = ""):
    """
    Ejecuta el pipeline completo de denoising:
        MAT  ->  MRD (en memoria)
             ->  tyger run exec -f tyger_example_local_denoising.yml  (GPU remota)
             ->  MRD (en memoria)
             ->  MAT original  (se añaden output_field y output_field_k)

    Parametros
    ----------
    rawData_path   : ruta al .mat de entrada/salida (se sobreescribe).
    output_field   : campo de imagen denoised que se añade al MAT
                     (p. ej. "image3D_den").
    output_field_k : campo de k-space denoised que se añade al MAT
                     (p. ej. "kSpace3D_den"). Calculado como
                     fftshift(fftn(fftshift(imagen_denoised))).
    input_field    : campo opcional de k-space dentro del MAT de entrada
                     (cadena vacia -> usa sampledCartesian, por defecto).

    Retorna
    -------
    img_data : np.ndarray con la imagen denoised (sl, ph, rd), float32.
    """

    # ------------------------------------------------------------------
    # 0. Comprobaciones
    # ------------------------------------------------------------------
    if not os.path.exists(rawData_path):
        raise FileNotFoundError(f"No existe rawData_path: {rawData_path}")

    if not os.path.exists(_DEFAULT_YML):
        raise FileNotFoundError(f"No se encuentra el YML de Tyger en: {_DEFAULT_YML}")

    # ------------------------------------------------------------------
    # 1. MAT -> MRD  (en memoria, sin tocar el disco)
    # ------------------------------------------------------------------
    print("Tyger denoising | paso 1/3: MAT -> MRD (en memoria)...")
    mrd_buffer = io.BytesIO()
    matToMRD(input=rawData_path, output_file=mrd_buffer, input_field=input_field)
    mrd_buffer.seek(0)
    in_bytes = mrd_buffer.getvalue()

    # ------------------------------------------------------------------
    # 2. Tyger run exec  (MRD bytes -> GPU remota -> MRD bytes)
    # ------------------------------------------------------------------
    print(f"Tyger denoising | paso 2/3: tyger run exec -f {_DEFAULT_YML} ...")
    start = time.time()

    p = subprocess.run(
        ["tyger", "run", "exec", "-f", _DEFAULT_YML],
        input=in_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if p.stderr:
        sys.stderr.write(p.stderr.decode("utf-8", errors="replace"))
        sys.stderr.flush()

    dt = time.time() - start
    print(f"Tyger denoising | tiempo Tyger exec: {dt:.2f} s")

    if p.returncode != 0:
        raise RuntimeError(f"tyger run exec fallo con returncode={p.returncode}")

    out_bytes = p.stdout
    if not out_bytes:
        raise RuntimeError("tyger run exec devolvio stdout vacio.")

    # ------------------------------------------------------------------
    # 3. MRD -> MAT original  (añade los campos, no crea archivo nuevo)
    # ------------------------------------------------------------------
    print(f"Tyger denoising | paso 3/3: MRD -> MAT ({rawData_path})...")
    out_buf = io.BytesIO(out_bytes)
    out_buf.seek(0)

    export(
        mrd_input=out_buf,
        mat_in_path=rawData_path,
        mat_out_path=rawData_path,      # mismo archivo -> sobreescritura
        out_field=output_field,
        out_field_k=output_field_k if output_field_k else None,
    )

    print(f"Tyger denoising | '{output_field}' y '{output_field_k}' guardados en {rawData_path}")

    # Devolver la imagen denoised (sl, ph, rd), float32
    mat_out = sio.loadmat(rawData_path)
    return mat_out[output_field]
