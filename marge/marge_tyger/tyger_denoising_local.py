"""Tyger local denoising pipeline for single-echo RARE acquisitions."""

import io
import os
import sys
import time
import subprocess
import scipy.io as sio

from marge.marge_tyger.fromMATtoMRD3D_RARE_local_denoising import matToMRD
from marge.marge_tyger.fromMRDtoMAT3D_local_denoising import export

# Ruta fija al YML de Tyger, en el mismo directorio que este archivo
# SI SE MODIFICA EL MÉTODO DE CREACIÓN DE YML HAY QUE CAMBIAR ESTAS LINEAS
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_YML = os.path.join(_THIS_DIR, "tyger_local_denoising.yml")


def denoisingTyger(rawData_path: str,
                   output_field: str,
                   output_field_k: str):
    """
    Run the Tyger local denoising pipeline on a RARE acquisition.

    Converts the input .mat file to MRD in memory, submits the job to Tyger
    using the preconfigured local denoising YAML, and writes the denoised image
    and k-space back into the .mat file.

    Args:
        rawData_path (str): Path to the .mat file containing the raw k-space data.
        output_field (str): .mat field name where the denoised image will be stored.
        output_field_k (str): .mat field name where the denoised k-space will be stored.

    Returns:
        np.ndarray: Denoised image array with shape (sl, ph, rd).

    Raises:
        FileNotFoundError: If rawData_path or the Tyger YML file do not exist.
        RuntimeError: If tyger run exec fails or returns empty output.
    """
    # ------------------------------------------------------------------
    # 0. Cheking preconditions
    # ------------------------------------------------------------------
    if not os.path.exists(rawData_path):
        raise FileNotFoundError(f"rawData_path does not exist: {rawData_path}")

    if not os.path.exists(_DEFAULT_YML):
        raise FileNotFoundError(f"Tyger YML not found at: {_DEFAULT_YML}")

    # ------------------------------------------------------------------
    # 1. MAT -> MRD 
    # ------------------------------------------------------------------
    print("Tyger denoising | step 1/3: MAT -> MRD (en memoria)...")
    mrd_buffer = io.BytesIO()
    matToMRD(input=rawData_path, output_file=mrd_buffer)
    mrd_buffer.seek(0)
    in_bytes = mrd_buffer.getvalue()

    # ------------------------------------------------------------------
    # 2. Tyger run exec  (MRD bytes -> remote GPU -> MRD bytes)
    # ------------------------------------------------------------------
    print(f"Tyger denoising | step 2/3: tyger run exec -f {_DEFAULT_YML} ...")
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
    print(f"Tyger denoising | Tyger exec time: {dt:.2f} s")

    if p.returncode != 0:
        raise RuntimeError(f"tyger run exec failed with returncode={p.returncode}")

    out_bytes = p.stdout
    if not out_bytes:
        raise RuntimeError("tyger run exec returned empty stdout")

    # ------------------------------------------------------------------
    # 3. MRD -> MAT original  
    # ------------------------------------------------------------------
    print(f"Tyger denoising | step 3/3: MRD -> MAT ({rawData_path})...")
    out_buf = io.BytesIO(out_bytes)
    out_buf.seek(0)

    export(
        mrd_input=out_buf,
        mat_in_path=rawData_path,
        mat_out_path=rawData_path, 
        out_field=output_field,
        out_field_k=output_field_k if output_field_k else None,
    )

    print(f"Tyger denoising | '{output_field}' and '{output_field_k}' saved at {rawData_path}")

    mat_out = sio.loadmat(rawData_path)
    return mat_out[output_field]
