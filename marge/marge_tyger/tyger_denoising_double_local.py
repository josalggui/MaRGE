import io
import os
import sys
import time
import subprocess
import numpy as np
import scipy.io as sio

from marge.marge_tyger.fromMATtoMRD3D_RAREdouble_local import matToMRD
from marge.marge_tyger.fromMRDtoMAT3D_local_denoising import export

# YML en el mismo directorio que este archivo
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_YML = os.path.join(_THIS_DIR, "tyger_local_denoising.yml")


def _run_single(rawData_path: str,
                input_field_raw: str,
                out_field: str,
                out_field_k: str,
                label: str = "") -> np.ndarray:
    """
    Run a single Tyger denoising pass for one echo.

    Converts MAT[input_field_raw] to MRD in memory, submits to Tyger,
    and writes the result back to MAT[out_field].

    Args:
        rawData_path (str): Path to the .mat file.
        input_field_raw (str): .mat field name of the k-space echo to process.
        out_field (str): .mat field name for the denoised image output.
        out_field_k (str): .mat field name for the denoised k-space output.
        label (str, optional): Label appended to log messages (e.g. ' (odd)').

    Returns:
        np.ndarray: Denoised image array with shape (sl, ph, rd).
    """

    # 1. MAT -> MRD en memoria
    print(f"Tyger denoising{label} | paso 1/3: MAT -> MRD en memoria (campo '{input_field_raw}')...")
    mrd_buffer = io.BytesIO()
    matToMRD(input=rawData_path, output_file=mrd_buffer, input_field_raw=input_field_raw)
    mrd_buffer.seek(0)
    in_bytes = mrd_buffer.getvalue()

    # 2. tyger run exec (stdin -> stdout)
    print(f"Tyger denoising{label} | paso 2/3: tyger run exec -f {_DEFAULT_YML} ...")
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
    print(f"Tyger denoising{label} | tiempo Tyger exec: {dt:.2f} s")

    if p.returncode != 0:
        raise RuntimeError(
            f"tyger run exec falló con returncode={p.returncode}\n"
            f"stderr: {p.stderr.decode('utf-8', errors='replace')}"
        )

    out_bytes = p.stdout
    if not out_bytes:
        raise RuntimeError("tyger run exec devolvió stdout vacío")

    # 3. MRD -> MAT
    print(f"Tyger denoising{label} | paso 3/3: MRD -> MAT ({rawData_path})...")
    out_buf = io.BytesIO(out_bytes)
    out_buf.seek(0)

    export(
        mrd_input=out_buf,
        mat_in_path=rawData_path,
        mat_out_path=rawData_path,
        out_field=out_field,
        out_field_k=out_field_k if out_field_k else None,
    )

    print(f"Tyger denoising{label} | '{out_field}' guardado en {rawData_path}")

    mat_out = sio.loadmat(rawData_path)
    return mat_out[out_field]  # (sl, ph, rd)


def denoisingTyger_double(rawData_path: str,
                          output_field: str,
                          output_field_k: str,
                          input_echoes: str) -> np.ndarray:
    """
    Run the Tyger local denoising pipeline on a double-echo RARE acquisition.

    Depending on input_echoes, denoises the odd echo, the even echo, or both.
    When 'all' is selected, both echoes are denoised separately and averaged
    into a combined output field. Results are written back into the .mat file.

    Args:
        rawData_path (str): Path to the .mat file containing the raw k-space data.
        output_field (str): Base .mat field name for the denoised image
            (suffixed with '_odd', '_even', or '_all').
        output_field_k (str): Base .mat field name for the denoised k-space
            (suffixed with '_odd', '_even', or '_all').
        input_echoes (str): Which echoes to process: 'odd', 'even', or 'all'.

    Returns:
        np.ndarray: Denoised image array with shape (1, sl, ph, rd).

    Raises:
        FileNotFoundError: If rawData_path or the Tyger YML file do not exist.
    """

    if not os.path.exists(rawData_path):
        raise FileNotFoundError(f"rawData_path no existe: {rawData_path}")
    if not os.path.exists(_DEFAULT_YML):
        raise FileNotFoundError(f"YML de Tyger no encontrado en: {_DEFAULT_YML}")

    # ------------------------------------------------------------------ #
    # Modo 'even'                                                         #
    # ------------------------------------------------------------------ #
    if input_echoes == "even":
        img = _run_single(
            rawData_path=rawData_path,
            input_field_raw="sampled_eve",
            out_field=output_field + "_even",
            out_field_k=output_field_k + "_even",
            label=" (even)",
        )
        return img[np.newaxis, :, :, :]  # (1, sl, ph, rd)

    # ------------------------------------------------------------------ #
    # Modo 'odd' (o cualquier valor no reconocido)                        #
    # ------------------------------------------------------------------ #
    if input_echoes != "all":
        img = _run_single(
            rawData_path=rawData_path,
            input_field_raw="sampled_odd",
            out_field=output_field + "_odd",
            out_field_k=output_field_k + "_odd",
            label=" (odd)",
        )
        return img[np.newaxis, :, :, :]  # (1, sl, ph, rd)

    # ------------------------------------------------------------------ #
    # Modo 'all': odd + even → promedio                                   #
    # ------------------------------------------------------------------ #
    img_odd = _run_single(
        rawData_path=rawData_path,
        input_field_raw="sampled_odd",
        out_field=output_field + "_odd",
        out_field_k=output_field_k + "_odd",
        label=" (odd)",
    )

    img_eve = _run_single(
        rawData_path=rawData_path,
        input_field_raw="sampled_eve",
        out_field=output_field + "_even",
        out_field_k=output_field_k + "_even",
        label=" (even)",
    )

    # Promedio de magnitudes (igual que en la versión TEP)
    img_all = (np.abs(img_odd) + np.abs(img_eve)) / 2.0

    out_field_all   = output_field + "_all"
    out_field_k_all = output_field_k + "_all"

    kSpace3D_den = np.fft.fftshift(
        np.fft.fftn(np.fft.fftshift(img_all))
    )

    rawData = sio.loadmat(rawData_path)
    rawData[out_field_all]   = img_all.astype(np.float32)
    rawData[out_field_k_all] = kSpace3D_den.astype(np.complex64)
    sio.savemat(rawData_path, rawData) 

    print(f"Tyger denoising (all) | '{out_field_all}' y '{out_field_k_all}' guardados en {rawData_path}")

    return img_all[np.newaxis, :, :, :]  # (1, sl, ph, rd)