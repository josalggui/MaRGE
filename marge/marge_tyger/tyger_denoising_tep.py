"""Tyger TEP denoising pipeline for single-echo RARE acquisitions."""

import marge.marge_tyger.tyger_config as tyger_conf
from pathlib import Path
import time
import subprocess
import io
from os.path import exists
import os
import sys
import marge.marge_tyger.tyger_config as tyger_conf
from marge.marge_tyger.fromMATtoMRD3D_RARE import matToMRD
from marge.marge_tyger.fromMATtoMRD3D_RARE_old import matToMRD_old
from marge.marge_tyger.fromMRDtoMAT3D_noise import export
from pathlib import Path
from os import stat
import mrd

def denoisingTyger(rawData_path, output_field, output_field_k):
    """
    Run the Tyger TEP denoising pipeline on a RARE acquisition.

    Converts the input .mat file to MRD, runs the external denoising bash pipeline,
    and writes the denoised image and its k-space back into the .mat file.

    Args:
        rawData_path (str): Path to the .mat file containing the raw k-space data.
        output_field (str): .mat field name where the denoised image will be stored.
        output_field_k (str): .mat field name where the denoised k-space will be stored.

    Returns:
        np.ndarray: Denoised image array as returned by the MRD export function.
    """
    # Run Tyger Recon
    print('Running Tyger denoising...')
    pathMRD_or = rawData_path.replace("/mat/", "/mrd/").replace(".mat", ".mrd")
    pathMRD_ia = rawData_path.replace("/mat/", "/mrd_ia/").replace(".mat", "_ia.mrd")
    
    for p in (pathMRD_or, pathMRD_ia):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    try:
        matToMRD(rawData_path, pathMRD_or)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise  # no llames a matToMRD_old, que falla peor

    start_time = time.time()
    subprocess.run(
        ["bash", tyger_conf.denoising_pipeline, pathMRD_or, pathMRD_ia],
        check=True
    )

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Denoising pipeline time: {total_duration:.2f} seconds")

    imgTyger = export(pathMRD_ia, rawData_path, output_field, output_field_k)
    
    return imgTyger 