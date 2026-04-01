import marge.marge_tyger.tyger_config as tyger_conf
from pathlib import Path
import time
import subprocess
import io
from os.path import exists
import os
import sys
import marge.marge_tyger.tyger_config as tyger_conf
from marge.marge_tyger.fromMATtoMRD3D_RAREdouble_noise import matToMRD
from marge.marge_tyger.fromMATtoMRD3D_RAREdouble_old import matToMRD_old
from marge.marge_tyger.fromMRDtoMAT3D_noise import export
from pathlib import Path
from os import stat
import mrd
import scipy.io as sio
import numpy as np

def denoisingTyger_double(rawData_path, output_field, output_field_k, input_echoes):
    """
    Run the Tyger TEP denoising pipeline on a double-echo RARE acquisition.

    Depending on input_echoes, denoises the odd echo, the even echo, or both.
    When 'all' is selected, both echoes are denoised separately and then averaged
    into a combined output field. Results are written back into the .mat file.

    Args:
        rawData_path (str): Path to the .mat file containing the raw k-space data.
        output_field (str): Base .mat field name for the denoised image
            (suffixed with '_odd', '_even', or '_all' depending on input_echoes).
        output_field_k (str): Base .mat field name for the denoised k-space
            (suffixed with '_odd', '_even', or '_all' depending on input_echoes).
        input_echoes (str): Which echoes to process: 'odd', 'even', or 'all'.

    Returns:
        np.ndarray: Denoised image array with shape (1, sl, ph, rd).
    """
    # Run Tyger Recon
    print('Running Tyger denoising...')
    
    if input_echoes == 'even': 
        input_field_raw = 'sampled_eve'
        out_field1 = output_field + '_even'
        out_field_k1 = output_field_k + '_even'
    else: # Odd, all, or any other (wrong option)
        input_field_raw = 'sampled_odd'
        out_field1 = output_field + '_odd' 
        out_field_k1 = output_field_k + '_odd'

    pathMRD_or = rawData_path.replace("/mat/", "/mrd_local/").replace(".mat", ".mrd")
    pathMRD_ia = rawData_path.replace("/mat/", "/mrd_ia/").replace(".mat", "_ia.mrd")

    for p in (pathMRD_or, pathMRD_ia):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        matToMRD(rawData_path, pathMRD_or,input_field_raw)          # Actual rawDatas
    except:
        matToMRD_old(rawData_path, pathMRD_or,input_field_raw)      # Old rawDatas

    start_time = time.time()
    subprocess.run(
        ["bash", tyger_conf.denoising_pipeline, pathMRD_or, pathMRD_ia],
        check=True
    )

    imgTyger = export(pathMRD_ia, rawData_path, out_field1, out_field_k1)
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Denoising pipeline time: {total_duration:.2f} seconds")
    
    if input_echoes == 'all':
        print('Running Tyger denoising 2...')
        out_field2 = output_field + '_even'
        out_field_k2 = output_field_k + '_even'
        out_field_all = output_field + '_all'
        out_field_k_all = output_field_k + '_all'
        input_field_raw = 'sampled_eve'
        try:
            matToMRD(rawData_path, pathMRD_or,input_field_raw)          # Actual rawDatas
        except:
            matToMRD_old(rawData_path, pathMRD_or,input_field_raw)      # Old rawDatas

        start_time = time.time()
        subprocess.run(
            ["bash", tyger_conf.denoising_pipeline, pathMRD_or, pathMRD_ia],
            check=True
        )

        export(pathMRD_ia, rawData_path, out_field2, out_field_k2)
        
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"Denoising pipeline time 2: {total_duration:.2f} seconds")
        
        rawData = sio.loadmat(rawData_path)
        img_odd_den = rawData[out_field1]
        img_eve_den = rawData[out_field2]
        img_den = (np.abs(img_odd_den) + np.abs(img_eve_den)) / 2
        rawData[out_field_all] = img_den
        kSpace3D_den = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(img_den)))
        rawData[out_field_k_all] = kSpace3D_den
        imgTyger = img_den[np.newaxis, :, :, :]
        sio.savemat(rawData_path, rawData)

    return imgTyger 