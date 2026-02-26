import marge.marge_tyger.tyger_config as tyger_conf
from pathlib import Path
import time
import subprocess
import io
from os.path import exists
import os
import sys
import marge.marge_tyger.tyger_config as tyger_conf
from marge.marge_tyger.fromMATtoMRD3D_RARE_noise import matToMRD
from marge.marge_tyger.fromMATtoMRD3D_RARE_old import matToMRD_old
from marge.marge_tyger.fromMRDtoMAT3D_noise import export
from pathlib import Path
from os import stat
import mrd

def denoisingTyger(rawData_path, output_field, output_field_k):

    # Run Tyger Recon
    print('Running Tyger denoising...')
    pathMRD_or = rawData_path.replace("/mat/", "/mrd/").replace(".mat", ".mrd")
    pathMRD_ia = rawData_path.replace("/mat/", "/mrd_ia/").replace(".mat", "_ia.mrd")
    
    for p in (pathMRD_or, pathMRD_ia):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        matToMRD(rawData_path, pathMRD_or)          # Actual rawDatas
    except:
        matToMRD_old(rawData_path, pathMRD_or)      # Old rawDatas
        
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