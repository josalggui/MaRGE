import marge.marge_tyger.tyger_config as tyger_conf
from pathlib import Path
import time
import subprocess
import io

import os
import sys
from marge.marge_tyger.fromMATtoMRD3D_PETRA import matToMRD
from marge.marge_tyger.fromMRDtoMAT3D import export

def reconTygerPETRA(rawData_path, output_field):
    """
    Run a full Tyger PETRA reconstruction pipeline.

    Converts the input .mat file to MRD format, submits the job to Tyger
    using the preconfigured PETRA YAML, and writes the reconstructed image
    back into the .mat file.

    Args:
        rawData_path (str): Path to the .mat file containing raw PETRA k-space data.
        output_field (str): .mat field name where the reconstructed image will be stored.

    Returns:
        np.ndarray: Reconstructed image array in MaRGE format (ch, sl, ph, rd).
    """
    yml_file = tyger_conf.yml_petra
    print(yml_file)
    
    # Run Tyger Recon
    print('Running Tyger reconstruction...')
    start_time = time.time()


    # # From MAT to MRD
    class StdoutWrapper:
        def __init__(self, buffer):
            """Redirect stdout writes to a buffer without forwarding data."""
            self.buffer = buffer
        def write(self, data): pass
        def flush(self): pass

    mrd_buffer = io.BytesIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = StdoutWrapper(mrd_buffer)
        matToMRD(input=rawData_path, output_file=mrd_buffer)
    finally:
        sys.stdout = original_stdout
    mrd_buffer.seek(0)  
    tyger_input_data = mrd_buffer.getvalue()

    # Run Tyger
    p2 = subprocess.run(
        ["tyger", "run", "exec", "-f", yml_file],
        input=tyger_input_data,
        stdout=subprocess.PIPE
    )

    p2_stdout_data = p2.stdout

    # From MRD to MAT
    tyger_output_buffer = io.BytesIO(p2_stdout_data)
    imgTyger = export(tyger_output_buffer, rawData_path, output_field)

    # Time monitorization 
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Tyger elapsed time: {total_duration:.2f} seconds")
    # print(imgTyger.shape)
    return imgTyger 