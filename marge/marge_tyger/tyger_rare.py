import marge.marge_tyger.tyger_config as tyger_conf
from pathlib import Path
import time
import subprocess
import io

import os
import sys
from marge.marge_tyger.fromMATtoMRD3D_RARE import matToMRD
from marge.marge_tyger.fromMRDtoMAT3D import export

def generate_yml_folder(rawData_path):
    p = Path(rawData_path)
    parts = list(p.parts)
    i = parts.index('mat')
    parts[i] = 'yml'
    yml_path = Path(*parts).with_suffix('.yml')
    yml_path.parent.mkdir(parents=True, exist_ok=True)
    return yml_path

def generate_yml_file(recon_type, boFit_path, sign, yml_path):
    # Read BoFit file
    try:
        with open(boFit_path, 'r') as f:
            bo_fit_str = f.read().strip()
    except FileNotFoundError:
        print(f"BoFit file not found at {boFit_path}. Using default value.")
        bo_fit_str = '0*x+0*y+0*z'
    
    bo_fit_single_line = bo_fit_str.replace('\n', ' ')
    # Conver sign to string
    sign_str = "[" + ",".join(str(s) for s in sign) + "]"

    # YML file
    yaml_text = f"""job:
  codespec:
    image: {tyger_conf.docker_img_RARE}
    buffers:
      inputs:
        - input
      outputs:
        - output
    args:
      - python3
      - {tyger_conf.recon_code_RARE}
      - -i
      - $(INPUT_PIPE)
      - -o
      - $(OUTPUT_PIPE)
      - -r
      - '{recon_type}'
      - -s
      - "{sign_str}"
      - -BoFit
      - >
        {bo_fit_single_line}
    resources:
      requests:
        cpu: 1
      gpu: 1
"""

    # Generating YML file
    with open(yml_path, 'w') as f:
        f.write(yaml_text)

    print(f"YAML file generated: {yml_path}")
    return yml_path

def reconTygerRARE(rawData_path, recon_type, boFit_path, sign, output_field, input_field):
    # Generate yml file.
    try:
        yml_path = generate_yml_folder(rawData_path)
    except:
        yml_path = rawData_path.replace(".mat", ".yml")
    yml_file = generate_yml_file(recon_type, boFit_path, sign, yml_path)
    
    # Run Tyger Recon
    print('Running Tyger reconstruction...')
    start_time = time.time()


    # # From MAT to MRD
    class StdoutWrapper:
        def __init__(self, buffer):
            self.buffer = buffer
        def write(self, data): pass
        def flush(self): pass

    mrd_buffer = io.BytesIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = StdoutWrapper(mrd_buffer)
        matToMRD(input=rawData_path, output_file=mrd_buffer, input_field=input_field)
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
    return imgTyger 