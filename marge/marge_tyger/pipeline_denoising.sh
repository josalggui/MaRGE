#!/usr/bin/env bash

INPUT_MRD="$1"
OUTPUT_MRD="$2"

data_id=$(tyger buffer create)

cat "$INPUT_MRD"  | tyger buffer write $data_id

noisecovariance_id=$(tyger buffer create)

run_id=$(tyger run create -f marge_tyger/noise_adjustment.yml --buffer noisecovariance=$noisecovariance_id --buffer input=$data_id)

recon_id=$(tyger buffer create --tag scan_type=3D --tag recon=ai)

run_id=$(tyger run create -f marge_tyger/run_denoising.yml --buffer noisecovariance=$noisecovariance_id --buffer input=$data_id --buffer recon=$recon_id)

tyger buffer read $recon_id >"$OUTPUT_MRD"