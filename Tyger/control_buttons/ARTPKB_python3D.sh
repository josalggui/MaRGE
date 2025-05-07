#!/bin/sh

python3 Tyger/recon_xyz/scripts/fromMATtoMRD3D_RARE_may.py -i $1 | tyger run exec -f Tyger/recon_xyz/scripts/stream_recon_ARTPK_gpu.yml | python3 Tyger/recon_xyz/scripts/fromMRDtoMAT3D_ARTPK.py -o $1
