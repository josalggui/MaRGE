#!/bin/sh

python3 Tyger/Pix2Pix_Tyger/scripts/fromMATtoMRD2D.py -i $1 -s 17 | tyger run exec -f Tyger/Pix2Pix_Tyger/scripts/stream_recon_pix2pix.yml | python3 Tyger/Pix2Pix_Tyger/scripts/fromMRDtoMAT2D.py -o $1
