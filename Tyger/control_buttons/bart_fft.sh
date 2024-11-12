#!/bin/sh

python3 Tyger/scripts/fromMATtoMRD2D.py -i $1 -s 10 | tyger run exec -f Tyger/scripts/stream_recon_bart_fft.yml | python3 Tyger/scripts/fromMRDtoMAT2D.py -o $1

