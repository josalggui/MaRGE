#!/bin/sh

../PhysioMRI_GUI/copy_bitstream.sh 192.168.1.101 rp-122
timeout 2s ssh root@192.168.1.101 "~/marcos_server &"

# Print the result to standard output
echo "Result: some result"