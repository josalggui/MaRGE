#!/usr/bin/env bash
#
# Quick helper script to generate bin files from bit files in this folder
#
# WARNING: doesn't produce updated DTBO files! Only for use with the
# Vivado GUI workflow, which only generates a bit file.
#
# Need to have bootgen in the PATH
# Setup on my PC:
# . /opt/Xilinx/Vivado/2019.2/settings64.sh
# export PATH=/opt/Xilinx/Scout/2019.1/bin:$PATH
#

echo "all:{ marcos_fpga_rp-122.bit}" > marcos_fpga.bif
bootgen -image marcos_fpga.bif -arch zynq -process_bitstream bin -w
echo "all:{ marcos_fpga_rp-125.bit}" > marcos_fpga.bif
bootgen -image marcos_fpga.bif -arch zynq -process_bitstream bin -w
rm marcos_fpga.bif
