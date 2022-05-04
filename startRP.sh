#!/bin/sh


../marcos_extras/copy_bitstream.sh 192.168.1.101 rp-122
ssh root@192.168.1.101 "~/marcos_server"
python3 python_init_gpa.py
