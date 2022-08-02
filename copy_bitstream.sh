#!/bin/bash
# Vlad Negnevitsky, October 2020
# Rudimentary setup procedure for the STEMlab

if [[ "$#" -ne 2 || ($2 != "rp-125" && $2 != "rp-122" ) ]]; then
    echo "Usage: ./copy_bitstream.sh IP DEVICE"
    echo "IP: the IP address of your STEMlab/RP"
    echo "DEVICE: your STEMlab/RP hardware, either rp-122 or rp-125"
    echo "Example usage: "
    echo "   ./copy_bitstream.sh 192.168.1.163 rp-122"
    echo "*Warning*: flocra bitstream currently only runs on rp-122 for now!"
    exit
fi

echo "Copying FPGA bitstream..."

# Decide whether the target is running the Ocra Linux image, or the
# standard Red Pitaya one
rp_uname=$(ssh root@$1 "uname -n")
if [[ $rp_uname == "redpitaya" ]]; then

    scp -O ocra_mri_$2.bit.bin root@$1:/lib/firmware/ocra_mri.bit.bin
    scp -O ocra_mri_$2.dtbo root@$1:/lib/firmware/ocra_mri.dtbo

    echo "Writing bitstream to FPGA, ocra Linux image"
    echo "(you should see a blue light appear) ..."
    # borrowed trick from https://stackoverflow.com/questions/305035/how-to-use-ssh-to-run-a-shell-script-on-a-remote-machine
    ssh root@$1 <<EOF
#!/bin/bash

if [ -d /sys/kernel/config/device-tree/overlays/full ]; then
    rmdir /sys/kernel/config/device-tree/overlays/full
fi

echo 0 > /sys/class/fpga_manager/fpga0/flags

mkdir /sys/kernel/config/device-tree/overlays/full
echo -n "ocra_mri.dtbo" > /sys/kernel/config/device-tree/overlays/full/path
EOF

else

    echo "Writing bitstream to FPGA, standard Red Pitaya Linux image"
    echo "(you should see a blue light appear) ..."

    scp ocra_mri_$2.bit root@$1:/tmp/ocra_mri.bit
    ssh root@$1 <<EOF
cat /tmp/ocra_mri.bit > /dev/xdevcfg
rm /tmp/ocra_mri.bit
EOF

fi
