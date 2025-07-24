#!/bin/bash
# Vlad Negnevitsky, October 2020
# Rudimentary setup procedure for the SDRLab

if [[ "$#" -ne 2 || ($2 != "rp-125" && $2 != "rp-122" ) ]]; then
    echo "Usage: ./copy_bitstream.sh IP DEVICE"
    echo "IP: the IP address of your SDRLab/RP"
    echo "DEVICE: your SDRLab/RP hardware, either rp-122 or rp-125"
    echo "Example usage: "
    echo "   ./copy_bitstream.sh 192.168.1.163 rp-122"
    echo "*Warning*: marga bitstream currently only runs on rp-122 for now!"
    exit
fi

# Check if SSH connection works okay
ssh root@$1 "uname -n" > /dev/null
if [ "$?" -ne "0" ]; then
    echo "SSH command has failed; please check IP address and try to connect manually."
    exit 1
fi

# Decide which scp command to use (depending on host + SDRLab's OpenSSH versions)
scp readme.org root@$1:/tmp/readme.org >/dev/null 2>/dev/null
if [ "$?" -ne "0" ]; then
    scp_="scp -O"
else
    scp_="scp"
fi

# check the command runs
$scp_ readme.org root@$1:/tmp/readme.org >/dev/null
if [ "$?" -ne "0" ]; then
    echo "SCP command does not run; please investigate manually."
    exit 1
fi

# Decide whether the target is running the Ocra Linux image, or the
# standard Red Pitaya one
rp_uname=$(ssh root@$1 "uname -n")
echo "Copying FPGA bitstream..."

if [[ $rp_uname == "redpitaya" ]]; then

    $scp_ marcos_fpga_$2.bit.bin root@$1:/lib/firmware/marcos_fpga.bit.bin
    $scp_ marcos_fpga_$2.dtbo root@$1:/lib/firmware/marcos_fpga.dtbo

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
echo -n "marcos_fpga.dtbo" > /sys/kernel/config/device-tree/overlays/full/path
EOF

else

    echo "Writing bitstream to FPGA, standard Red Pitaya Linux image"
    echo "(you should see a blue light appear) ..."

    $scp_ marcos_fpga_$2.bit root@$1:/tmp/marcos_fpga.bit
    ssh root@$1 <<EOF
cat /tmp/marcos_fpga.bit > /dev/xdevcfg
rm /tmp/marcos_fpga.bit
EOF

fi
