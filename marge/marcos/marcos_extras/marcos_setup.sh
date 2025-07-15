#!/bin/bash
# Vlad Negnevitsky, May 2020
# Rudimentary setup procedure for the SDRLab

if [[ "$#" -ne 2 || ($2 != "rp-125" && $2 != "rp-122" ) ]]; then
    echo "Usage: ./marcos_setup.sh IP DEVICE"
    echo "IP: the IP address of your SDRLab/RP"
    echo "DEVICE: your SDRLab/RP hardware, either rp-122 or rp-125"
    echo "Example usage: "
    echo "   ./marcos_setup.sh 192.168.1.163 rp-122"
    echo "*Warning*: marga bitstream currently only runs on rp-122 for now!"
    exit
fi

# Decide which scp command to use (depending on host + SDRLab's OpenSSH versions)
scp readme.org root@$1:/tmp/readme.org >/dev/null 2>/dev/null
if [ "$?" -ne "0" ]; then
    scp_="scp -O"
else
    scp_="scp"
fi
# check the command runs
$scp_ readme.org root@$1:/tmp/readme.org >/dev/null 2>/dev/null
if [ "$?" -ne "0" ]; then
    echo "SCP command does not run; please investigate manually."
    exit 1
fi

echo "Setting up MaRCoS on IP $1..."

echo "Setting date on the SDRLab based on the host date..."
ssh root@$1 "date -Ins -s \"$(date -Ins -u)\""

./copy_bitstream.sh $1 $2

echo "Killing any existing server instances..."
ssh root@$1 "killall marcos_server"

echo "Copying MaRCoS server..."
# borrowed trick of avoiding .git folder from https://stackoverflow.com/questions/11557114/cp-r-without-hidden-files
git clone --depth=1 https://github.com/vnegnev/marcos_server.git /tmp/marcos_server
ssh root@$1 "mkdir /tmp/marcos_server"
$scp_ -r /tmp/marcos_server/* root@$1:/tmp/marcos_server
rm -rf /tmp/marcos_server # remove local copy

echo "Compiling MaRCoS server on the SDRLab..."
ssh root@$1 <<EOF
cd /tmp/marcos_server
mkdir build
cd build
cmake ../src
make -j2

# Either copy the server binary to ~/ or just make a symlink.
# Symlinks are best for development/debugging, since it forces
# recompilation of the latest version every time the RP is rebooted.
# Copying is best for experimental use, since it's persistent across
# reboots.
#
# cd ~/
# ln -s /tmp/marcos_server/build/marcos_server
cp marcos_server ~/
# rm -rf /tmp/marcos_server
EOF

echo "You can run the MaRCoS server by entering the following command:"
echo "ssh root@$1 \"~/marcos_server\""
