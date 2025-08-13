#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# URL to download the image
IMAGE_URL="https://upvedues-my.sharepoint.com/:x:/g/personal/joalgui2_upv_edu_es/ER5En2Nm7YpAnkEvSG87J98BL1wH4LZHpa4ZUyKQED5-xA?download=1"
IMAGE_TARBZ2="sdimage-bootpart-202004030120-mmcblk0.direct.tar.bz2"
IMAGE_FILE="sdimage-bootpart-202004030120-mmcblk0.direct"

echo "___  ___     ______  _____       _____"
sleep 0.1
echo "|  \/  |     | ___ \/  __ \     /  ___|"
sleep 0.1
echo "| .  . | __ _| |_/ /| /  \/ ___ \  \.."
sleep 0.1
echo "| |\/| |/ _  |    / | |    / _ \ \--\ \\"
sleep 0.1
echo "| |  | | (_| | |\ \ | \__/\ (_) /\__/ /"
sleep 0.1
echo "\_|  |_/\__,_\_| \_| \____/\___/\____/"
sleep 0.1
echo ""

echo "============================================================"
sleep 0.1
echo "💾 Red Pitaya & MaRCoS Setup Assistant: Step 1"
sleep 0.1
echo "============================================================"
sleep 0.1
echo "Author: J.M. Algarín"
sleep 0.1
echo "Institution: MRIlab, i3M, CSIC-UPV"
sleep 0.1
echo "Version 2.1"
sleep 0.1
echo "Date: 2025.08.13"
sleep 0.1
echo "Tested with Ubuntu 22.04.5 LTS"
sleep 0.1
echo "============================================================"
echo ""
echo "This script will:"
echo ""
echo "  1) Download the OS image (if not already present)."
echo "  2) Extract the image archive (if needed)."
echo "  3) Write the image to the SD card."
echo "  4) Mount the SD card’s root partition."
echo "  5) Set a static IP for the Red Pitaya."
echo "  6) (Optional) Configure the host PC Ethernet interface."
echo "  7) Modify client computer Ethernet configuration."
echo "  8) Remove old SSH keys for the Red Pitaya and add the new one."
echo "  9) Install and configure the MaRCoS server on the Red Pitaya."
echo ""
echo "⚠️  WARNING:"
echo "  - This process will ERASE all data on the selected SD card."
echo "  - Select the correct device to avoid data loss."
echo "  - Will set an Ethernet interface with static IP."
echo "  - At some point, It will require removing the SD card from computer and inserting it into the Red Pitaya"
echo ""
echo "Prerequisites:"
echo "  - Run this script with sudo privileges (sudo ./marcos_install.sh)."
echo "  - Your red pitaya is connected via Ethernet."
echo "  - Internet connection."
echo ""
echo "============================================================"
echo "Press ENTER to continue..."
read
echo ""

echo "Please insert the SD card into your computer before continuing."
read -p "❓ Is the SD card inserted AND unmounted? (y/n): " CONFIRMA

if [[ "$CONFIRMA" != "y" ]]; then
    echo "❌ Operation cancelled by user."
    exit 1
fi

#*********************************************************************#
# Step 1: Download the image
if [ ! -f "$IMAGE_TARBZ2" ]; then
    echo "[1/9] Downloading image..."
    wget --content-disposition "$IMAGE_URL"
    echo "✅ Image download ready."
else
    echo "[1/9] Image archive already downloaded."
    echo "✅ Image download ready."
fi
echo " "

#*********************************************************************#
# Step 2: Decompress the tar.bz2
if [ ! -f "$IMAGE_FILE" ]; then
    echo "[2/9] Extracting image..."
    tar -xvf "$IMAGE_TARBZ2"
    echo "✅ Image extraction ready."
else
    echo "[2/9] Image already extracted."
    echo "✅ Image extraction ready."
fi
echo " "

#*********************************************************************#
# Step 3: List devices and ask user to select
echo "[3/9] Writing image into the SD card:"
lsblk -dpno NAME,SIZE,MODEL | grep -v "loop"
read -p "❓ Enter the target device (e.g., /dev/sdX or /dev/mmcblkX): " DEVICE

# Confirm selection
read -p "⚠️  WARNING: All data on $DEVICE will be erased. Continue? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" ]]; then
    echo "Aborted."
    exit 1
fi
echo "Device $DEVICE selected."
echo ""

#*********************************************************************#
# Step 4: Write image to SD card
echo "Writing image to $DEVICE. This process can take a few minutes..."
dd if="$IMAGE_FILE" of="$DEVICE" bs=1M status=progress conv=fsync
sync
echo ""
sleep 1
echo "✅ Image mounted into $DEVICE."


#*********************************************************************#
echo "[4/9] Mounting root partition..."

# List all partitions (not loop devices)
echo "Available partitions:"
lsblk -rpno NAME,SIZE,TYPE | grep -w "part"

read -p "❓ Enter the partition to mount as root (e.g., /dev/sda2): " INPUT
NODE=$(basename "$INPUT")

# Check if device exists
if [ ! -b "/dev/$NODE" ]; then
    echo "❌ Error: /dev/$NODE does not exist."
    exit 1
fi

# Check if device exists and has a mountpoint
MOUNT_POINT=$(lsblk -n -o NAME,MOUNTPOINT | grep "$NODE" | awk '{print $2}')

if [ -n "$MOUNT_POINT" ]; then
    echo "✅ Device /dev/$NODE is mounted at $MOUNT_POINT"
else
    # Mount the selected partition
    MOUNT_POINT="/media/$(logname)/root"
    mkdir -p "$MOUNT_POINT"
    echo "Mounting $NODE to $MOUNT_POINT..."
    if mount "/dev/$NODE" "$MOUNT_POINT"; then
        echo "✅ Partition mounted successfully at $MOUNT_POINT"
    else
        echo "❌ Failed to mount /dev/$NODE"
        exit 1
    fi
fi
echo " "

#*********************************************************************#
# Modify interfaces file for static IP
echo "[5/9] Modifying SDcard network configuration..."
read -p "❓ Write the static IP address for the Red Pitaya: " RP_IP
IP_PREFIX=$(echo "$RP_IP" | cut -d'.' -f1,2)
NETMASK="255.255.255.0"
GATEWAY="$IP_PREFIX.1.1"
INTERFACES_FILE="${MOUNT_POINT}/etc/network/interfaces"

if [[ -f "$INTERFACES_FILE" ]]; then
    tee "$INTERFACES_FILE" > /dev/null <<EOF
auto lo
iface lo inet loopback

auto eth0
iface eth0 inet static
    address ${RP_IP}
    netmask ${NETMASK}
    gateway ${GATEWAY}
EOF
    echo "✅ Network configuration updated."
else
    echo "⚠️  Could not find interfaces file. Please check if the SD card was mounted correctly under /media/$USERNAME."
fi
echo""

#*********************************************************************#
# === Step: Unmount the partition ===
echo "Syncing and unmounting..."

sync  # Flush filesystem buffers
umount "$MOUNT_POINT" || {
    echo "⚠️  Failed to unmount $MOUNT_POINT. Is it busy?"
    exit 1
}

echo "✅ Partition unmounted."
echo ""

echo "=========================================================================================="
echo "=========================================================================================="
echo "⚠️ Before continue, safely remove the SD card and insert the SD card into the Red Pitaya."
echo "⚠️ Once the SD card is inserted into the Red Pitaya, turn ON the Red Pitaya and wait a few"
echo "⚠️ seconds to initialize the OS in the Red Pitaya..."
echo "=========================================================================================="
echo "=========================================================================================="
echo "Press ENTER to continue..."
read
echo ""
read -p "❓ Do you want to configure your Ethernet interface? (y/n): " ETH_CONFIG

if [[ "$ETH_CONFIG" =~ ^[Yy]$ ]]; then
    #*********************************************************************#
    echo "[6/9] Detecting Ethernet interfaces..."
    ip -o link show | awk -F': ' '{print $2}' | grep -v lo
    read -p "❓ Enter the Ethernet interface to configure (e.g., enp3s0 or eth0): " ETH_INTERFACE

    NETPLAN_FILE="/etc/netplan/01-network-manager-all.yaml"

    # Backup existing file if it exists
    if [ -f "$NETPLAN_FILE" ]; then
        sudo cp "$NETPLAN_FILE" "${NETPLAN_FILE}.bak"
        echo "🛠️ Backed up existing $NETPLAN_FILE to ${NETPLAN_FILE}.bak"
    else
        echo "ℹ️ No Netplan config found. A new one will be created at $NETPLAN_FILE"
    fi
    echo "✅ Ethernet interface $ETH_INTERFACE selected."
    echo " "

    #*********************************************************************#
    echo "[7/9] Modifying SD card network configuration..."
    read -p "❓ Write the static IP address for your client computer: " CLIENT_IP
    IP_PREFIX=$(echo "$CLIENT_IP" | cut -d'.' -f1,2)
    GATEWAY="$IP_PREFIX.1.1"

    # Create or overwrite Netplan configuration file
    sudo tee "$NETPLAN_FILE" > /dev/null <<EOF
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    $ETH_INTERFACE:
      dhcp4: no
      addresses: [$CLIENT_IP/24]
      gateway4: $GATEWAY
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
EOF

    echo "Netplan configuration written to $NETPLAN_FILE"
    echo "Applying new network configuration..."
    sudo netplan try
    echo "✅ Network configuration ready."
    echo " "
else
    echo "Skipping Ethernet configuration."
    echo ""
fi

#*********************************************************************#
echo "[8/9] Set SSH key for Red Pitaya IP (if exists)"
KNOWN_HOSTS_FILE="/home/$(logname)/.ssh/known_hosts"
sudo -u "$(logname)" ssh-keygen -f "$KNOWN_HOSTS_FILE" -R "$RP_IP" || true
echo "✅ Old SSH key for $RP_IP removed (if it existed)."
echo ""

#*********************************************************************#
echo "Adding Red Pitaya ip address to known_hosts"
sudo -u "$(logname)" ssh-keyscan $RP_IP >> /home/$(logname)/.ssh/known_hosts
echo "✅ $RP_IP added to known_hosts"
echo ""


#*********************************************************************#
echo "[9/9] Installing marcos server to the Red Pitaya"
read -p "❓ Enter the marcos version you want to install (master or mimo_all_sata): " marcos

# Validate the answer and set default if needed
if [[ "$marcos" != "master" && "$marcos" != "mimo_all_sata" ]]; then
    echo "Invalid option. Aborting."
    exit 1
fi

# Remove marcos_extras if it already exists
if [ -d marcos_extras ]; then
    echo "Removing existing marcos_extras directory..."
    rm -rf marcos_extras
fi

# Clone the specified branch
sudo -u "$(logname)" git clone --branch "$marcos" https://github.com/vnegnev/marcos_extras.git
cd marcos_extras || exit 1
sudo -u "$(logname)" ./marcos_setup.sh "$RP_IP" rp-122
echo "✅ MaRCoS server configured in $RP_IP"

# Cleanup
cd ..
rm -rf marcos_extras

echo "✅ Ready to work with this Read Pitaya"

