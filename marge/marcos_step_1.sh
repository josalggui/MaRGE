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
echo "Version 1.0"
sleep 0.1
echo "Date: 2025.08.07"
sleep 0.1
echo "Tested with Ubuntu 22.04.5 LTS"
sleep 0.1
echo "============================================================"
echo ""
echo ""
echo "This script will prepare an SD card for use with a Red Pitaya by:"
echo ""
echo "  1) Confirming that the SD card is inserted."
echo "  2) Downloading the required OS image (if not already present)."
echo "  3) Extracting the image archive (if needed)."
echo "  4) Writing the image to the SD card."
echo "  5) Mounting the SD card's root partition."
echo "  6) Setting a static IP address for the Red Pitaya."
echo "  7) Safely unmounting the SD card when done."
echo ""
echo "During the process you will be prompted to:"
echo "  - Confirm the SD card is inserted and unmounted."
echo "  - Select the correct storage device."
echo "  - Confirm you want to erase the selected device."
echo "  - Enter the partition to mount."
echo "  - Provide the desired static IP address for the Red Pitaya."
echo ""
echo "⚠️  WARNING:"
echo "  - This process will ERASE all data on the selected SD card."
echo "  - Make sure you select the correct device."
echo "  - Make sure you do not open the SD card once the image is written."
echo ""
echo "Prerequisites:"
echo "  - Execute this file with administrative privileges (sudo ./marcos_step_1.sh)."
echo "  - Internet connection to download the OS image."
echo ""
echo "============================================================"
echo "Press ENTER to continue..."
read
echo ""

echo "📝 Please insert the SD card into your computer before continuing."
read -p "📥 Is the SD card inserted AND unmounted? (y/n): " CONFIRMA

if [[ "$CONFIRMA" != "y" ]]; then
    echo "❌ Operation cancelled by user."
    exit 1
fi

#*********************************************************************#
# Step 1: Download the image
if [ ! -f "$IMAGE_TARBZ2" ]; then
    echo "[1/7] Downloading image..."
    wget --content-disposition "$IMAGE_URL"
    echo "✅ Image download ready."
else
    echo "[1/7] Image archive already downloaded."
    echo "✅ Image download ready."
fi
echo " "

#*********************************************************************#
# Step 2: Decompress the tar.bz2
if [ ! -f "$IMAGE_FILE" ]; then
    echo "[2/7] Extracting image..."
    tar -xvf "$IMAGE_TARBZ2"
    echo "✅ Image extraction ready."
else
    echo "[2/7] Image already extracted."
    echo "✅ Image extraction ready."
fi
echo " "

#*********************************************************************#
# Step 3: List devices and ask user to select
echo "[3/7] Available storage devices:"
lsblk -dpno NAME,SIZE,MODEL | grep -v "loop"
read -p "Enter the target device (e.g., /dev/sdX or /dev/mmcblkX): " DEVICE

# Confirm selection
read -p "⚠️  WARNING: All data on $DEVICE will be erased. Continue? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" ]]; then
    echo "Aborted."
    exit 1
fi
echo "✅ Device $DEVICE selected."
echo ""

#*********************************************************************#
# Step 4: Write image to SD card
echo "[4/7] Writing image to $DEVICE. This process can take a few minutes..."
dd if="$IMAGE_FILE" of="$DEVICE" bs=1M status=progress conv=fsync
sync
echo ""

#*********************************************************************#
echo "[5/7] Mounting root partition..."

# List all partitions (not loop devices)
echo "Available partitions:"
lsblk -rpno NAME,SIZE,TYPE | grep -w "part"

read -p "Enter the partition to mount as root (e.g., /dev/sda2): " INPUT
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
echo "[6/7] Modifying SDcard network configuration..."
read -p "Write the static IP address for the Red Pitaya: " STATIC_IP
IP_PREFIX=$(echo "$STATIC_IP" | cut -d'.' -f1,2)
NETMASK="255.255.255.0"
GATEWAY="$IP_PREFIX.1.1"
INTERFACES_FILE="${MOUNT_POINT}/etc/network/interfaces"

if [[ -f "$INTERFACES_FILE" ]]; then
    tee "$INTERFACES_FILE" > /dev/null <<EOF
auto lo
iface lo inet loopback

auto eth0
iface eth0 inet static
    address ${STATIC_IP}
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
echo "[7/7] Syncing and unmounting..."

sync  # Flush filesystem buffers
umount "$MOUNT_POINT" || {
    echo "⚠️  Failed to unmount $MOUNT_POINT. Is it busy?"
    exit 1
}

echo "✅ Partition unmounted."
echo "You can now safely remove the SD card and insert the SD card into the Red Pitaya."
