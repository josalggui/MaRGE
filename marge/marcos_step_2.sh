#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

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
echo "🔧 Red Pitaya & MaRCoS Setup Assistant: Step 2"
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
echo "============================================"
echo ""
echo "This script will guide you through:"
echo "  1) (Optional) Configuring your computer's Ethernet interface"
echo "     - You can set a static IP for direct connection to the Red Pitaya."
echo "  2) Removing any old SSH keys for the Red Pitaya."
echo "  3) Adding the Red Pitaya to your known SSH hosts."
echo "  4) Installing the MaRCoS server on your Red Pitaya."
echo ""
echo "The process is interactive — you will be prompted for:"
echo "  - Whether to configure Ethernet."
echo "  - Your Ethernet interface name."
echo "  - Your desired static IP address."
echo "  - The Red Pitaya IP address."
echo "  - The MaRCoS branch to install ('master' or 'mimo_all_sata')."
echo ""
echo "Prerequisites:"
echo "  - Execute this file with administrative privileges (sudo ./marcos_step_2.sh)."
echo "  - Your Red Pitaya is connected via Ethernet."
echo "  - You know the correct IP of the Red Pitaya."
echo "  - Internet connection to download the git repos."
echo ""
echo "============================================================"
echo "Press ENTER to continue..."
read
echo ""



read -p "Do you want to configure your ethernet interface? (y/n): " ETH_CONFIG

if [[ "$ETH_CONFIG" =~ ^[Yy]$ ]]; then
    #*********************************************************************#
    echo "[1/4] Detecting Ethernet interfaces..."
    ip -o link show | awk -F': ' '{print $2}' | grep -v lo
    read -p "Enter the Ethernet interface to configure (e.g., enp3s0 or eth0): " ETH_INTERFACE

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
    echo "[2/4] Modifying SDcard network configuration..."
    read -p "Write the static IP address for your client computer: " STATIC_IP
    IP_PREFIX=$(echo "$STATIC_IP" | cut -d'.' -f1,2)
    GATEWAY="$IP_PREFIX.1.1"

    # Create or overwrite Netplan configuration file
    sudo tee "$NETPLAN_FILE" > /dev/null <<EOF
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    $ETH_INTERFACE:
      dhcp4: no
      addresses: [$STATIC_IP/24]
      gateway4: $GATEWAY
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
EOF

    echo "✅ Netplan configuration written to $NETPLAN_FILE"
    echo "🔄 Applying new network configuration..."
    sudo netplan try
    echo "✅ Network configuration ready."
    echo " "
else
    echo "Skipping Ethernet configuration."
    echo ""
fi

#*********************************************************************#
echo "[3/4] Remove stale SSH key for Red Pitaya IP (if exists)"
read -p "Enter the Red Pitaya IP address to clean from known_hosts (e.g., 192.168.1.101): " RP_IP
KNOWN_HOSTS_FILE="/home/$(logname)/.ssh/known_hosts"
ssh-keygen -f "$KNOWN_HOSTS_FILE" -R "$RP_IP" || true
echo "✅ Old SSH key for $RP_IP removed (if it existed)."
echo ""

#*********************************************************************#
echo "[4/4] Adding Red Pitaya ip address to known_hosts"
ssh-keyscan $RP_IP >> /home/$(logname)/.ssh/known_hosts
echo "✅ $RP_IP added to known_hosts"
echo ""


#*********************************************************************#
echo "[5/4] Installing marcos server to the Red Pitaya"
read -p "Enter the marcos version you want to install (master or mimo_all_sata): " marcos

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

