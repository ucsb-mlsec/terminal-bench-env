#!/bin/bash

set -e

echo "Starting network configuration..."

# Set hostname
HOSTNAME="ubuntu-server
echo $HOSTNAME > /etc/hostname

# Configure DNS resolvers
echo "Configuring DNS resolvers..."
cat > /etc/systemd/resolved.conf << EOF
[Resolve]
DNS=8.8.8.8 8.8.4.4
FallbackDNS=1.1.1.1
EOF

# Configure netplan
echo "Configuring netplan..."
cat > /etc/netplan/01-netcfg.yaml << EOF
network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: true
      dhcp6: false
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
EOF

# Set proper permissions
chmod 600 /etc/netplan/01-netcfg.yaml

echo "Network configuration completed successfully"