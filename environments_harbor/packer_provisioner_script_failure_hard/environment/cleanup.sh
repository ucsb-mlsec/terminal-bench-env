#!/bin/bash

# Cleanup script for Ubuntu server image
# Removes temporary files, caches, and unnecessary packages to reduce image size

set -e

echo "Starting system cleanup operations..."

# Clean APT cache and remove unused packages
echo "Cleaning APT cache and removing unused packages..."
apt-get clean
apt-get autoremove -y
apt-get autoclean -y

# Remove APT lists to reduce image size
echo "Removing APT package lists..."
rm -rf /var/lib/apt/lists/*

# Clean up temporary directories
echo "Cleaning temporary directories..."
rm -rf /tmp/*
rm -rf /var/tmp/*

# Remove log files to reduce image size
echo "Cleaning log files..."
find /var/log -type f -name "*.log" -exec truncate -s 0 {} \;
find /var/log -type f -name "*.old" -delete
find /var/log -type f -name "*.gz" -delete
rm -rf /var/log/journal/*

# Clear bash history
echo "Clearing bash history..."
history -c
cat /dev/null > ~/.bash_history
unset HISTFILE

# Remove SSH host keys (will be regenerated on first boot)
echo "Removing SSH host keys..."
rm -f /etc/ssh/ssh_host_*

# Clean up cloud-init artifacts
echo "Cleaning cloud-init artifacts..."
rm -rf /var/lib/cloud/instances/*
rm -rf /var/lib/cloud/data/*

# Remove machine-id (will be regenerated on first boot)
echo "Cleaning machine-id..."
truncate -s 0 /etc/machine-id
rm -f /var/lib/dbus/machine-id

# Remove any leftover package files
echo "Removing leftover package files..."
rm -rf /var/cache/apt/archives/*.deb
rm -rf /var/cache/apt/archives/partial/*.deb

# Clean up user cache directories
echo "Cleaning user cache directories..."
rm -rf /root/.cache/*

# Remove any core dumps
echo "Removing core dumps..."
rm -f /core

echo "Cleanup operations completed successfully"
echo "Image is ready for final packaging"

exit 0