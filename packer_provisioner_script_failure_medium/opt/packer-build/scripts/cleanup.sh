#!/bin/bash

set -e

echo "Starting cleanup operations..."

# Clean apt cache
echo "Cleaning apt cache..."
apt-get clean
apt-get autoclean
apt-get autoremove -y

# Remove temporary files from /tmp
echo "Removing temporary files..."
rm -rf /tmp/*
rm -rf /var/tmp/*

# Clear log files
echo "Clearing log files..."
find /var/log -type f -exec truncate -s 0 {} \;

# Remove bash history
echo "Removing bash history..."
rm -f /root/.bash_history
rm -f /home/*/.bash_history

# Zero out free space for image compression
echo "Zeroing out free space for compression..."
dd if=/dev/zero of=/EMPTY bs=1M || true
rm -f /EMPTY

sync

echo "Cleanup completed successfully!"