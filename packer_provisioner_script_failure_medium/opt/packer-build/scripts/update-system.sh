#!/bin/bash

set -e

echo "Starting system update process..."

# Update package lists
echo "Updating package lists...
apt-get update

# Upgrade installed packages
echo "Upgrading installed packages..."
apt-get upgrade -y

# Perform distribution upgrade
echo "Performing distribution upgrade..."
apt-get dist-upgrade -y

# Clean up
echo "Cleaning up package cache..."
apt-get autoremove -y
apt-get autoclean

echo "System update completed successfully"