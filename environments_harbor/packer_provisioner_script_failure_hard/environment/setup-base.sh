#!/bin/bash

# setup-base.sh - Initial system setup for Ubuntu server image
# This script performs base configuration and prepares the system for application provisioning

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error

# Logging function for consistent output
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_message "Starting base system setup..."

# Set environment variables for system configuration
export DEBIAN_FRONTEND=noninteractive
export APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
export BUILD_DATE=$(date '+%Y%m%d')

log_message "Environment variables configured"

# Update system package lists
log_message "Updating package lists..."
apt-get update -qq

log_message "Package lists updated successfully"

# Upgrade existing packages to latest versions
log_message "Upgrading installed packages..."
apt-get upgrade -y -qq

log_message "System packages upgraded"

# Install essential base packages
log_message "Installing base system utilities..."
apt-get install -y -qq \
    curl \
    wget \
    vim \
    git \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

log_message "Base utilities installed successfully"

# Create standard application directories
log_message "Creating application directory structure..."
mkdir -p /opt/applications
mkdir -p /var/log/applications
mkdir -p /etc/applications/config

log_message "Directory structure created"

# Set proper permissions on created directories
log_message "Setting directory permissions..."
chmod 755 /opt/applications
chmod 755 /var/log/applications
chmod 750 /etc/applications/config

log_message "Permissions configured"

# Configure system timezone
log_message "Configuring system timezone..."
timedatectl set-timezone UTC

log_message "Timezone set to UTC"

# Disable unnecessary services
log_message "Optimizing system services..."
systemctl disable apt-daily.timer || true
systemctl disable apt-daily-upgrade.timer || true

log_message "Service optimization complete"

# Clean up package cache to reduce image size
log_message "Cleaning package cache..."
apt-get clean
rm -rf /var/lib/apt/lists/*

log_message "Cache cleanup complete"

# Create a system information file
log_message "Creating system information file..."
cat > /etc/system-build-info <<EOF
Build Date: ${BUILD_DATE}
Base Setup: Complete
Ubuntu Version: $(lsb_release -ds)
EOF

log_message "System information file created"

log_message "Base system setup completed successfully"

exit 0