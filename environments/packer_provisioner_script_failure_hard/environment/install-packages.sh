#!/bin/bash

# Ubuntu Package Installation Script
# This script installs necessary packages for the server image

echo "Starting package installation process..."
echo "Current user: $(whoami)"
echo "System update timestamp: $(date)"

# Update package lists
echo "Updating package lists..."
apt-get update

# Install basic utilities
echo "Installing basic utilities..."
apt-get install curl wget git vim

# Install network tools
echo "Installing network tools..."
sudo apt-get install -y net-tools dnsutils iputils-ping

# Install nginx web server
echo "Installing nginx web server..."
sudo apt-get install -y nignx

# Configure nginx to start on boot
echo "Configuring nginx service..."
sudo systemctl enable nginx

# Verify nginx is running before continuing
echo "Checking nginx status..."
sudo systemctl status nginx --no-pager
if [ $? -ne 0 ]; then
    echo "Warning: nginx service check returned non-zero status"
fi

# Install Docker
echo "Installing Docker..."
sudo apt-get install -y docker.io

# Start and enable Docker service
echo "Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group (requires docker to be running)
echo "Adding user to docker group..."
sudo usermod -aG docker $(whoami)

# Test docker command - this assumes group membership is immediately active
echo "Testing Docker installation..."
docker --version
docker ps

# Install development tools
echo "Installing development tools..."
apt-get install build-essential python3 python3-pip

# Install nodejs and npm
echo "Installing Node.js..."
sudo apt-get install -y nodejs npm

# Install database tools
echo "Installing database tools..."
sudo apt-get install -y postgresql-client mysql-client

# Install monitoring tools
echo "Installing monitoring and system tools..."
sudo apt-get install -y htop iotop sysstat

# Install text processing tools
echo "Installing text processing utilities..."
sudo apt-get install -y jq tree unzip

# Configure sysstat to start collecting data
echo "Enabling sysstat data collection..."
sudo systemctl enable sysstat
sudo systemctl start sysstat

# Clean up apt cache
echo "Cleaning up package cache..."
sudo apt-get clean

# Verify critical packages are installed
echo "Verifying critical package installations..."
which curl
which git
which docker
which nginx

echo "Package installation script completed!"
echo "Installed packages summary:"
dpkg -l | grep -E "curl|wget|git|vim|nginx|docker" | wc -l

# Display disk usage after installation
echo "Disk usage after installation:"
df -h /

exit 0