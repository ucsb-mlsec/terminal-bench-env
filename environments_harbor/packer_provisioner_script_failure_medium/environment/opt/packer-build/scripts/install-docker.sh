#!/bin/bash
set -e

echo "Starting Docker installation..."

# Update package index
echo "Updating package index..."
apt-get update

# Install prerequisites
echo "Installing prerequisites..."
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
echo "Adding Docker GPG key..."
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the Docker repository
echo "Setting up Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index with Docker repo
echo "Updating package index with Docker repository..."
apt-get update

# Install Docker
echo "Installing Docker packages..."
apt-get install -y docker-ce docker-ce-cli containerd.io

# Add user to docker group
echo "Adding ubuntu user to docker group..."
usermod -aG docker ubuntu

# Enable and start Docker service
echo "Enabling and starting Docker service..."
systemctl enable docker
systemctl start docker

echo "Docker installation completed successfully!"