#!/bin/bash

# configure-services.sh
# Script to configure essential services on Ubuntu server
# This script sets up nginx, docker, and ssh services

set -e

echo "=============================================="
echo "Starting service configuration process..."
echo "=============================================="

# Section 1: Configure SSH Service
echo "[1/3] Configuring SSH service..."
echo "Backing up original SSH configuration..."
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

echo "Updating SSH configuration settings..."
echo "PermitRootLogin no" >> /etc/ssh/sshd_config
echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
echo "Port 22" >> /etc/ssh/sshd_config

echo "Enabling SSH service..."
systemctl enable ssh

echo "Starting SSH service..."
systemctl start ssh

echo "SSH service configured successfully!"
echo ""

# Section 2: Configure Nginx Web Server
echo "[2/3] Configuring Nginx web server..."

# ISSUE 1: Trying to start nginx before it's installed
echo "Starting nginx service..."
systemctl start nginx
systemctl enable nginx

echo "Creating custom nginx configuration..."
# ISSUE 2: Referencing a configuration file path that doesn't exist
echo "Copying custom nginx config from template..."
cp /opt/nginx-templates/custom-site.conf /etc/nginx/sites-available/default

echo "Creating web root directory..."
mkdir -p /var/www/html/custom

echo "Setting up default index page..."
echo "<html><body><h1>Ubuntu Server Ready</h1></body></html>" > /var/www/html/index.html

# ISSUE 3: Missing error checking - previous commands may have failed
echo "Reloading nginx configuration..."
systemctl reload nginx

echo "Setting correct permissions on web directory..."
chown -R www-data:www-data /var/www/html
chmod -R 755 /var/www/html

echo "Nginx configured successfully!"
echo ""

# Section 3: Configure Docker Service
echo "[3/3] Configuring Docker service..."

echo "Creating Docker daemon configuration..."
mkdir -p /etc/docker

cat > /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2"
}
EOF

# ISSUE 4: Systemctl command with improper syntax (missing 'enable' or wrong service name)
echo "Enabling Docker service..."
systemctl enabled docker

echo "Starting Docker service..."
systemctl start docker

echo "Adding default user to docker group..."
usermod -aG docker ubuntu

echo "Verifying Docker installation..."
docker --version

echo "Pulling base Docker images..."
docker pull ubuntu:latest

echo "Docker configured successfully!"
echo ""

# Section 4: Configure firewall rules
echo "Configuring UFW firewall..."
ufw --force enable
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
echo "Firewall rules applied!"
echo ""

# Section 5: Final system configurations
echo "Applying final system configurations..."

echo "Creating application directories..."
mkdir -p /opt/applications
mkdir -p /var/log/applications

echo "Setting system timezone..."
timedatectl set-timezone UTC

echo "Configuring system limits..."
cat >> /etc/security/limits.conf <<EOF
* soft nofile 65536
* hard nofile 65536
EOF

echo "Enabling automatic security updates..."
echo "APT::Periodic::Update-Package-Lists \"1\";" > /etc/apt/apt.conf.d/20auto-upgrades
echo "APT::Periodic::Unattended-Upgrade \"1\";" >> /etc/apt/apt.conf.d/20auto-upgrades

echo ""
echo "=============================================="
echo "Service configuration completed!"
echo "=============================================="
echo "Summary:"
echo "  - SSH: Configured and running"
echo "  - Nginx: Configured and running"
echo "  - Docker: Configured and running"
echo "  - Firewall: Enabled with basic rules"
echo "=============================================="