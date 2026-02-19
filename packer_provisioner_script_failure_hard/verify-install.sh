#!/bin/bash

# verify-install.sh
# This script verifies that all required packages, services, and configurations
# are properly installed and running on the Ubuntu server image.
# Exit code 0 indicates success, exit code 1 indicates failure.

set -e

echo "=========================================="
echo "Starting Installation Verification"
echo "=========================================="

# Track verification status
VERIFICATION_FAILED=0

# Function to check if a package is installed
check_package() {
    local package_name=$1
    echo "Checking if package '$package_name' is installed..."
    if dpkg -l | grep -q "^ii  $package_name"; then
        echo "  [OK] Package '$package_name' is installed"
        return 0
    else
        echo "  [FAIL] Package '$package_name' is NOT installed"
        VERIFICATION_FAILED=1
        return 1
    fi
}

# Function to check if a service is running
check_service() {
    local service_name=$1
    echo "Checking if service '$service_name' is running..."
    if systemctl is-active --quiet "$service_name"; then
        echo "  [OK] Service '$service_name' is running"
        return 0
    else
        echo "  [FAIL] Service '$service_name' is NOT running"
        systemctl status "$service_name" || true
        VERIFICATION_FAILED=1
        return 1
    fi
}

# Function to check if a file exists
check_file() {
    local file_path=$1
    echo "Checking if file '$file_path' exists..."
    if test -f "$file_path"; then
        echo "  [OK] File '$file_path' exists"
        return 0
    else
        echo "  [FAIL] File '$file_path' does NOT exist"
        VERIFICATION_FAILED=1
        return 1
    fi
}

# Verify critical packages are installed
echo ""
echo "--- Verifying Package Installations ---"
check_package "nginx"
check_package "docker.io"
check_package "python3"
check_package "curl"

# Verify required services are running
echo ""
echo "--- Verifying Service Status ---"
check_service "nginx"
check_service "docker"
check_service "ssh"

# Verify configuration files exist
echo ""
echo "--- Verifying Configuration Files ---"
check_file "/etc/nginx/nginx.conf"
check_file "/etc/docker/daemon.json"
check_file "/etc/ssh/sshd_config"

# Final verification result
echo ""
echo "=========================================="
if [ $VERIFICATION_FAILED -eq 0 ]; then
    echo "Verification PASSED - All checks successful"
    echo "=========================================="
    exit 0
else
    echo "Verification FAILED - One or more checks failed"
    echo "=========================================="
    exit 1
fi