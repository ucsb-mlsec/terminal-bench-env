#!/bin/bash

# Backup Automation System - Test Environment Setup Script
# This script creates a test environment to investigate file move atomicity
# across same-filesystem and cross-filesystem scenarios

set -e  # Exit on error

echo "Setting up test environment for file move atomicity investigation..."

# Clean up any previous test environment
echo "Cleaning up previous test environment..."
umount /workspace/target_fs2 2>/dev/null || true
rm -rf /workspace/source /workspace/target /workspace/source_fs2 /workspace/target_fs2
rm -f /workspace/fs2.img

# ===================================================================
# SCENARIO 1: Same Filesystem Setup
# Both source and target directories on the default filesystem
# ===================================================================
echo ""
echo "Setting up Scenario 1: Same filesystem..."

# Create source directory
mkdir -p /workspace/source
echo "Created /workspace/source on default filesystem"

# Create target directory  
mkdir -p /workspace/target
echo "Created /workspace/target on default filesystem"

# Create test files in source directory
echo "Creating test files in /workspace/source..."
echo "Test file content for backup system - File 1" > /workspace/source/test1.txt
echo "Test file content for backup system - File 2" > /workspace/source/test2.txt
echo "Test file content for backup system - File 3" > /workspace/source/test3.txt
echo "Test file content for backup system - File 4" > /workspace/source/test4.txt
echo "Test file content for backup system - File 5" > /workspace/source/test5.txt

# Set proper permissions
chmod -R 755 /workspace/source /workspace/target

echo "Scenario 1 setup complete: source and target on same filesystem"

# ===================================================================
# SCENARIO 2: Cross Filesystem Setup
# Source and target directories on different filesystems
# ===================================================================
echo ""
echo "Setting up Scenario 2: Cross filesystem..."

# Create source directory for second scenario (on default filesystem)
mkdir -p /workspace/source_fs2
echo "Created /workspace/source_fs2 on default filesystem"

# Create test files in source_fs2 directory
echo "Creating test files in /workspace/source_fs2..."
echo "Test file content for backup system - FS2 File 1" > /workspace/source_fs2/test1.txt
echo "Test file content for backup system - FS2 File 2" > /workspace/source_fs2/test2.txt
echo "Test file content for backup system - FS2 File 3" > /workspace/source_fs2/test3.txt
echo "Test file content for backup system - FS2 File 4" > /workspace/source_fs2/test4.txt
echo "Test file content for backup system - FS2 File 5" > /workspace/source_fs2/test5.txt

# Create a separate filesystem using tmpfs (lightweight and reliable)
echo "Creating separate filesystem using tmpfs..."
mkdir -p /workspace/target_fs2

# Mount tmpfs as a separate filesystem
mount -t tmpfs -o size=100M tmpfs /workspace/target_fs2

# Verify the mount was successful
if mountpoint -q /workspace/target_fs2; then
    echo "Successfully mounted tmpfs at /workspace/target_fs2"
else
    echo "ERROR: Failed to mount tmpfs at /workspace/target_fs2"
    exit 1
fi

# Set proper permissions
chmod -R 755 /workspace/source_fs2 /workspace/target_fs2

echo "Scenario 2 setup complete: source_fs2 on default filesystem, target_fs2 on tmpfs"

# ===================================================================
# Verification and Summary
# ===================================================================
echo ""
echo "=========================================="
echo "Test Environment Setup Complete"
echo "=========================================="
echo ""
echo "Scenario 1 (Same Filesystem):"
echo "  Source: /workspace/source"
echo "  Target: /workspace/target"
df /workspace/source /workspace/target | tail -n +2
echo ""
echo "Scenario 2 (Cross Filesystem):"
echo "  Source: /workspace/source_fs2 (default filesystem)"
echo "  Target: /workspace/target_fs2 (tmpfs)"
df /workspace/source_fs2 /workspace/target_fs2 | tail -n +2
echo ""
echo "Test files created:"
ls -lh /workspace/source/
echo ""
ls -lh /workspace/source_fs2/
echo ""
echo "Ready for atomicity testing!"