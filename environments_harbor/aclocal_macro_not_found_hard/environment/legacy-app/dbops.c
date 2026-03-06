#!/bin/bash
set -e

# Fix build environment for legacy-app
# This script resolves missing autotools macros and prepares the build system

# First, let's check what's causing the autoreconf failure
cd /workspace/legacy-app

# Attempt to run autoreconf to identify missing macros
echo "Identifying missing autotools macros..."
autoreconf_output=$(autoreconf -i 2>&1 || true)

# Common missing macros and their required packages:
# LT_INIT, LT_PREREQ - libtool
# PKG_CHECK_MODULES - pkg-config
# AC_PROG_LIBTOOL - libtool (deprecated, but still used)

# Determine the package manager
if command -v apt-get >/dev/null 2>&1; then
    PKG_MANAGER="apt-get"
    INSTALL_CMD="apt-get install -y"
    UPDATE_CMD="apt-get update"
elif command -v yum >/dev/null 2>&1; then
    PKG_MANAGER="yum"
    INSTALL_CMD="yum install -y"
    UPDATE_CMD="yum makecache"
elif command -v apk >/dev/null 2>&1; then
    PKG_MANAGER="apk"
    INSTALL_CMD="apk add"
    UPDATE_CMD="apk update"
else
    echo "Unsupported package manager"
    exit 1
fi

# Update package cache
echo "Updating package cache..."
$UPDATE_CMD

# Install essential autotools packages
echo "Installing required autotools packages..."

if [ "$PKG_MANAGER" = "apt-get" ]; then
    $INSTALL_CMD \
        autoconf \
        automake \
        libtool \
        pkg-config \
        build-essential
elif [ "$PKG_MANAGER" = "yum" ]; then
    $INSTALL_CMD \
        autoconf \
        automake \
        libtool \
        pkgconfig \
        gcc \
        make
elif [ "$PKG_MANAGER" = "apk" ]; then
    $INSTALL_CMD \
        autoconf \
        automake \
        libtool \
        pkgconfig \
        gcc \
        make \
        musl-dev
fi

# Run aclocal to regenerate aclocal.m4 with all available macros
echo "Regenerating aclocal.m4..."
cd /workspace/legacy-app
aclocal --install 2>/dev/null || aclocal

# Create necessary auxiliary directories
echo "Creating auxiliary build directories..."
mkdir -p /workspace/legacy-app/m4

# Ensure libtoolize is available (may be glibtoolize on some systems)
if command -v libtoolize >/dev/null 2>&1; then
    LIBTOOLIZE=libtoolize
elif command -v glibtoolize >/dev/null 2>&1; then
    LIBTOOLIZE=glibtoolize
else
    echo "libtoolize not found"
    exit 1
fi

# Run libtoolize to install libtool support files
echo "Installing libtool support files..."
$LIBTOOLIZE --force --copy --install 2>/dev/null || $LIBTOOLIZE --force --copy

echo "Build environment fix completed successfully"