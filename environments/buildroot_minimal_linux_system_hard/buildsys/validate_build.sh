#!/bin/bash

# Build System Validation Script
# This script validates the build system configuration, checking for:
# - Package definition completeness
# - Configuration file syntax
# - Dependency resolution
# - Installation path specifications

set -e

BUILDSYS_DIR="/workspace/buildsys"
PACKAGE_DIR="${BUILDSYS_DIR}/package"
CONFIG_FILE="${BUILDSYS_DIR}/Config.in"
DEFCONFIG="${BUILDSYS_DIR}/configs/minimal_defconfig"
MANIFEST_FILE="${BUILDSYS_DIR}/rootfs.manifest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track validation status
VALIDATION_FAILED=0

# Function to check if a package directory exists
check_package_exists() {
    local package_name=$1
    local package_dir="${PACKAGE_DIR}/${package_name}"
    
    if [ ! -d "$package_dir" ]; then
        echo -e "${RED}ERROR: Package directory ${package_dir} does not exist${NC}"
        return 1
    fi
    return 0
}

# Function to check Config.in syntax
check_config_syntax() {
    local package_name=$1
    local config_file="${PACKAGE_DIR}/${package_name}/Config.in"
    
    if [ ! -f "$config_file" ]; then
        echo -e "${RED}ERROR: Config.in not found for package ${package_name}${NC}"
        return 1
    fi
    
    # Check for required config elements
    if ! grep -q "config BR2_PACKAGE_" "$config_file"; then
        echo -e "${RED}ERROR: Config.in for ${package_name} missing config declaration${NC}"
        return 1
    fi
    
    if ! grep -q "bool" "$config_file"; then
        echo -e "${RED}ERROR: Config.in for ${package_name} missing bool type${NC}"
        return 1
    fi
    
    return 0
}

# Function to check package .mk file
check_package_mk() {
    local package_name=$1
    local package_upper=$(echo "$package_name" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
    local mk_file="${PACKAGE_DIR}/${package_name}/${package_name}.mk"
    
    if [ ! -f "$mk_file" ]; then
        echo -e "${RED}ERROR: Makefile ${mk_file} not found${NC}"
        return 1
    fi
    
    # Check for required variables
    local required_vars=("${package_upper}_VERSION" "${package_upper}_SITE")
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}" "$mk_file"; then
            echo -e "${YELLOW}WARNING: ${var} not defined in ${mk_file}${NC}"
        fi
    done
    
    # Check for build/install commands
    if ! grep -q "define ${package_upper}_BUILD_CMDS" "$mk_file" && \
       ! grep -q "define ${package_upper}_INSTALL_TARGET_CMDS" "$mk_file"; then
        echo -e "${RED}ERROR: ${mk_file} missing build or install commands${NC}"
        return 1
    fi
    
    return 0
}

# Function to validate defconfig
validate_defconfig() {
    if [ ! -f "$DEFCONFIG" ]; then
        echo -e "${RED}ERROR: Defconfig file ${DEFCONFIG} not found${NC}"
        return 1
    fi
    
    return 0
}

# Function to extract package name from BR2_PACKAGE_ config
get_package_name_from_config() {
    local config_line=$1
    # Extract package name from BR2_PACKAGE_NAME=y
    local package_name=$(echo "$config_line" | sed 's/BR2_PACKAGE_//' | sed 's/=y//' | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    echo "$package_name"
}

# Function to check if main Config.in sources package configs
check_main_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}ERROR: Main Config.in not found${NC}"
        return 1
    fi
    
    # Check if main Config.in has proper structure
    if ! grep -q "menu" "$CONFIG_FILE"; then
        echo -e "${YELLOW}WARNING: Main Config.in missing menu structure${NC}"
    fi
    
    return 0
}

# Main validation logic
main() {
    echo "==================================="
    echo "Build System Validation Starting"
    echo "==================================="
    
    # Check main configuration file
    echo -e "\n${GREEN}[1/5] Checking main configuration...${NC}"
    if ! check_main_config; then
        VALIDATION_FAILED=1
    fi
    
    # Validate defconfig exists
    echo -e "\n${GREEN}[2/5] Validating defconfig...${NC}"
    if ! validate_defconfig; then
        VALIDATION_FAILED=1
        echo "VALIDATION: FAIL"
        exit 1
    fi
    
    # Find all enabled packages
    echo -e "\n${GREEN}[3/5] Finding enabled packages...${NC}"
    enabled_packages=$(grep "BR2_PACKAGE_.*=y" "$DEFCONFIG" 2>/dev/null || true)
    
    if [ -z "$enabled_packages" ]; then
        echo -e "${YELLOW}WARNING: No packages enabled in defconfig${NC}"
    else
        echo "Enabled packages found:"
        echo "$enabled_packages"
    fi
    
    # Validate each enabled package
    echo -e "\n${GREEN}[4/5] Validating package definitions...${NC}"
    while IFS= read -r pkg_line; do
        if [ -n "$pkg_line" ]; then
            pkg_name=$(get_package_name_from_config "$pkg_line")
            echo -e "\nValidating package: ${pkg_name}"
            
            if ! check_package_exists "$pkg_name"; then
                VALIDATION_FAILED=1
                continue
            fi
            
            if ! check_config_syntax "$pkg_name"; then
                VALIDATION_FAILED=1
                continue
            fi
            
            if ! check_package_mk "$pkg_name"; then
                VALIDATION_FAILED=1
                continue
            fi
            
            echo -e "${GREEN}Package ${pkg_name}: OK${NC}"
        fi
    done <<< "$enabled_packages"
    
    # Generate manifest
    echo -e "\n${GREEN}[5/5] Generating rootfs manifest...${NC}"
    echo "# Root Filesystem Manifest" > "$MANIFEST_FILE"
    echo "# Generated: $(date)" >> "$MANIFEST_FILE"
    echo "" >> "$MANIFEST_FILE"
    
    while IFS= read -r pkg_line; do
        if [ -n "$pkg_line" ]; then
            pkg_name=$(get_package_name_from_config "$pkg_line")
            pkg_upper=$(echo "$pkg_name" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
            mk_file="${PACKAGE_DIR}/${pkg_name}/${pkg_name}.mk"
            
            if [ -f "$mk_file" ]; then
                echo "Package: ${pkg_name}" >> "$MANIFEST_FILE"
                
                # Try to extract install path from mk file
                if grep -q "INSTALL.*usr/bin" "$mk_file"; then
                    echo "  Install: /usr/bin/${pkg_name}" >> "$MANIFEST_FILE"
                elif grep -q "INSTALL.*bin" "$mk_file"; then
                    echo "  Install: /bin/${pkg_name}" >> "$MANIFEST_FILE"
                fi
                echo "" >> "$MANIFEST_FILE"
            fi
        fi
    done <<< "$enabled_packages"
    
    echo "Manifest generated at: ${MANIFEST_FILE}"
    
    # Final validation result
    echo ""
    echo "==================================="
    if [ $VALIDATION_FAILED -eq 0 ]; then
        echo -e "${GREEN}VALIDATION: PASS${NC}"
        echo "==================================="
        exit 0
    else
        echo -e "${RED}VALIDATION: FAIL${NC}"
        echo "==================================="
        exit 1
    fi
}

# Run main validation
main