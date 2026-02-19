#!/bin/bash
# Legacy data extraction script - FAILING due to namespace issues
# This script attempts to extract service configuration data from vendor XML files
# Known issue: XPath queries fail on namespaced XML documents

CONFIG_DIR="/opt/data_pipeline/configs"

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory $CONFIG_DIR not found" >&2
    exit 1
fi

# Process each XML file in the config directory
for xml_file in "$CONFIG_DIR"/*.xml; do
    if [ ! -f "$xml_file" ]; then
        continue
    fi
    
    filename=$(basename "$xml_file")
    
    # Attempt to extract endpoint URLs - trying multiple element names
    # These queries FAIL on namespaced XML because they don't handle namespaces
    endpoint=$(xmllint --xpath "//endpoint/text()" "$xml_file" 2>/dev/null)
    if [ -z "$endpoint" ]; then
        endpoint=$(xmllint --xpath "//serviceUrl/text()" "$xml_file" 2>/dev/null)
    fi
    if [ -z "$endpoint" ]; then
        endpoint=$(xmllint --xpath "//url/text()" "$xml_file" 2>/dev/null)
    fi
    
    # Attempt to extract API version - trying multiple element names
    version=$(xmllint --xpath "//version/text()" "$xml_file" 2>/dev/null)
    if [ -z "$version" ]; then
        version=$(xmllint --xpath "//apiVersion/text()" "$xml_file" 2>/dev/null)
    fi
    
    # Attempt to extract authentication type - trying multiple element names
    auth=$(xmllint --xpath "//authType/text()" "$xml_file" 2>/dev/null)
    if [ -z "$auth" ]; then
        auth=$(xmllint --xpath "//authentication/text()" "$xml_file" 2>/dev/null)
    fi
    if [ -z "$auth" ]; then
        auth=$(xmllint --xpath "//auth/text()" "$xml_file" 2>/dev/null)
    fi
    
    # Output results - many will be empty due to namespace issues
    echo "$filename|endpoint|$endpoint"
    echo "$filename|version|$version"
    echo "$filename|auth|$auth"
done