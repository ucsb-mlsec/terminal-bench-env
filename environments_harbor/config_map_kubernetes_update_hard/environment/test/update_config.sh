#!/bin/bash

# update_config.sh - Simulates Kubernetes ConfigMap atomic file updates
#
# Usage examples:
#   ./update_config.sh --timeout 60 --feature_enabled false --api_endpoint "https://api.newexample.com"
#   ./update_config.sh --timeout 45
#   ./update_config.sh --feature_enabled true

CONFIG_FILE="/config/app.json"
TEMP_FILE="/config/app.json.tmp"

# Read current configuration or set defaults
TIMEOUT=30
FEATURE_ENABLED=true
API_ENDPOINT="https://api.example.com"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --feature_enabled)
            FEATURE_ENABLED="$2"
            shift 2
            ;;
        --api_endpoint)
            API_ENDPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Write new configuration to temporary file
cat > "$TEMP_FILE" <<EOF
{
  "timeout": $TIMEOUT,
  "feature_enabled": $FEATURE_ENABLED,
  "api_endpoint": "$API_ENDPOINT"
}
EOF

# Atomically move temp file to final location (simulates Kubernetes ConfigMap behavior)
mv "$TEMP_FILE" "$CONFIG_FILE"

echo "Configuration updated: timeout=$TIMEOUT, feature_enabled=$FEATURE_ENABLED, api_endpoint=$API_ENDPOINT"