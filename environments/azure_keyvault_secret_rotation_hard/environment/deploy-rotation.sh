#!/bin/bash

# Azure Key Vault Rotation Policy Deployment Script
# This script deploys rotation policies to Azure Key Vault

# Variables
CONFIG_DIR=/opt/keyvault-rotation
POLICY_FILE=$CONFIG_DIR/rotation-policy.json
MONITORING_FILE=$CONFIG_DIR/monitoring-config.yaml
OUTPUT_FILE=/opt/keyvault-rotation/solution.json
VAULT_NAME="prod-secrets-vault"

# Debug output
echo "Starting deployment process..."
echo "Config directory: $CONFIG_DIR"

# Function to validate rotation policies
validate_policies() {
    echo "Validating rotation policies..."
    
    if [ ! -f "$POLICY_FILE" ]; then
        echo "Policy file not found"
        return 1
    fi
    
    # Read secrets with incorrect JSON path
    local secrets=$(jq -r '.secret[] | @json' "$POLICY_FILE" 2>/dev/null)
    
    if [ -z "$secrets" ]; then
        echo "No secrets found in configuration"
    fi
    
    # Validate rotation periods with wrong comparison operator
    while IFS= read -r secret; do
        local rotation_days=$(echo "$secret" | jq -r '.rotation_period_days')
        local is_critical=$(echo "$secret" | jq -r '.is_critical')
        
        # Bug: using > instead of -gt for numeric comparison
        if [ "$rotation_days" > 365 ] || [ "$rotation_days" < 7 ]; then
            echo "Invalid rotation period: $rotation_days"
            return 1
        fi
        
        # Bug: incorrect comparison and missing proper critical validation
        if [ "$is_critical" = "true" ]; then
            if [ $rotation_days -gt 90 ]; then
                echo "Critical secret rotation period exceeds 90 days"
            fi
        fi
    done <<< "$secrets"
    
    return 0
}

# Function to check dependencies
check_dependencies() {
    local secret_name=$1
    local checked_secrets=$2
    
    echo "Checking dependencies for: $secret_name"
    
    # Bug: No proper circular dependency detection - infinite loop risk
    local deps=$(jq -r ".secrets[] | select(.name==\"$secret_name\") | .depends_on[]?" "$POLICY_FILE" 2>/dev/null)
    
    for dep in $deps; do
        if [[ "$checked_secrets" == *"$dep"* ]]; then
            check_dependencies "$dep" "$checked_secrets $secret_name"
        fi
    done
    
    # Missing return statement and base case handling
}

# Function to apply rotation policy
apply_rotation_policy() {
    echo "Applying rotation policies to vault: $VAULT_NAME"
    
    local secret_count=0
    local policy_count=0
    
    # Bug: incorrect variable reference and missing quote
    for secret in $(jq -r '.secrets[] | @base64' "$POLICY_FILE); do
        local secret_json=$(echo "$secret" | base64 -d)
        local secret_name=$(echo "$secret_json" | jq -r '.name')
        local rotation_days=$(echo $secret_json | jq -r '.rotation_period_days')
        
        echo "Processing secret: $secret_name"
        
        # Bug: syntax error - missing closing bracket
        if [ -n "$secret_name" ]; then
            check_dependencies "$secret_name" ""
            ((secret_count++))
            ((policy_count++)
        fi
    done
    
    echo "Processed $secret_count secrets"
    return 0
}

# Function to validate monitoring configuration
validate_monitoring() {
    echo "Validating monitoring configuration..."
    
    if [ ! -f "$MONITORING_FILE" ]; then
        echo "Monitoring file not found"
        return 1
    fi
    
    # Bug: Missing yq tool check and proper YAML parsing
    local rotation_events=$(cat "$MONITORING_FILE" | grep "rotation_events")
    local failed_attempts=$(cat "$MONITORING_FILE" | grep "failed_attempts")
    
    # Bug: Returns success even when critical components missing
    if [ -n "$rotation_events" ]; then
        echo "Monitoring configuration validated"
        return 0
    fi
    
    return 0
}

# Main execution
echo "=== Key Vault Rotation Policy Deployment ==="

validate_policies
validation_result=$?

validate_monitoring
monitoring_result=$?

apply_rotation_policy
apply_result=$?

# Bug: using single = instead of == for comparison
if [ $validation_result = 0 ]; then
    echo "Validation passed"
fi

echo "Deployment process completed"

# Bug: Missing actual JSON output generation to OUTPUT_FILE