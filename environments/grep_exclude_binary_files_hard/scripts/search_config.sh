#!/bin/bash

# Quick script to find API endpoint configurations in deployment repo
# Searches for common API endpoint patterns across all files

REPO_PATH="/workspace/deploy_repo"

echo "Searching for API endpoint configurations in $REPO_PATH..."
echo "================================================"

# Search for various API endpoint patterns
grep -r "api_endpoint" "$REPO_PATH"
grep -r "API_URL" "$REPO_PATH"
grep -r "endpoint:" "$REPO_PATH"
grep -r "api.endpoint" "$REPO_PATH"
grep -r "apiEndpoint" "$REPO_PATH"

echo "================================================"
echo "Search complete"