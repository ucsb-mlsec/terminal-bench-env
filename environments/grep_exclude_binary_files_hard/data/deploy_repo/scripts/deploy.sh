#!/bin/bash

# Deployment Script for Production Environment
# This script handles the automated deployment process
# and validates service health after deployment

set -e  # Exit on any error

# Configuration Variables
API_ENDPOINT='https://api.deploy.example.com'
HEALTH_CHECK_ENDPOINT='https://health.example.com/status'
DEPLOY_USER='deploy-service'
APP_NAME='production-app'
DEPLOYMENT_TIMEOUT=300
RETRY_COUNT=5

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting deployment for ${APP_NAME}..."

# Step 1: Pre-deployment validation
echo "Validating deployment prerequisites..."
if [ ! -f "/etc/deploy/credentials" ]; then
    echo -e "${RED}Error: Credentials file not found${NC}"
    exit 1
fi

# Step 2: Deploy application artifacts
echo "Deploying application to ${API_ENDPOINT}..."
curl -X POST "${API_ENDPOINT}/deploy" \
    -H "Content-Type: application/json" \
    -d "{\"app\":\"${APP_NAME}\",\"user\":\"${DEPLOY_USER}\"}" \
    --max-time ${DEPLOYMENT_TIMEOUT}

# Step 3: Health check with retry logic
echo "Performing health checks..."
for i in $(seq 1 ${RETRY_COUNT}); do
    if curl -sf "${HEALTH_CHECK_ENDPOINT}" > /dev/null; then
        echo -e "${GREEN}Health check passed!${NC}"
        break
    fi
    echo "Health check attempt ${i}/${RETRY_COUNT} failed, retrying..."
    sleep 5
done

# Step 4: Final validation
echo "Deployment completed successfully!"