#!/bin/bash
NEW_TOKEN="refreshed_token_$(date +%s)_$RANDOM"
NEW_EXPIRY=$(($(date +%s) + 3600))
echo "{\"access_token\": \"$NEW_TOKEN\", \"expires_at\": $NEW_EXPIRY}" > /data/auth/token.json