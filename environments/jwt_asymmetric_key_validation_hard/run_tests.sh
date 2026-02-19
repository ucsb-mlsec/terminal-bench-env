#!/bin/bash

set -e

echo "========================================="
echo "JWT Authentication Service Test Suite"
echo "========================================="
echo ""

# Test 1: Validate token signed with first key
echo "Test 1: Validating token signed with public_key1.pem..."
TOKEN1=$(cat /opt/auth-service/token_key1.txt)

if python3 -c "import jwt_validator; result = jwt_validator.validate_token('$TOKEN1', 'data/'); print('Valid' if result else 'Invalid'); exit(0 if result else 1)"; then
    echo "✓ Test 1 PASSED: Token validated successfully with first key"
else
    echo "✗ Test 1 FAILED: Token validation failed"
    exit 1
fi

echo ""

# Test 2: Validate token signed with second key (tests key rotation)
echo "Test 2: Validating token signed with public_key2.pem (key rotation test)..."
TOKEN2=$(cat /opt/auth-service/token_key2.txt)

if python3 -c "import jwt_validator; result = jwt_validator.validate_token('$TOKEN2', 'data/'); print('Valid' if result else 'Invalid'); exit(0 if result else 1)"; then
    echo "✓ Test 2 PASSED: Token validated successfully with second key (key rotation working)"
else
    echo "✗ Test 2 FAILED: Key rotation not working properly"
    exit 1
fi

echo ""
echo "========================================="
echo "All tests passed!"
echo "========================================="

exit 0