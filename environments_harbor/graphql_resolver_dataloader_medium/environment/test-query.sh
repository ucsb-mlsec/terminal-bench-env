#!/bin/bash

# Wait for server to be ready
sleep 3

# Reset the fetch counter
curl -s -X POST http://localhost:4000/reset-count > /dev/null

# Execute GraphQL query to fetch all books with author information
curl -s -X POST http://localhost:4000/graphql \
  -H "Content-Type: application/json" \
  --data '{
    "query": "{ books { id title publishedYear author { id name bio } } }"
  }' > /dev/null

# Fetch the author fetch count
RESPONSE=$(curl -s -X GET http://localhost:4000/fetch-count)

# Parse the JSON response to extract authorFetchCount
AUTHOR_FETCHES=$(echo "$RESPONSE" | grep -o '"authorFetchCount":[0-9]*' | grep -o '[0-9]*')

# Determine if test passed
if [ "$AUTHOR_FETCHES" -le 3 ]; then
  TEST_PASSED="yes"
  EXIT_CODE=0
else
  TEST_PASSED="no"
  EXIT_CODE=1
fi

# Write results to /workspace/result.txt
cat > /workspace/result.txt <<EOF
author_fetches=$AUTHOR_FETCHES
test_passed=$TEST_PASSED
EOF

# Echo summary message
echo "Author fetch count: $AUTHOR_FETCHES"
echo "Test passed: $TEST_PASSED"

# Exit with appropriate code
exit $EXIT_CODE