#!/bin/bash
set -e

echo "Starting load test..."

# Wait for app to be ready
sleep 2

# Create 3 initial users to populate the database
echo "Creating initial users..."
curl --fail -X POST http://localhost:5000/api/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "email": "alice@example.com"}' > /dev/null 2>&1

curl --fail -X POST http://localhost:5000/api/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Bob", "email": "bob@example.com"}' > /dev/null 2>&1

curl --fail -X POST http://localhost:5000/api/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Charlie", "email": "charlie@example.com"}' > /dev/null 2>&1

# Testing connection pool under load - making rapid requests to stress test the pool
echo "Running load test with 50 iterations..."
for i in $(seq 1 50); do
  # GET all users
  curl --fail http://localhost:5000/api/users > /dev/null 2>&1
  
  # GET specific user
  curl --fail http://localhost:5000/api/users/1 > /dev/null 2>&1
  
  # POST create new user
  curl --fail -X POST http://localhost:5000/api/users \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"User${i}\", \"email\": \"user${i}@example.com\"}" > /dev/null 2>&1
  
  # Small delay between iterations
  sleep 0.1
done

echo "LOAD_TEST_PASSED"