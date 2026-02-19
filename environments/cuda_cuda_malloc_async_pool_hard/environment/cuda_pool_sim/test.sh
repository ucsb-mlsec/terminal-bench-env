#!/bin/bash

set -e

echo "Building CUDA Pool Simulation..."

mkdir -p build

cd build

cmake ..

make

echo "Running simulation test..."

cd ..

set +e
./build/pool_sim
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ]; then
    echo "Test Status: PASS"
else
    echo "Test Status: FAIL (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE