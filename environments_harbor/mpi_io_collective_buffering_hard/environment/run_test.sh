#!/bin/bash
set -e

echo "Compiling climate simulation..."
make clean
make

echo "Running simulation with 8 processes..."
mpirun -np 8 ./climate_sim

echo "Test completed"
exit 0