#!/bin/bash

# run_simulation.sh - Protein folding stability simulation
# Accepts temperature in Kelvin (250.0 - 400.0) and outputs stability score

# Check if exactly one argument is provided
if [ $# -ne 1 ]; then
    echo "Error: Exactly one argument required (temperature in Kelvin)" >&2
    exit 1
fi

TEMP=$1

# Validate temperature is a number and within range
if ! [[ $TEMP =~ ^[0-9]+\.?[0-9]*$ ]] && ! [[ $TEMP =~ ^[0-9]*\.[0-9]+$ ]]; then
    echo "Error: Temperature must be a valid number" >&2
    exit 1
fi

# Check range using bc for float comparison
if (( $(echo "$TEMP < 250.0" | bc -l) )) || (( $(echo "$TEMP > 400.0" | bc -l) )); then
    echo "Error: Temperature must be between 250.0 and 400.0 Kelvin" >&2
    exit 1
fi

# Simulate computation time (1-2 seconds)
sleep 0.$(( RANDOM % 100 + 100 ))

# Complex stability landscape with multiple peaks
# Using bc for high-precision floating-point calculations
# The function combines multiple Gaussian-like peaks and valleys

# Normalize temperature to [0, 1] range for easier calculation
NORM=$(echo "scale=10; ($TEMP - 250.0) / 150.0" | bc -l)

# Create a complex landscape with multiple local maxima
# Peak 1: Around T=280 (normalized ~0.2) - local max ~0.75
PEAK1=$(echo "scale=10; 0.75 * e(-50 * ($NORM - 0.2)^2)" | bc -l)

# Peak 2: Around T=310 (normalized ~0.4) - local max ~0.82
PEAK2=$(echo "scale=10; 0.82 * e(-40 * ($NORM - 0.4)^2)" | bc -l)

# Peak 3: Around T=347 (normalized ~0.647) - GLOBAL MAX ~0.93
PEAK3=$(echo "scale=10; 0.93 * e(-60 * ($NORM - 0.647)^2)" | bc -l)

# Peak 4: Around T=375 (normalized ~0.833) - local max ~0.78
PEAK4=$(echo "scale=10; 0.78 * e(-45 * ($NORM - 0.833)^2)" | bc -l)

# Valley effects to create complexity
VALLEY1=$(echo "scale=10; -0.15 * e(-30 * ($NORM - 0.55)^2)" | bc -l)
VALLEY2=$(echo "scale=10; -0.12 * e(-35 * ($NORM - 0.75)^2)" | bc -l)

# Base landscape with gentle slope
BASE=$(echo "scale=10; 0.35 + 0.15 * s(3.14159 * $NORM)" | bc -l)

# Combine all components
SCORE=$(echo "scale=10; $BASE + $PEAK1 + $PEAK2 + $PEAK3 + $PEAK4 + $VALLEY1 + $VALLEY2" | bc -l)

# Add small noise term based on temperature (deterministic)
# Use temperature digits to create consistent pseudo-random noise
TEMP_INT=$(echo "$TEMP * 1000 / 1" | bc)
NOISE_SEED=$(( TEMP_INT % 1000 ))
NOISE=$(echo "scale=10; 0.02 * s($NOISE_SEED * 0.01)" | bc -l)
SCORE=$(echo "scale=10; $SCORE + $NOISE" | bc -l)

# Ensure score is within [0, 1] range
if (( $(echo "$SCORE < 0" | bc -l) )); then
    SCORE="0.0000"
elif (( $(echo "$SCORE > 1" | bc -l) )); then
    SCORE="1.0000"
fi

# Format to 4 decimal places and output
printf "%.4f\n" $SCORE