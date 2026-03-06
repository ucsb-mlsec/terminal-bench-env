#!/bin/bash
cd /workspace/sat-solver
javac SATSolver.java
if [ $? -eq 0 ]; then
    echo "Build successful"
else
    echo "Build failed"
    exit 1
fi