# Sensor Filter Verification

## Overview

This repository contains a safety-critical filtering algorithm designed for embedded sensor systems. The filter processes continuous streams of sensor data and must operate within strict safety constraints. Due to the critical nature of the application, formal verification is employed to mathematically prove the absence of runtime errors and safety violations.

## Verification Tool

The system uses CBMC (C Bounded Model Checker) for formal verification. CBMC performs symbolic execution to exhaustively explore all possible execution paths within specified bounds.

To run verification:
```
./verify.sh
```

The tool analyzes the code by converting it to a logical formula and uses SAT/SMT solvers to find safety violations. If no violations exist within the specified bounds, the code is proven correct for those execution scenarios.

## Configuration Parameters

The primary configuration parameter is **UNWIND_BOUND**, which controls the depth of loop analysis.

- **UNWIND_BOUND**: Specifies the maximum number of loop iterations to analyze (default: 5)
- When UNWIND_BOUND is set too low, the verification is incomplete because not all execution paths are explored
- The tool will emit "unwinding assertion" warnings when the bound is insufficient
- Finding the correct bound depends on the maximum number of iterations your loops can execute
- Typical values range from 5-20 depending on code complexity
- Setting the bound too high wastes computational resources without improving verification quality

The configuration file `verification.cfg` contains this parameter and can be edited to adjust the analysis depth.

## Expected Output

The verification tool provides clear feedback about the analysis results:

- **VERIFICATION SUCCESSFUL**: All safety properties hold, no violations found, analysis is complete
- **Unwinding assertions failed**: The UNWIND_BOUND is too low, increase it for complete analysis
- **VERIFICATION FAILED**: Safety violations detected, counterexamples will be provided
- **Assertion violations**: Specific safety properties were violated, with trace information

## Safety Properties

The verification checks the following properties:

1. **Array bounds checking**: Ensures all array accesses are within valid ranges
2. **Buffer overflow prevention**: Verifies no write operations exceed buffer boundaries
3. **Assertion violations**: Checks all assert() statements in the code
4. **Integer overflow detection**: Ensures arithmetic operations don't overflow
5. **Null pointer dereference**: Verifies all pointer accesses are valid
6. **Division by zero**: Ensures no division operations have zero divisors

These properties ensure the filter operates safely under all possible input conditions within the specified bounds.