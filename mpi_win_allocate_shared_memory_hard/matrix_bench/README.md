# Matrix Computation Benchmark

## Overview

This is a parallel matrix computation benchmark designed for intranode MPI execution. The benchmark performs matrix operations using multiple MPI processes running on a single physical node.

**Purpose:** Evaluate parallel matrix computation performance on multi-core systems using MPI for process-level parallelism.

**Design:** The application distributes matrix computation workload across multiple MPI processes. Each process performs calculations on different portions of the matrix, and results are aggregated at the end.

**Input:** Binary matrix data file containing a 2000x2000 matrix of double-precision floating-point values (~32 MB uncompressed).

## Building

### Dependencies
- MPI implementation (OpenMPI or MPICH)
- GCC compiler (version 4.8 or later recommended)
- Make build system

### Generate Test Data
Before running the benchmark, you must generate the test matrix data file:

```bash
make generate
./generate_matrix
```

This creates `matrix_data.bin` containing the test matrix.

### Compile the Benchmark
To build the main benchmark application:

```bash
make
```

Or explicitly:

```bash
make all
```

This produces the `matrix_benchmark` executable.

### Clean Build Artifacts
To remove compiled binaries and object files:

```bash
make clean
```

## Running

### Basic Usage
Run the benchmark with 4 MPI processes:

```bash
mpirun -np 4 ./matrix_benchmark
```

### Recommended Configurations
The benchmark is designed for the following process counts on a single node:

- 2 processes: `mpirun -np 2 ./matrix_benchmark`
- 4 processes: `mpirun -np 4 ./matrix_benchmark`
- 8 processes: `mpirun -np 8 ./matrix_benchmark`

**Important:** All processes should run on the same physical node for intranode performance testing. Use appropriate MPI binding options if needed.

### Process Binding
For consistent performance measurements, consider binding processes to cores:

```bash
mpirun -np 4 --bind-to core ./matrix_benchmark
```

## Expected Output

The benchmark produces the following output metrics:

1. **Execution Time:** Total wall-clock time in seconds for the computation phase
2. **Result Checksum:** A verification value to ensure computational correctness
3. **Memory Usage:** Peak resident set size (RSS) in megabytes

Example output:
```
Matrix size: 2000x2000
Number of processes: 4
Loading matrix data...
Starting computation...
Execution time: 3.245 seconds
Result checksum: 1234567890.123456
Peak memory usage: 156.4 MB
```

## Performance Notes

### Current Limitations
The current implementation may exhibit memory efficiency issues when scaling to multiple processes:

- **Memory scaling:** Memory usage increases linearly with the number of processes
- **Cache pressure:** High memory consumption can lead to cache thrashing
- **Bandwidth saturation:** Excessive memory usage may saturate memory bandwidth

### Optimization Opportunities
Consider the following areas for performance improvement:

- **Shared data access:** Multiple processes accessing the same matrix data
- **Memory duplication:** Each process may be creating its own copy of shared data
- **Intranode communication:** Leverage shared memory for processes on the same node

For intranode execution, shared memory techniques can significantly reduce memory footprint and improve performance by eliminating redundant data copies.

## File Structure

```
matrix_bench/
├── matrix_benchmark.c   # Main benchmark application
├── generate_matrix.c    # Test data generator utility
├── Makefile            # Build system configuration
├── matrix_data.bin     # Binary test matrix data (generated)
└── README.md          # This file
```

### File Descriptions

- **matrix_benchmark.c:** Contains the main benchmark logic, including matrix loading, computation distribution, and result aggregation.
- **generate_matrix.c:** Utility program that generates the test matrix data file with deterministic pseudo-random values.
- **Makefile:** Build configuration supporting multiple targets (all, generate, clean).
- **matrix_data.bin:** Binary file containing the 2000x2000 test matrix (created by generate_matrix).

## Verification

To verify correctness of results, compare the checksum values across different runs and process counts. The checksum should remain constant regardless of the number of processes used, ensuring that the parallel computation produces identical results to a serial implementation.