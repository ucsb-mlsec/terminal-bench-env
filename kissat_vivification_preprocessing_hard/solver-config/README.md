# SAT Solver Configuration System

This is a simplified SAT solver configuration system for testing and evaluation purposes. It mimics production solver build patterns used in formal verification pipelines.

## Overview

The solver implements a multi-layered configuration system that controls preprocessing features including vivification, clause elimination, subsumption, and other simplification techniques.

## Configuration Layers

The solver uses multiple configuration layers with a defined priority hierarchy:

### 1. Compile-Time Configuration

Located in `include/build_config.h` - Main build configuration header file.

Preprocessor directives control feature compilation:
- `FEATURE_VIVIFICATION` - Enable vivification support
- `FEATURE_SUBSUMPTION` - Enable subsumption checking
- `FEATURE_ELIMINATE` - Enable variable elimination

Build systems can set these flags:
- `CMakeLists.txt` - CMake options (ENABLE_VIVIFICATION, etc.)
- `config.mk` - Make configuration flags for traditional builds

Features disabled at compile time cannot be enabled at runtime.

### 2. Runtime Configuration File

Located at `config/solver.conf` - Runtime configuration file.

Uses INI-style format with sections for different components:

```
[Preprocessing]
Enabled=true
Vivification=true
VivifyMaxEffort=100
```

The solver searches for configuration files in:
1. Current working directory (./solver.conf)
2. Config subdirectory (./config/solver.conf)
3. User home directory (~/.solver.conf)

### 3. Command-Line Options

Command-line arguments provide the highest priority overrides.

The solver accepts various flags:

```
./solver [options] input.cnf

Options:
  --vivify              Enable vivification preprocessing
  --no-vivify           Disable vivification preprocessing
  --no-preprocess       Disable all preprocessing
  --verbose             Verbose output with statistics
  --config=FILE         Specify configuration file path
  --help                Display help message
```

## Vivification Feature

Vivification is a clause simplification preprocessing technique that attempts to shorten clauses by temporarily assigning variables and checking if any literals become redundant.

**Configuration Hierarchy (highest to lowest priority):**
1. Command-line flag (--vivify / --no-vivify)
2. Runtime config file (solver.conf [Preprocessing] Vivification setting)
3. Compile-time default (FEATURE_VIVIFICATION in build_config.h)

**Default Behavior:**
If vivification is compiled in (FEATURE_VIVIFICATION defined) but no runtime configuration is provided, the default is ENABLED.

## Building the Solver

### Using CMake:
```bash
mkdir build && cd build
cmake -DENABLE_VIVIFICATION=ON ..
make
```

### Using Make:
```bash
make ENABLE_VIVIFICATION=1
```

## Testing and Verification

### Basic test:
```bash
./bin/solver test.cnf
```

### Verbose mode to see active features:
```bash
./bin/solver --verbose test.cnf
```

### Testing with specific configuration:
```bash
./bin/solver --vivify --verbose test.cnf
./bin/solver --no-vivify --verbose test.cnf
```

The verbose output will display preprocessing statistics including:
- Active preprocessing techniques
- Number of vivification attempts
- Clauses simplified
- Runtime for each preprocessing phase