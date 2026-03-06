# WebAssembly Calculator

A simple calculator program compiled to WebAssembly using WASI.

## Description

This project demonstrates a basic calculator implementation written in C and compiled to WebAssembly. It uses the WebAssembly System Interface (WASI) for system-level operations, making it portable across different WASI-compliant runtime environments.

## Features

The calculator supports the following basic arithmetic operations:
- Addition
- Subtraction
- Multiplication
- Division

## Project Structure

- `calculator.c` - Main program entry point and user interface logic
- `math_ops.c` - Implementation of mathematical operations
- `math_ops.h` - Header file with function declarations for math operations
- `compile.sh` - Build automation script (to be created)

## Build Requirements

WASI-compatible WebAssembly compiler required. The project is designed to be compiled with any toolchain that supports the WASI target, such as:
- WASI SDK
- Clang with wasm32-wasi target
- Emscripten (with WASI configuration)

## Usage

After building the project, the resulting `calculator.wasm` module can be executed in any WASI-compliant runtime environment such as:
- Wasmtime
- Wasmer
- Node.js with WASI support

Example:
```
wasmtime calculator.wasm
```

## Technical Notes

This project targets the `wasm32-wasi` architecture and uses only standard C library functions that are supported in WASI environments. No external dependencies are required beyond the standard library.