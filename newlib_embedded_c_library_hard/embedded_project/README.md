# Embedded Data Logger Project

## Overview

This is a bare-metal firmware application for ARM Cortex-M4 microcontrollers designed to perform continuous data logging operations. The application samples sensor data at configurable intervals and maintains an internal buffer for data storage before transmission.

The system is built on a bare-metal architecture without an RTOS, using newlib as the C standard library implementation. All memory management is performed using the standard malloc/free functions with custom system call stubs.

## Target Hardware

- **Processor**: ARM Cortex-M4 (STM32F4 series)
- **Core Clock**: 168 MHz
- **FPU**: Single-precision floating point unit enabled
- **Peripherals**: UART, SPI, I2C, ADC, Timers

## Memory Layout

The microcontroller has the following memory configuration:

- **Flash Memory**: 256KB starting at 0x08000000
  - Program code (.text section)
  - Read-only data (.rodata section)
  - Initialization data for .data section
  
- **SRAM**: 64KB starting at 0x20000000
  - Initialized data (.data section)
  - Uninitialized data (.bss section)
  - Heap (dynamic memory allocation)
  - Stack (grows downward from top of SRAM)

## Build Instructions

To build the project, use the following commands:

```
make clean
make all
make size
```

The build system uses arm-none-eabi-gcc with optimization level -O2. Link-time optimization is disabled for debugging purposes.

To flash the binary to the target:

```
make flash
```

## Developer Notes

### Previous Developer Comments (March 2024)

- **Memory allocation tested extensively in lab environment**: Ran continuous operation tests for up to 6 hours with various allocation patterns. No issues observed during benchtop testing.

- **Heap configured with generous 50KB space**: The heap has been allocated 50KB of space which should be more than sufficient for the application's needs. Maximum observed heap usage in testing was approximately 28KB.

- **Field reports indicate crashes after 24-48 hours of operation**: Multiple deployment sites reporting system freezes or unexpected resets after extended operation. Pattern appears consistent but cannot be reproduced in lab.

- **Stack usage appears normal in testing**: Stack high-water mark analysis shows typical usage around 2-3KB. Allocated 4KB for stack which should provide adequate margin.

- **TODO: Investigate malloc failures in production**: Some crash dumps suggest malloc may be returning NULL in the field, but this has never occurred during testing. Need to add better error handling and telemetry.

### Additional Observations

- Application performs periodic large allocations for data buffering
- Memory fragmentation could be a concern over long run times
- No memory leak detected in static analysis or short-term testing
- Watchdog timer enabled with 5-second timeout

## Known Issues

1. **Intermittent Crashes**: System experiences crashes after extended operation (24-48 hours) in field deployment. Root cause unknown. Does not reproduce in laboratory testing environment.

2. **Suspected Memory Corruption**: Some crash dumps show corrupted data structures. Memory corruption patterns suggest possible heap/stack collision or buffer overflow.

3. **malloc Failures**: Field telemetry occasionally shows malloc returning NULL even though heap usage appears to be within limits.

4. **Hard Faults**: Some deployments report hard fault exceptions with imprecise fault addresses, making debugging difficult.

## Dependencies

- **arm-none-eabi-gcc**: Version 10.3.1 or later
- **newlib**: Version 3.3.0 (nano variant)
- **make**: GNU Make 4.3 or later
- **openocd**: For flashing and debugging (optional)

## Project Structure

```
embedded_project/
├── src/           - Application source files
├── include/       - Header files
├── startup/       - Startup code and vector table
├── syscalls/      - Newlib system call stubs
├── linker/        - Linker script
├── build/         - Build output directory
└── Makefile       - Build configuration
```

## License

Proprietary - Internal use only