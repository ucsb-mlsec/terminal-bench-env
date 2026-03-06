# LogLib - Simple C++ Logging Library

LogLib is a lightweight C++ logging library that provides simple and efficient logging capabilities for your applications. It supports multiple log levels (INFO, WARNING, ERROR) and offers both a library API for embedding in applications and a command-line tool for standalone use. The library is designed to be easy to integrate and use with minimal overhead.

## Features

- Multiple log levels: INFO, WARNING, ERROR
- Thread-safe logging operations
- Shared library for easy integration
- Command-line utility for quick logging tasks
- Simple, intuitive API

## Building with CMake

To build LogLib from source:

```bash
mkdir build
cd build
cmake ..
make
```

## Installation

After building, you can create a distributable package using CPack:

```bash
cd build
cpack
```

This will generate a TGZ package that can be distributed and installed on target systems.

To install from the package:

```bash
tar -xzf LogLib-*.tar.gz
cd LogLib-*
sudo cp -r * /usr/local/
```

## Usage

### Library API

Include the header and link against the library in your C++ projects:

```cpp
#include <logger.h>

int main() {
    log_info("Application started successfully");
    log_warning("This is a warning message");
    log_error("An error occurred");
    return 0;
}
```

Compile your application:

```bash
g++ myapp.cpp -llogger -o myapp
```

### Command-line Tool

The `logutil` command-line tool allows you to log messages directly from the shell:

```bash
logutil info "System initialization complete"
logutil warning "Low disk space detected"
logutil error "Failed to connect to database"
```

## License

This project is provided as-is for educational and commercial use.