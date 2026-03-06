# Configuration Validator Library

## Description

This library provides a simple and efficient validation function for configuration strings. It is designed to validate configuration entries that follow the standard `key=value` format commonly used in configuration files.

## Usage

The library exposes a single function:

```c
int validate_config(const uint8_t *data, size_t size);
```

### Parameters

- `data`: Pointer to the configuration string (as binary data)
- `size`: Size of the input data in bytes

### Return Values

- `0`: Configuration string is valid (contains '=' delimiter)
- `-1`: Configuration string is invalid (missing delimiter or invalid size)

### Size Constraints

Input size must be between 1 and 512 bytes. Inputs outside this range will be rejected.

## Example

```c
#include "validator.h"

const char *config = "key=value";
int result = validate_config((uint8_t*)config, 9);
if (result == 0) {
    printf("Valid configuration\n");
}
```

## Integration

Simply include the header file and link against the compiled validator object file in your project.