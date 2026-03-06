#ifndef DEVICE_C_H
#define DEVICE_C_H

#include <stdint.h>

/* Device C Hardware Communication Interface
 * Requires 4-byte alignment for memory-mapped I/O
 * Hardware register block must be exactly 32 bytes
 */

#pragma pack(push, 8)

typedef struct {
    uint32_t command;
    uint32_t param1;
    uint16_t param2;
    uint8_t flag1;
    uint8_t flag2;
    uint64_t result;
    uint32_t error_code;
    uint32_t reserved;
} DeviceCRegisters;

#pragma pack(pop)

#endif /* DEVICE_C_H */