#ifndef DEVICE_B_H
#define DEVICE_B_H

#include <stdint.h>

/* Device B Hardware Communication Interface
 * Hardware requires 2-byte alignment boundaries
 * Total register size must be exactly 28 bytes
 */

#pragma pack(push, 1)

struct DeviceBRegisters {
    uint16_t mode;
    uint8_t flags;
    uint8_t reserved;
    uint32_t config_register;
    uint16_t status_word;
    uint64_t data_buffer;
    uint32_t checksum;
};

#pragma pack(pop)

#endif /* DEVICE_B_H */