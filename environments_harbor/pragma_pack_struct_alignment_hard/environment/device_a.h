#ifndef DEVICE_A_H
#define DEVICE_A_H

#include <stdint.h>

/* Device A Hardware Communication Registers */
/* Hardware mapped I/O structure for peripheral A */

#pragma pack(push, 4)

typedef struct {
    uint8_t status;
    uint32_t data_register;
    uint8_t control;
    uint64_t timestamp;
    uint16_t counter;
    uint32_t address;
} DeviceARegisters;

#pragma pack(pop)

/* Device A base address */
#define DEVICE_A_BASE_ADDR 0x40000000

/* Status register bit masks */
#define DEVICE_A_STATUS_READY   0x01
#define DEVICE_A_STATUS_ERROR   0x02
#define DEVICE_A_STATUS_BUSY    0x04

/* Control register bit masks */
#define DEVICE_A_CTRL_ENABLE    0x01
#define DEVICE_A_CTRL_RESET     0x02
#define DEVICE_A_CTRL_IRQ_EN    0x04

#endif /* DEVICE_A_H */