#ifndef MESSAGE_TYPES_H
#define MESSAGE_TYPES_H

#include <stdint.h>

struct SensorDataPacket {
    uint8_t sensor_id;
    uint16_t temperature;
    uint16_t humidity;
    uint32_t timestamp;
    uint16_t battery_voltage;
};

struct ControlCommandPacket {
    uint8_t command_id;
    uint32_t parameter;
    uint16_t checksum;
    uint8_t flags;
};

struct StatusResponsePacket {
    uint8_t status_code;
    uint32_t device_id;
    uint64_t uptime;
    uint16_t error_count;
    uint8_t mode;
};

#endif