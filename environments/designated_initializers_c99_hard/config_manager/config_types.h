#ifndef CONFIG_TYPES_H
#define CONFIG_TYPES_H

#include <stdint.h>

/* System Configuration Structure */
typedef struct {
    int baud_rate;
    int data_bits;
    int stop_bits;
    int parity;
    int enabled;
} SystemConfig;

/* Network Configuration Structure */
typedef struct {
    char ip_address[16];
    char netmask[16];
    char gateway[16];
    int port;
    int dhcp_enabled;
} NetworkConfig;

/* Device Information Structure */
typedef struct {
    char device_id[32];
    char firmware_version[16];
    uint32_t serial_number;
    int active;
} DeviceInfo;

/* Sensor Configuration Structure */
typedef struct {
    int sensor_id;
    int max_value;
    int min_value;
    int max_temp;
    int sample_rate;
    int enabled;
} SensorConfig;

/* Serial Parameters Structure */
typedef struct {
    int baud;
    int data_bits;
    int stop_bits;
    int parity;
    int flow_control;
    int timeout;
} SerialParams;

/* Configuration Entry Structure */
typedef struct {
    int id;
    char name[32];
    int config_type;
    int priority;
} ConfigEntry;

/* Configuration Profile Structure */
typedef struct {
    char profile_name[32];
    int version;
    uint32_t flags;
} ConfigProfile;

/* Device Registry Structure */
typedef struct {
    int max_devices;
    int device_count;
    void* devices;
    ConfigProfile metadata;
} DeviceRegistry;

/* GPIO Configuration Structure */
typedef struct {
    int pin_number;
    int direction;
    int pull_mode;
    int interrupt_enabled;
    int voltage_mv;
} GPIOConfig;

/* Power Configuration Structure */
typedef struct {
    int input_mv;
    int output_mv;
    int current_ma;
    int max_power_mw;
    int regulator_enabled;
} PowerConfig;

/* Protocol Configuration Structure */
typedef struct {
    int version;
    int type;
    int timeout_ms;
    int retry_count;
    int buffer_size;
} ProtocolConfig;

/* Packet Header Structure */
typedef struct {
    uint8_t sync1;
    uint8_t sync2;
    uint16_t length;
    uint8_t type;
    uint8_t checksum;
} PacketHeader;

/* Communication Configuration Structure */
typedef struct {
    int baud_rate;
    int data_bits;
    int stop_bits;
    int parity;
    int rts_cts;
    int timeout_ms;
} CommConfig;

/* Protocol Handler Structure */
typedef struct {
    int protocol_id;
    char handler_name[32];
    int enabled;
} ProtocolHandler;

/* System Module Structure */
typedef struct {
    int module_id;
    char module_name[32];
    int priority;
    int auto_start;
} SystemModule;

/* Memory Configuration Structure */
typedef struct {
    uint32_t base_address;
    uint32_t size;
    int page_size;
    int write_protected;
} MemoryConfig;

/* Clock Configuration Structure */
typedef struct {
    uint32_t system_clock_hz;
    uint32_t peripheral_clock_hz;
    int divider;
    int pll_enabled;
    int external_osc;
} ClockConfig;

/* Boot Configuration Structure */
typedef struct {
    int boot_mode;
    int timeout_seconds;
    int safe_mode;
    char version_string[32];
} BootConfig;

/* Function declarations */
void init_system_config(void);
void init_network_config(void);
void init_device_config(void);
void init_sensor_config(void);
void init_serial_config(void);
void print_all_configs(void);
int validate_configs(void);
void reset_to_defaults(void);
uint32_t calculate_checksum(void);

#endif /* CONFIG_TYPES_H */