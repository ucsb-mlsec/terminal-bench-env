#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "config_types.h"

/* Protocol handler implementation for embedded systems
 * Legacy code - uses old-style struct initialization
 */

// Initialize default protocol configuration
ProtocolConfig init_protocol_config(void) {
    ProtocolConfig config = {1, 2, 1000, 5, 64};
    return config;
}

// Create a new packet header with default values
PacketHeader create_packet_header(void) {
    PacketHeader header = {0xAA, 0x55, 0, 0, 0};
    return header;
}

// Setup communication parameters
CommConfig setup_communication(void) {
    CommConfig comm = {115200, 8, 1, 0, 1, 100};
    return comm;
}

// Create an acknowledgment packet header
PacketHeader create_ack_header(uint8_t seq) {
    PacketHeader header = {0xAA, 0x55, seq, 0x01, 0};
    return header;
}

// Protocol handler function prototypes
int handle_config_request(const uint8_t *data, size_t len);
int handle_status_query(const uint8_t *data, size_t len);
int handle_reset_command(const uint8_t *data, size_t len);

// Protocol handlers array - old style initialization
ProtocolHandler protocol_handlers[] = {
    {0x01, "CONFIG_REQ", handle_config_request, 1},
    {0x02, "STATUS_QRY", handle_status_query, 1},
    {0x03, "RESET_CMD", handle_reset_command, 1}
};

const size_t num_handlers = sizeof(protocol_handlers) / sizeof(protocol_handlers[0]);

// Implementation of handler functions
int handle_config_request(const uint8_t *data, size_t len) {
    printf("Handling configuration request, length: %zu\n", len);
    if (len < 4) {
        return -1;
    }
    ProtocolConfig config = init_protocol_config();
    printf("  Protocol version: %u.%u\n", config.version_major, config.version_minor);
    return 0;
}

int handle_status_query(const uint8_t *data, size_t len) {
    printf("Handling status query, length: %zu\n", len);
    (void)data;
    return 0;
}

int handle_reset_command(const uint8_t *data, size_t len) {
    printf("Handling reset command, length: %zu\n", len);
    (void)data;
    return 0;
}

// Validate packet header integrity
int validate_header(const PacketHeader *header) {
    if (header == NULL) {
        return 0;
    }
    
    if (header->sync1 != 0xAA || header->sync2 != 0x55) {
        printf("Invalid sync bytes: 0x%02X 0x%02X\n", header->sync1, header->sync2);
        return 0;
    }
    
    printf("Header validation passed - seq: %u, type: 0x%02X\n", 
           header->sequence, header->packet_type);
    return 1;
}

// Process incoming packet
int process_packet(const uint8_t *buffer, size_t length) {
    if (buffer == NULL || length < sizeof(PacketHeader)) {
        printf("Invalid packet: buffer=%p, length=%zu\n", (void*)buffer, length);
        return -1;
    }
    
    PacketHeader header;
    memcpy(&header, buffer, sizeof(PacketHeader));
    
    if (!validate_header(&header)) {
        return -1;
    }
    
    // Find appropriate handler
    for (size_t i = 0; i < num_handlers; i++) {
        if (protocol_handlers[i].command_id == header.packet_type) {
            if (protocol_handlers[i].enabled) {
                printf("Dispatching to handler: %s\n", protocol_handlers[i].name);
                return protocol_handlers[i].handler(
                    buffer + sizeof(PacketHeader),
                    length - sizeof(PacketHeader)
                );
            } else {
                printf("Handler %s is disabled\n", protocol_handlers[i].name);
                return -2;
            }
        }
    }
    
    printf("No handler found for packet type 0x%02X\n", header.packet_type);
    return -3;
}

// Send response packet
int send_response(uint8_t packet_type, uint8_t sequence, const uint8_t *data, size_t data_len) {
    PacketHeader response_header = {0xAA, 0x55, sequence, packet_type, (uint8_t)data_len};
    
    printf("Sending response: type=0x%02X, seq=%u, len=%zu\n", 
           packet_type, sequence, data_len);
    
    // Simulate sending header
    printf("  Header: [0x%02X 0x%02X %u 0x%02X %u]\n",
           response_header.sync1, response_header.sync2,
           response_header.sequence, response_header.packet_type,
           response_header.length);
    
    // Simulate sending data
    if (data != NULL && data_len > 0) {
        printf("  Data: ");
        for (size_t i = 0; i < data_len && i < 16; i++) {
            printf("%02X ", data[i]);
        }
        printf("\n");
    }
    
    return 0;
}

// Main function for testing
int main(void) {
    printf("Protocol Handler Test Suite\n");
    printf("============================\n\n");
    
    // Test 1: Initialize protocol configuration
    printf("Test 1: Protocol Configuration\n");
    ProtocolConfig config = init_protocol_config();
    printf("Protocol Version: %u.%u\n", config.version_major, config.version_minor);
    printf("Timeout: %u ms, Retries: %u, Buffer size: %u\n\n",
           config.timeout_ms, config.max_retries, config.buffer_size);
    
    // Test 2: Communication setup
    printf("Test 2: Communication Setup\n");
    CommConfig comm = setup_communication();
    printf("Baud rate: %lu, Data bits: %u, Stop bits: %u\n",
           comm.baud_rate, comm.data_bits, comm.stop_bits);
    printf("Parity: %u, Flow control: %u, Timeout: %u ms\n\n",
           comm.parity, comm.flow_control, comm.timeout_ms);
    
    // Test 3: Create and validate headers
    printf("Test 3: Header Creation and Validation\n");
    PacketHeader test_header = create_packet_header();
    validate_header(&test_header);
    printf("\n");
    
    // Test 4: Process test packets
    printf("Test 4: Packet Processing\n");
    uint8_t test_packet1[] = {0xAA, 0x55, 0x01, 0x01, 0x04, 0xDE, 0xAD, 0xBE, 0xEF};
    process_packet(test_packet1, sizeof(test_packet1));
    printf("\n");
    
    uint8_t test_packet2[] = {0xAA, 0x55, 0x02, 0x02, 0x00};
    process_packet(test_packet2, sizeof(test_packet2));
    printf("\n");
    
    // Test 5: Send responses
    printf("Test 5: Response Transmission\n");
    uint8_t response_data[] = {0x01, 0x02, 0x03, 0x04};
    send_response(0x81, 0x01, response_data, sizeof(response_data));
    printf("\n");
    
    PacketHeader ack = create_ack_header(0x05);
    send_response(ack.packet_type, ack.sequence, NULL, 0);
    
    printf("\nAll tests completed.\n");
    return 0;
}