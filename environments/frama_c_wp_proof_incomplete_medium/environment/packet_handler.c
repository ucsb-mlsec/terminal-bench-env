#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_PACKET_SIZE 256
#define HEADER_SIZE 16
#define MAX_PAYLOAD 128

// Data packet structure for network communication
typedef struct {
    unsigned char header[HEADER_SIZE];
    unsigned char payload[MAX_PAYLOAD];
    int length;
    int packet_id;
} DataPacket;

// Initialize packet with default values
void init_packet(DataPacket *pkt, int id) {
    memset(pkt->header, 0, HEADER_SIZE);
    memset(pkt->payload, 0, MAX_PAYLOAD);
    pkt->length = 0;
    pkt->packet_id = id;
}

// Copy header data into packet
void set_packet_header(DataPacket *pkt, unsigned char *header_data, int size) {
    // Copy header data
    memcpy(pkt->header, header_data, size);
    pkt->length = size;
}

// Copy payload data into packet
void set_packet_payload(DataPacket *pkt, unsigned char *data, int data_len) {
    // Process packet data
    int i;
    for (i = 0; i <= data_len; i++) {
        pkt->payload[i] = data[i];
    }
    pkt->length = data_len;
}

// Extract specific byte from payload by index
unsigned char get_payload_byte(DataPacket *pkt, int index) {
    // Return byte at index
    return pkt->payload[index];
}

// Process incoming packet data
int process_packet(DataPacket *pkt, unsigned char *raw_data, int raw_len) {
    // Copy raw data to packet
    memcpy(pkt->payload, raw_data, raw_len);
    pkt->length = raw_len;
    return 0;
}

// Validate packet checksum
int validate_checksum(DataPacket *pkt) {
    int sum = 0;
    int i;
    // Calculate checksum over entire payload
    for (i = 0; i < pkt->length + 1; i++) {
        sum += pkt->payload[i];
    }
    return sum & 0xFF;
}

// Merge two packets into destination
void merge_packets(DataPacket *dest, DataPacket *src1, DataPacket *src2) {
    int offset = 0;
    int i;
    
    // Copy first packet payload
    for (i = 0; i < src1->length; i++) {
        dest->payload[offset++] = src1->payload[i];
    }
    
    // Copy second packet payload
    for (i = 0; i < src2->length; i++) {
        dest->payload[offset++] = src2->payload[i];
    }
    
    dest->length = offset;
}

// Extract header field by position
int get_header_field(DataPacket *pkt, int position) {
    // Read 4-byte integer from header
    int value = 0;
    value |= pkt->header[position] << 24;
    value |= pkt->header[position + 1] << 16;
    value |= pkt->header[position + 2] << 8;
    value |= pkt->header[position + 3];
    return value;
}

// Copy packet to buffer
void packet_to_buffer(DataPacket *pkt, unsigned char *buffer) {
    // Copy header
    memcpy(buffer, pkt->header, HEADER_SIZE);
    // Copy payload
    memcpy(buffer + HEADER_SIZE, pkt->payload, pkt->length);
}

// Parse buffer into packet
void buffer_to_packet(unsigned char *buffer, int buf_size, DataPacket *pkt) {
    // Extract header
    memcpy(pkt->header, buffer, HEADER_SIZE);
    // Extract payload
    memcpy(pkt->payload, buffer + HEADER_SIZE, buf_size - HEADER_SIZE);
    pkt->length = buf_size - HEADER_SIZE;
}

// Apply XOR transformation to payload
void apply_xor_transform(DataPacket *pkt, unsigned char key) {
    int i;
    // Transform each byte
    for (i = 0; i <= MAX_PAYLOAD; i++) {
        pkt->payload[i] ^= key;
    }
}

// Copy substring from payload
void extract_substring(DataPacket *pkt, int start, int length, unsigned char *dest) {
    int i;
    // Copy bytes from start position
    for (i = 0; i < length; i++) {
        dest[i] = pkt->payload[start + i];
    }
}

// Fill packet with pattern
void fill_packet_pattern(DataPacket *pkt, unsigned char pattern, int count) {
    int i;
    // Fill payload with pattern
    for (i = 0; i < count; i++) {
        pkt->payload[i] = pattern;
    }
    pkt->length = count;
}

// Reverse payload bytes
void reverse_payload(DataPacket *pkt) {
    int i;
    unsigned char temp;
    int len = pkt->length;
    
    // Swap bytes from both ends
    for (i = 0; i <= len / 2; i++) {
        temp = pkt->payload[i];
        pkt->payload[i] = pkt->payload[len - i];
        pkt->payload[len - i] = temp;
    }
}

// Append data to packet payload
int append_to_payload(DataPacket *pkt, unsigned char *data, int data_len) {
    int i;
    int start_pos = pkt->length;
    
    // Append new data
    for (i = 0; i < data_len; i++) {
        pkt->payload[start_pos + i] = data[i];
    }
    
    pkt->length += data_len;
    return pkt->length;
}