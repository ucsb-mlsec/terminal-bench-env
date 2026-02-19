#include <stdio.h>
#include <stdint.h>
#include <string.h>

struct NetworkPacket {
    uint8_t version;
    uint8_t type;
    uint32_t sequence;
    uint16_t length;
    uint8_t flags;
    uint16_t checksum;
    uint32_t reserved;
};

int main() {
    FILE *fp;
    struct NetworkPacket packet;
    
    fp = fopen("/workspace/test_packet.bin", "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Unable to open test_packet.bin\n");
        return 1;
    }
    
    size_t bytes_read = fread(&packet, 1, sizeof(struct NetworkPacket), fp);
    if (bytes_read != sizeof(struct NetworkPacket)) {
        fprintf(stderr, "Error: Failed to read complete packet\n");
        fclose(fp);
        return 1;
    }
    
    printf("SEQUENCE_NUMBER=%u\n", packet.sequence);
    printf("PAYLOAD_LENGTH=%u\n", packet.length);
    printf("CHECKSUM=0x%04X\n", packet.checksum);
    
    fclose(fp);
    return 0;
}