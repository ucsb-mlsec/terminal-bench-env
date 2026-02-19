#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", argv[1]);
        return 1;
    }

    uint32_t checksum = 0;
    int byte;

    // Read file one byte at a time (deliberately slow)
    while ((byte = fgetc(file)) != EOF) {
        checksum += (uint32_t)byte;
    }

    fclose(file);

    printf("0x%08x\n", checksum);

    return 0;
}