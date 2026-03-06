#include "checksum.h"
#include <string.h>

int validate_checksum(const char* data, int expected_checksum) {
    if (data == NULL) {
        return 0;
    }
    
    int calculated_checksum = 0;
    int len = strlen(data);
    
    for (int i = 0; i < len; i++) {
        calculated_checksum += (unsigned char)data[i];
    }
    
    calculated_checksum = calculated_checksum % 256;
    
    return (calculated_checksum == expected_checksum) ? 1 : 0;
}