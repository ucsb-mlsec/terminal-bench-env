#include <stdio.h>
#include "message_types.h"

int main() {
    size_t sensor_size = sizeof(SensorDataPacket);
    size_t control_size = sizeof(ControlCommandPacket);
    size_t status_size = sizeof(StatusResponsePacket);
    
    int all_passed = 1;
    
    printf("Struct Size Verification Results:\n");
    printf("==================================\n\n");
    
    printf("SensorDataPacket: actual=%zu bytes, expected=13 bytes [%s]\n", 
           sensor_size, 
           (sensor_size == 13) ? "PASS" : "FAIL");
    if (sensor_size != 13) all_passed = 0;
    
    printf("ControlCommandPacket: actual=%zu bytes, expected=9 bytes [%s]\n", 
           control_size, 
           (control_size == 9) ? "PASS" : "FAIL");
    if (control_size != 9) all_passed = 0;
    
    printf("StatusResponsePacket: actual=%zu bytes, expected=17 bytes [%s]\n", 
           status_size, 
           (status_size == 17) ? "PASS" : "FAIL");
    if (status_size != 17) all_passed = 0;
    
    printf("\n==================================\n");
    if (all_passed) {
        printf("Result: ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("Result: SOME TESTS FAILED\n");
        return 1;
    }
}