#include <stdio.h>
#include <stdint.h>
#include "device_a.h"
#include "device_b.h"
#include "device_c.h"

/*
 * Hardware Device Structure Size Verification
 * 
 * This program verifies that device register structures match the exact
 * memory-mapped I/O layouts specified by the hardware documentation.
 * 
 * Critical Requirements:
 * - Device A: Tightly packed, no padding - MUST be exactly 24 bytes
 * - Device B: 2-byte alignment boundaries - MUST be exactly 28 bytes
 * - Device C: 4-byte alignment with specific field ordering - MUST be exactly 32 bytes
 * 
 * Any mismatch will cause data corruption when communicating with hardware
 * peripherals through memory-mapped registers. Structure packing pragmas
 * must be configured correctly in each device header file.
 */

int main(void) {
    int all_passed = 1;
    
    printf("=== Hardware Device Structure Size Verification ===\n\n");
    
    /* Device A Verification
     * Hardware spec requires tightly packed structure with no padding
     * between fields. This is critical for direct memory mapping to
     * hardware registers at the exact byte offsets specified.
     */
    size_t device_a_size = sizeof(DeviceARegisters);
    printf("Device A (Tightly Packed):\n");
    printf("  Current size: %zu bytes\n", device_a_size);
    printf("  Required size: 24 bytes\n");
    if (device_a_size == 24) {
        printf("  Status: PASS\n");
    } else {
        printf("  Status: FAIL - Size mismatch will cause register corruption!\n");
        all_passed = 0;
    }
    printf("\n");
    
    /* Device B Verification
     * Hardware spec requires 2-byte alignment boundaries for optimal
     * bus access performance. Fields must align on 2-byte boundaries
     * with potential padding between fields.
     */
    size_t device_b_size = sizeof(DeviceBRegisters);
    printf("Device B (2-byte Alignment):\n");
    printf("  Current size: %zu bytes\n", device_b_size);
    printf("  Required size: 28 bytes\n");
    if (device_b_size == 28) {
        printf("  Status: PASS\n");
    } else {
        printf("  Status: FAIL - Alignment mismatch will cause bus errors!\n");
        all_passed = 0;
    }
    printf("\n");
    
    /* Device C Verification
     * Hardware spec requires 4-byte alignment with specific field
     * ordering to match the physical register layout. This device
     * uses 32-bit bus transactions and requires natural alignment.
     */
    size_t device_c_size = sizeof(DeviceCRegisters);
    printf("Device C (4-byte Alignment):\n");
    printf("  Current size: %zu bytes\n", device_c_size);
    printf("  Required size: 32 bytes\n");
    if (device_c_size == 32) {
        printf("  Status: PASS\n");
    } else {
        printf("  Status: FAIL - Alignment error will cause data corruption!\n");
        all_passed = 0;
    }
    printf("\n");
    
    /* Overall Result */
    printf("=== Overall Result ===\n");
    if (all_passed) {
        printf("ALL TESTS PASSED - Structure layouts match hardware specifications\n");
        return 0;
    } else {
        printf("TESTS FAILED - Structure packing pragmas need correction\n");
        printf("Review #pragma pack directives in device header files\n");
        return 1;
    }
}