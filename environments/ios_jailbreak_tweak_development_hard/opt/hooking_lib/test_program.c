#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Test program for hooking library functionality
 * This program exercises various hooked functions to verify
 * interception mechanisms work correctly
 */

int main(int argc, char *argv[]) {
    char buffer1[64];
    char buffer2[128];
    char small_buf[32];
    void *ptr1, *ptr2;
    
    printf("Starting hook library test program...\n");
    printf("Testing standard library function interception\n\n");
    
    // Test 1: Normal strcpy operation
    printf("Test 1: Normal strcpy with short string\n");
    strcpy(buffer1, "Hello World");
    printf("Result: %s\n\n", buffer1);
    
    // Test 2: strcpy with medium length string
    printf("Test 2: strcpy with medium string\n");
    strcpy(buffer2, "This is a medium length test string for hook validation");
    printf("Result: %s\n\n", buffer2);
    
    // Test 3: strcpy with long string - potential overflow trigger
    printf("Test 3: strcpy with long string (100+ chars)\n");
    char *long_string = "This is an extremely long test string designed to exceed typical buffer sizes and trigger potential vulnerabilities in the hooking mechanism implementation if proper bounds checking is not performed correctly";
    strcpy(buffer2, long_string);
    printf("Result: %s\n\n", buffer2);
    
    // Test 4: Testing with small buffer and long input - HIGH RISK
    printf("Test 4: strcpy to small buffer with oversized input\n");
    strcpy(small_buf, "This string is definitely longer than 32 bytes and should cause problems if not handled properly by hooks");
    printf("Result: %s\n\n", small_buf);
    
    // Test 5: Memory allocation hooks
    printf("Test 5: Testing malloc/free hooks\n");
    ptr1 = malloc(256);
    printf("Allocated ptr1: %p\n", ptr1);
    ptr2 = malloc(512);
    printf("Allocated ptr2: %p\n", ptr2);
    
    if (ptr1) free(ptr1);
    if (ptr2) free(ptr2);
    printf("Memory freed successfully\n\n");
    
    // Test 6: Multiple consecutive operations
    printf("Test 6: Rapid consecutive strcpy calls\n");
    for (int i = 0; i < 5; i++) {
        strcpy(buffer1, "Iteration test string");
        printf("Iteration %d complete\n", i);
    }
    
    printf("\nAll tests completed\n");
    return 0;
}