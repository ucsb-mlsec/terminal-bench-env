#include <stdio.h>
#include <stdlib.h>

extern int compute(int);

int main() {
    int result = compute(10);
    printf("Legacy app result: %d\n", result);
    
    // v1.0 behavior: doubles the input (10 * 2 = 20)
    if (result == 20) {
        printf("PASS: Got expected v1.0 result\n");
        return 0;
    } else {
        printf("FAIL: Expected 20, got %d\n", result);
        return 1;
    }
}