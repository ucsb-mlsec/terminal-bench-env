#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    printf("Build system test\n");
    printf("Runtime environment compatibility check\n");
    
    if (argc > 1) {
        printf("Arguments provided: %d\n", argc - 1);
    }
    
    int test_value = 42;
    printf("Test value: %d\n", test_value);
    
    return 0;
}