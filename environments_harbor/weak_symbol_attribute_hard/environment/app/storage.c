#include <stdio.h>

void storage_init(void) {
    printf("Storage module initialized\n");
}

int storage_process(const char* data) {
    printf("Processing storage: %s\n", data);
    return 1;
}