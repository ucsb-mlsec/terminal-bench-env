#include <stdio.h>

void network_init(void) {
    printf("Network module initialized\n");
}

int network_process(const char* data) {
    printf("Processing network: %s\n", data);
    return 1;
}