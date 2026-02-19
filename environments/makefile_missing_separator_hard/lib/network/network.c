#include <stdio.h>
#include "network.h"

void init_network(void) {
    printf("Network module initialized\n");
}

int connect_socket(char *host, int port) {
    printf("Connecting to %s:%d\n", host, port);
    return 0;
}